"""Build a bundle whose cached decoder omits cross-attention K/V outputs.

Keep the source bundle read-only. In the destination bundle, clear outputs
1 and 2 of four MultiHeadAttention nodes and remove the eight corresponding
graph outputs. Copy the encoder and first-step decoder unchanged.
Run six integrity checks and exit nonzero if any fails. Save the ORT-optimized
graph under <out>/intermediate/ and metadata to <out>/xkv_manifest.json.
Usage: python donut_make_xkv_bundle.py --src <fused-bundle> --out <new-dir>
"""

import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper

EXPECT = {'encoder.onnx': 'fa5c2951233a063e3ffc498dfc1c21661757ba7df1ed0ee093e27d6d9768169b',
          'decoder.onnx': 'faa65092ca096c896628f72fb65ef0176fd09bc8d06503222f204b310b8deaec',
          'decoder_with_past.onnx': '8a2e889b90d978a64a12407e70cc5584a08723ddcf5fb8ee4022b585e106362f'}
CROSS_OUT = {f'present_{L}_{k}' for L in range(4) for k in ('cross_k', 'cross_v')}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    dst, inter = out / 'bundle', out / 'intermediate'

    report = {'src': str(src), 'dst': str(dst), 'script_sha256': sha(Path(__file__)), 'checks': {}}
    for name, expected in EXPECT.items():
        s = sha(src / name)
        assert s == expected, ('Source file SHA256 mismatch', name, s)
        report.setdefault('src_sha256', {})[name] = s
    dst.mkdir(parents=True, exist_ok=False)
    inter.mkdir(parents=True, exist_ok=False)

    # Copy the encoder and first-step decoder unchanged.
    for name in ('encoder.onnx', 'decoder.onnx'):
        shutil.copy2(src / name, dst / name)
    copied = {n: sha(dst / n) for n in ('encoder.onnx', 'decoder.onnx')}
    report['checks']['4_copies_identical'] = all(copied[n] == report['src_sha256'][n] for n in copied)

    # Clear cross-attention K/V outputs and remove the eight matching graph outputs.
    orig = onnx.load(src / 'decoder_with_past.onnx')
    m = onnx.load(src / 'decoder_with_past.onnx')
    changed = []
    for n in m.graph.node:
        if n.op_type == 'MultiHeadAttention':
            assert len(n.output) == 3 and set(n.output[1:]) <= CROSS_OUT, (n.name, list(n.output))
            changed.append((n.name, list(n.output[1:])))
            n.output[1] = ''
            n.output[2] = ''
    assert len(changed) == 4, changed
    kept = [o for o in m.graph.output if o.name not in CROSS_OUT]
    assert len(kept) == 9, [o.name for o in kept]
    del m.graph.output[:]
    m.graph.output.extend(kept)
    onnx.checker.check_model(m)
    onnx.save(m, dst / 'decoder_with_past.onnx')
    del m
    report['changed_mha'] = changed
    report['dst_sha256'] = {n: sha(dst / n) for n in EXPECT}

    # Check 1: only outputs 1 and 2 of four MHA nodes may change to empty strings.
    new = onnx.load(dst / 'decoder_with_past.onnx')
    ok1 = len(orig.graph.node) == len(new.graph.node)
    diffs = []
    for a, b in zip(orig.graph.node, new.graph.node):
        same_core = (a.op_type, a.domain, list(a.input), list(a.attribute)) == (b.op_type, b.domain, list(b.input), list(b.attribute))
        if not same_core:
            ok1 = False
            diffs.append(('core', a.name))
        if list(a.output) != list(b.output):
            if not (a.op_type == 'MultiHeadAttention' and list(b.output) == [a.output[0], '', '']):
                ok1 = False
            diffs.append(('output', a.name, list(b.output)))
    report['checks']['1_nodes_identical_except_mha_outputs'] = ok1 and len([d for d in diffs if d[0] == 'output']) == 4
    report['node_count'] = len(new.graph.node)

    # Check 2: initializer bytes must match.
    ia = {t.name: numpy_helper.to_array(t).tobytes() for t in orig.graph.initializer}
    ib = {t.name: numpy_helper.to_array(t).tobytes() for t in new.graph.initializer}
    report['checks']['2_initializers_identical'] = ia == ib
    report['initializer_count'] = len(ib)
    del ia, ib

    # Check 3: opset declarations and graph inputs must match.
    oa = sorted((o.domain, o.version) for o in orig.opset_import)
    ob = sorted((o.domain, o.version) for o in new.opset_import)
    report['checks']['3_opsets_and_inputs_identical'] = oa == ob and [i.name for i in orig.graph.input] == [i.name for i in new.graph.input]
    report['graph_inputs'] = len(new.graph.input)
    del orig, new

    # Check 5: ORT loads nine outputs; each optimized MHA has one nonempty output.
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt_path = inter / 'decoder_with_past.ort_optimized.onnx'
    so.optimized_model_filepath = str(opt_path)
    sess = ort.InferenceSession(str(dst / 'decoder_with_past.onnx'), so, providers=['CPUExecutionProvider'])
    outs = [o.name for o in sess.get_outputs()]
    expect_outs = ['logits'] + [f'present_{L}_{k}' for L in range(4) for k in ('self_k', 'self_v')]
    report['ort_version'] = ort.__version__
    report['ort_outputs'] = outs
    report['ort_inputs'] = len(sess.get_inputs())
    del sess
    mo = onnx.load(str(opt_path), load_external_data=False)
    report['ort_optimized_node_count'] = len(mo.graph.node)
    report['ort_optimized_ops'] = dict(collections.Counter(n.op_type for n in mo.graph.node).most_common())
    report['ort_optimized_mha_outputs'] = [[o for o in n.output] for n in mo.graph.node if n.op_type == 'MultiHeadAttention']
    report['checks']['5_ort_loads_9_outputs'] = outs == expect_outs and len(report['ort_optimized_mha_outputs']) == 4 and \
        all(sum(1 for o in os_ if o) == 1 for os_ in report['ort_optimized_mha_outputs'])
    del mo

    # Check 6: the source bundle must remain unchanged.
    report['checks']['6_src_unchanged'] = all(sha(src / n) == report['src_sha256'][n] for n in EXPECT)

    report['graphs'] = {n: {'bytes': (dst / n).stat().st_size, 'sha256': report['dst_sha256'][n]} for n in EXPECT}
    (out / 'xkv_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ('checks', 'node_count', 'initializer_count', 'graph_inputs', 'ort_version',
                                             'ort_inputs', 'ort_outputs', 'ort_optimized_node_count',
                                             'ort_optimized_mha_outputs', 'dst_sha256')}, ensure_ascii=False, indent=1))
    print('ORT_OPS', report['ort_optimized_ops'])
    if not all(report['checks'].values()):
        print('XKV_FAIL')
        sys.exit(1)
    print('XKV_PASS')


if __name__ == '__main__':
    main()

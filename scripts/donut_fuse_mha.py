"""Fuse the cached decoder with the ONNX Runtime BART optimizer and onnxsim.

Copy the encoder and first-step decoder unchanged. Run each --step in a
separate process, optionally guarded by the memory watchdog:
  fuse      Use model_type="bart", num_heads=16, hidden_size=1024, and
            FusionOptions("bart") with use_multi_head_attention enabled.
            Write intermediate/decoder_with_past.fused.onnx under --out.
  simplify  Simplify the fused graph into bundle/decoder_with_past.onnx.
  finalize  Copy the other graphs, verify SHA256 and onnx.checker, inspect
            operator counts and MultiHeadAttention Q/K/V sources, compare
            the raw and fused cached decoders on one real input, and write
            fuse_manifest.json with the same graphs schema as the exporter.
"""

import argparse
import collections
import hashlib
import importlib.metadata as metadata
import json
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GRAPHS = ('encoder.onnx', 'decoder.onnx', 'decoder_with_past.onnx')
RAW_SHA = {'encoder.onnx': 'fa5c2951233a063e3ffc498dfc1c21661757ba7df1ed0ee093e27d6d9768169b',
           'decoder.onnx': 'faa65092ca096c896628f72fb65ef0176fd09bc8d06503222f204b310b8deaec',
           'decoder_with_past.onnx': 'cd447f6b0f5c733eb31d6585c173d5a6c0a036a5f8fefcf7bc7f3c1565cc503b'}
KV_KINDS = ('self_k', 'self_v', 'cross_k', 'cross_v')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def maxrss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def op_counts(model):
    c = collections.Counter((n.domain + ':' if n.domain else '') + n.op_type for n in model.graph.node)
    return dict(c.most_common())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True, choices=('fuse', 'simplify', 'finalize'))
    parser.add_argument('--raw-bundle', required=True, help='Raw three-graph bundle from the export step')
    parser.add_argument('--out', required=True)
    parser.add_argument('--model-dir', default=None, help='finalize: checkpoint directory containing the processor')
    parser.add_argument('--manifest', default=None, help='finalize: CORD manifest.json')
    parser.add_argument('--extra-path', default=None)
    args = parser.parse_args()
    if args.extra_path:
        sys.path.insert(0, args.extra_path)

    import onnx

    raw = Path(args.raw_bundle)
    out = Path(args.out)
    bundle, inter = out / 'bundle', out / 'intermediate'
    bundle.mkdir(parents=True, exist_ok=True)
    inter.mkdir(parents=True, exist_ok=True)
    fused_path = inter / 'decoder_with_past.fused.onnx'
    final_path = bundle / 'decoder_with_past.onnx'
    report = {'step': args.step, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': digest(Path(__file__)), 'python': sys.version.split()[0],
              'versions': {k: version(k) for k in ('onnx', 'onnxruntime', 'onnx-simplifier', 'numpy', 'torch',
                                                   'transformers')}}
    step_file = out / f'fuse_step_{args.step}.json'

    def save():
        step_file.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    if args.step == 'fuse':
        from onnxruntime.transformers.fusion_options import FusionOptions
        from onnxruntime.transformers.optimizer import optimize_model
        src = raw / 'decoder_with_past.onnx'
        assert digest(src) == RAW_SHA['decoder_with_past.onnx']
        assert not fused_path.exists()
        options = FusionOptions('bart')
        options.use_multi_head_attention = True
        t0 = time.perf_counter()
        m = optimize_model(str(src), model_type='bart', num_heads=16, hidden_size=1024,
                           optimization_options=options)
        stats = m.get_fused_operator_statistics()
        m.save_model_to_file(str(fused_path), use_external_data_format=False)
        report.update({'source': str(src), 'opt_level': None,
                       'fusion_options': {k: v for k, v in vars(options).items() if not k.startswith('_')},
                       'fused_operator_statistics': stats, 'output': str(fused_path),
                       'bytes': fused_path.stat().st_size, 'sha256': digest(fused_path),
                       'elapsed_s': round(time.perf_counter() - t0, 2), 'maxrss_mib': maxrss_mib()})
        save()
        print('FUSED', {k: report[k] for k in ('bytes', 'elapsed_s', 'maxrss_mib')}, 'stats', stats, flush=True)
        return

    if args.step == 'simplify':
        import onnxsim
        assert fused_path.exists() and not final_path.exists()
        t0 = time.perf_counter()
        model = onnx.load(str(fused_path))
        before = len(model.graph.node)
        names_in = [i.name for i in model.graph.input]
        names_out = [o.name for o in model.graph.output]
        sim, check = onnxsim.simplify(model)
        assert [i.name for i in sim.graph.input] == names_in, 'Input names changed'
        assert [o.name for o in sim.graph.output] == names_out, 'Output names changed'
        onnx.save(sim, str(final_path))
        report.update({'nodes_before': before, 'nodes_after': len(sim.graph.node), 'onnxsim_check': bool(check),
                       'output': str(final_path), 'bytes': final_path.stat().st_size, 'sha256': digest(final_path),
                       'elapsed_s': round(time.perf_counter() - t0, 2), 'maxrss_mib': maxrss_mib()})
        save()
        print('SIMPLIFIED', {k: report[k] for k in ('nodes_before', 'nodes_after', 'onnxsim_check', 'bytes',
                                                     'elapsed_s', 'maxrss_mib')}, flush=True)
        assert check, 'onnxsim validation failed'
        return

    # ---------- finalize ----------
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from transformers import DonutProcessor

    for g in ('encoder.onnx', 'decoder.onnx'):
        dst = bundle / g
        if not dst.exists():
            shutil.copyfile(raw / g, dst)
        assert digest(dst) == RAW_SHA[g], ('Copied graph SHA256 mismatch', g)

    graphs, structure = {}, {}
    for g in GRAPHS:
        p = bundle / g
        onnx.checker.check_model(str(p))
        m = onnx.load(str(p))
        graphs[g] = {'bytes': p.stat().st_size, 'sha256': digest(p), 'n_nodes': len(m.graph.node),
                     'opset_imports': {o.domain or 'ai.onnx': o.version for o in m.opset_import},
                     'top_ops': sorted(op_counts(m).items(), key=lambda kv: -kv[1])[:15]}
        if g == 'decoder_with_past.onnx':
            graph_inputs = {i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
                            for i in m.graph.input}
            producer = {o: n for n in m.graph.node for o in n.output}
            mha = []
            for n in m.graph.node:
                if n.op_type != 'MultiHeadAttention':
                    continue
                ins = list(n.input)
                k_name = ins[1] if len(ins) > 1 else ''
                v_name = ins[2] if len(ins) > 2 else ''
                mha.append({'name': n.name, 'inputs': ins, 'outputs': list(n.output),
                            'key_from': k_name if k_name in graph_inputs else
                            (producer[k_name].op_type if k_name in producer else k_name),
                            'key_shape': graph_inputs.get(k_name), 'value_shape': graph_inputs.get(v_name),
                            'attributes': {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}})
            structure['fused_ops'] = op_counts(m)
            structure['mha'] = mha
            structure['raw_ops'] = op_counts(onnx.load(str(raw / g)))
        del m

    # Compare one raw/fused cached-decoder step on the same real input.
    processor = DonutProcessor.from_pretrained(str(args.model_dir))
    tok = processor.tokenizer
    item = json.loads(Path(args.manifest).read_text())['images'][0]
    assert digest(item['path']) == item['sha256']
    with Image.open(item['path']) as im:
        pv = processor(im.convert('RGB'), return_tensors='np').pixel_values.astype(np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    enc = ort.InferenceSession(str(raw / 'encoder.onnx'), so, providers=['CPUExecutionProvider'])
    enc_hs = enc.run(None, {'image': pv})[0]
    del enc
    dec = ort.InferenceSession(str(raw / 'decoder.onnx'), so, providers=['CPUExecutionProvider'])
    names = [o.name for o in dec.get_outputs()]
    prompt = np.array([[57579]], dtype=np.int64)
    first = dict(zip(names, dec.run(None, {'input_ids': prompt, 'attention_mask': np.ones((1, 1), np.int64),
                                           'encoder_hidden_states': enc_hs})))
    del dec
    ll = first['logits'][:, -1, :].copy()
    ll[:, tok.unk_token_id] = -np.inf
    nxt = ll.argmax(-1).reshape(1, 1).astype(np.int64)
    feed = {'input_ids': nxt, 'attention_mask': np.ones((1, 2), np.int64)}
    for L in range(4):
        for k in KV_KINDS:
            feed[f'past_{L}_{k}'] = first[f'present_{L}_{k}']
    outs = {}
    for tag, path in (('raw', raw / 'decoder_with_past.onnx'), ('fused', final_path)):
        s = ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])
        outs[tag] = dict(zip([o.name for o in s.get_outputs()], s.run(None, feed)))
        del s
    a, b = outs['raw']['logits'][:, -1, :], outs['fused']['logits'][:, -1, :]
    a2, b2 = a.copy(), b.copy()
    a2[:, tok.unk_token_id] = -np.inf
    b2[:, tok.unk_token_id] = -np.inf
    single = {'image': item['file'], 'first_token': int(nxt[0, 0]),
              'logits_max_abs_diff': float(np.abs(a - b).max()),
              'next_token_raw': int(a2.argmax()), 'next_token_fused': int(b2.argmax()),
              'present_self_max_abs_diff': max(float(np.abs(outs['raw'][f'present_{L}_{k}'] -
                                                            outs['fused'][f'present_{L}_{k}']).max())
                                               for L in range(4) for k in ('self_k', 'self_v')),
              'fused_outputs': sorted(outs['fused'])}
    single['next_token_identical'] = single['next_token_raw'] == single['next_token_fused']
    report.update({'graphs': graphs, 'structure': structure, 'single_step': single, 'maxrss_mib': maxrss_mib()})
    save()
    steps = {s: json.loads((out / f'fuse_step_{s}.json').read_text()) for s in ('fuse', 'simplify')}
    merged = {'created_utc': report['created_utc'], 'raw_bundle': str(raw), 'steps': steps,
              'graphs': graphs, 'structure': structure, 'single_step': single,
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    (out / 'fuse_manifest.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2, default=str))
    print('FINALIZED', {g: (graphs[g]['n_nodes'], graphs[g]['bytes']) for g in GRAPHS}, flush=True)
    print('MHA', [(x['name'], x['key_from'], x['key_shape']) for x in structure['mha']], flush=True)
    print('SINGLE_STEP', single, flush=True)
    assert single['next_token_identical'], 'Next token differs in the single-step comparison'


if __name__ == '__main__':
    main()

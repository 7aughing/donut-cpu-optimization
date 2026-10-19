"""Summarize operators from summary.json and three ORT profile files.

For each graph, skip warm-up calls and assign kernel events to model_run
windows. Check measured call counts against passes A/B and require the same
kernel count per call; compute shares from complete calls only.
Report run, kernel, and non-kernel time per call and group operators by type.
Identify lm_head by output dimension 57580 and group encoder operators by
compute/shape category and Swin stage (layers.N in node names).
Write <run_dir>/window.json. Usage: python donut_prof_window.py <run_dir>
"""

import bisect
import collections
import json
import re
import sys
from pathlib import Path

SHAPE_OPS = {'Reshape', 'Concat', 'Unsqueeze', 'Gather', 'Transpose', 'Shape', 'Expand', 'Where', 'Equal',
             'Cast', 'Range', 'Sub', 'Identity', 'Squeeze', 'Slice', 'ConstantOfShape'}
LM_HEAD_DIM = '57580'


def analyse(path, skip, expected_runs, tag):
    ev = json.load(open(path))
    runs = sorted((e for e in ev if e.get('cat') == 'Session' and e.get('name') == 'model_run'), key=lambda e: e['ts'])
    kern = sorted((e for e in ev if e.get('cat') == 'Node' and str(e.get('name', '')).endswith('_kernel_time')),
                  key=lambda e: e['ts'])
    del ev
    kts = [e['ts'] for e in kern]
    measured = runs[skip:]
    per = []
    for r in measured:
        a, b = r['ts'], r['ts'] + r['dur']
        i, j = bisect.bisect_left(kts, a), bisect.bisect_right(kts, b)
        per.append((r['dur'], kern[i:j]))
    counts = collections.Counter(len(k) for _, k in per)
    full_n = max(counts) if counts else 0
    full = [p for p in per if len(p[1]) == full_n]
    res = {'model_run_events': len(runs), 'kernel_events': len(kern), 'skipped_warmup_runs': skip,
           'measured_runs': len(measured), 'expected_measured_runs': expected_runs,
           'kernels_per_run_hist': dict(sorted(counts.items())), 'full_runs': len(full),
           'complete': len(measured) == expected_runs and len(full) == len(measured)}
    if not full:
        return res
    n = len(full)
    run_us = sum(d for d, _ in full)
    ker_us = sum(sum(k['dur'] for k in ks) for _, ks in full)
    res.update({'run_us_per_call': run_us / n, 'kernel_us_per_call': ker_us / n,
                'non_kernel_us_per_call': (run_us - ker_us) / n, 'non_kernel_share': (run_us - ker_us) / run_us})
    by_op = collections.defaultdict(float)
    by_group = collections.defaultdict(float)
    by_stage = collections.defaultdict(float)
    by_node = collections.defaultdict(lambda: [0.0, ''])
    for _, ks in full:
        for k in ks:
            a = k.get('args', {})
            op = a.get('op_name', '?')
            shp = json.dumps(a.get('output_type_shape', ''))
            name = re.sub(r'_kernel_time$', '', k['name'])
            if op in ('MatMul', 'FusedMatMul', 'Gemm') and LM_HEAD_DIM in shp:
                op = op + '(lm_head)'
            by_op[op] += k['dur']
            by_group['shape_ops' if op in SHAPE_OPS else 'compute_ops'] += k['dur']
            m = re.search(r'layers\.(\d+)', name)
            by_stage[f'layers.{m.group(1)}' if m else 'other'] += k['dur']
            key = re.sub(r'(layers|blocks)\.\d+', r'\1.N', name)
            by_node[(op, key)][0] += k['dur']
            by_node[(op, key)][1] = shp[:80]
    res['by_op'] = {op: {'us_per_call': us / n, 'share_of_run': us / run_us}
                    for op, us in sorted(by_op.items(), key=lambda kv: -kv[1])}
    res['by_group'] = {g: {'us_per_call': us / n, 'share_of_run': us / run_us} for g, us in by_group.items()}
    if tag == 'enc':
        res['by_stage'] = {s: {'us_per_call': us / n, 'share_of_run': us / run_us} for s, us in sorted(by_stage.items())}
    res['top_nodes'] = [{'op': op, 'node': key, 'us_per_call': us / n, 'share_of_run': us / run_us, 'out_shape': shp}
                        for (op, key), (us, shp) in sorted(by_node.items(), key=lambda kv: -kv[1][0])[:20]]
    return res


def main():
    d = Path(sys.argv[1])
    s = json.loads((d / 'summary.json').read_text())
    out = {'backend': s['backend'], 'images': s['images']}
    for tag in ('enc', 'dec', 'dec_p'):
        out[tag] = analyse(s['profile_files'][tag], s['warmup_runs'][tag], s['measured_runs'][tag], tag)
        r = out[tag]
        print(f'== {tag} complete={r["complete"]} measured={r["measured_runs"]}/{r["expected_measured_runs"]} '
              f'full={r["full_runs"]} kernels/run={r["kernels_per_run_hist"]}')
        if 'run_us_per_call' not in r:
            continue
        print(f'   run {r["run_us_per_call"]:.0f} us/call  kernel {r["kernel_us_per_call"]:.0f}  '
              f'non-kernel {r["non_kernel_us_per_call"]:.0f} ({100 * r["non_kernel_share"]:.1f}%)')
        for op, v in list(r['by_op'].items())[:14]:
            print(f'   {op:32s} {v["us_per_call"]:10.1f} us  {100 * v["share_of_run"]:5.1f}%')
        print('   groups', {g: f'{100 * v["share_of_run"]:.1f}%' for g, v in r['by_group'].items()})
        if 'by_stage' in r:
            print('   stages', {k: f'{100 * v["share_of_run"]:.1f}%' for k, v in r['by_stage'].items()})
        for t in r['top_nodes'][:10]:
            print(f'   top {t["us_per_call"]:9.1f} us {100 * t["share_of_run"]:5.1f}%  {t["op"]}  {t["node"][-80:]}')
    (d / 'window.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print('ALL_COMPLETE', all(out[t]['complete'] for t in ('enc', 'dec', 'dec_p')))


if __name__ == '__main__':
    main()

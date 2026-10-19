"""Summarize per-image results from donut_cpu_probe.py.

Write latency distributions, segment timings, encoder shares, output lengths,
and latency-versus-length fits to summary_<tag>.json in the run directory.
Use numpy.percentile's default linear interpolation for P95 and ordinary
least squares for the fits.
"""

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np


def dist(values):
    return {'n': len(values), 'median': st.median(values), 'mean': st.mean(values),
            'p95': float(np.percentile(values, 95)), 'min': min(values), 'max': max(values)}


def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    tt = sum((y - my) ** 2 for y in ys)
    return {'intercept_ms': a, 'slope_ms_per_token': b, 'r2': 1 - ss / tt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--tag', default='torch_fp32')
    args = parser.parse_args()
    run = Path(args.run_dir)
    rows = [json.loads(line) for line in (run / f'results_{args.tag}.jsonl').read_text().splitlines() if line.strip()]
    metas = [json.loads(line) for line in (run / f'manifest_{args.tag}.jsonl').read_text().splitlines() if line.strip()]
    files = [r['file'] for r in rows]
    assert len(files) == len(set(files)), 'Duplicate image records'

    seg = {k: dist([r[k] for r in rows]) for k in
           ('elapsed_ms', 'preprocess_ms', 'generate_ms', 'encoder_ms', 'decoder_ms', 'prob_ms', 'postprocess_ms')}
    gen = [r['n_generated'] for r in rows]
    per_step = [r['decoder_ms'] / r['decoder_calls'] for r in rows]
    enc_share = [r['encoder_ms'] / r['elapsed_ms'] for r in rows]
    other_ms = [r['generate_ms'] - r['encoder_ms'] - r['decoder_ms'] for r in rows]
    summary = {
        'run_dir': str(run), 'tag': args.tag, 'n_images': len(rows), 'n_process_runs': len(metas),
        'environment': {k: metas[-1].get(k) for k in ('hostname', 'python', 'executable', 'versions', 'cpu_quota',
                                                      'memory_max', 'torch_intra_threads', 'torch_interop_threads',
                                                      'tokenizer_class', 'sentencepiece_file', 'weight_sha256',
                                                      'script_sha256', 'max_length', 'forced_eos_token_id_top',
                                                      'forced_eos_token_id_decoder', 'n_params', 'lm_head_shape',
                                                      'lm_head_tied_to_embed', 'encoder_output_shape',
                                                      'pixel_values_shape', 'load_s', 'process_maxrss_kib',
                                                      'load_avg_start', 'load_avg_end', 'created_utc', 'finished_utc')},
        'segments_ms': seg,
        'n_generated': dist(gen),
        'hit_max_length': sum(r['hit_max_length'] for r in rows),
        'ended_with_eos': sum(r['ended_with_eos'] for r in rows),
        'encoder_calls_per_image': sorted({r['encoder_calls'] for r in rows}),
        'decoder_ms_per_step': dist(per_step),
        'generate_loop_overhead_ms': dist(other_ms),
        'encoder_share': {'of_median_elapsed': seg['encoder_ms']['median'] / seg['elapsed_ms']['median'],
                          'per_image': dist(enc_share)},
        'fit_elapsed_vs_n_generated': fit(gen, [r['elapsed_ms'] for r in rows]),
        'fit_decoder_ms_vs_n_generated': fit(gen, [r['decoder_ms'] for r in rows]),
        'min_token_probability': dist([min(r['token_probabilities']) for r in rows]),
    }
    out = run / f'summary_{args.tag}.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ('n_images', 'hit_max_length', 'ended_with_eos', 'encoder_calls_per_image')}))
    for k, v in seg.items():
        print(f"{k:16s} median {v['median']:9.1f}  p95 {v['p95']:9.1f}  mean {v['mean']:9.1f}  min {v['min']:9.1f}  max {v['max']:9.1f}")
    for k in ('n_generated', 'decoder_ms_per_step', 'generate_loop_overhead_ms', 'min_token_probability'):
        v = summary[k]
        print(f"{k:26s} median {v['median']:.3f}  p95 {v['p95']:.3f}  min {v['min']:.3f}  max {v['max']:.3f}")
    print('encoder_share', json.dumps(summary['encoder_share']))
    print('fit_elapsed', json.dumps(summary['fit_elapsed_vs_n_generated']))
    print('fit_decoder', json.dumps(summary['fit_decoder_ms_vs_n_generated']))
    print('WROTE', out)


if __name__ == '__main__':
    main()

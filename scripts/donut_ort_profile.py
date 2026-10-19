"""Profile Donut ONNX inference at the Python-loop, session, and operator levels.

Use the first N CORD-v2 test images in two passes, each warmed up on image 1:
  A: No ORT profiler. Time preprocessing, encoder, first decoder, cached
     decoder steps, Python loop overhead, surrounding work, and parsing.
     Use this pass for absolute latency.
  B: Rebuild the three sessions with per-graph ORT profiling. Record profile
     paths and call counts for donut_prof_window.py. Use operator shares from
     this pass because profiling adds overhead.
Load sessions from file paths without retaining ModelProto objects. Use 2
PyTorch threads and ONNX Runtime intra/inter-op thread counts of 2/1.
Usage: python donut_ort_profile.py --backend onnx_mha2 --bundle-dir <bundle>
       --graphs-manifest <json-with-graphs> --out <new-dir> ...
"""

import argparse
import collections
import gc
import hashlib
import json
import re
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GRAPHS = ('encoder.onnx', 'decoder.onnx', 'decoder_with_past.onnx')
TASK_PROMPT = '<s_cord-v2>'


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=('onnx_mha2', 'onnx_mha2_xkv'))
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--graphs-manifest', required=True)
    parser.add_argument('--out', required=True, help='Output directory; must not already exist')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--manifest', required=True, help='CORD manifest.json')
    parser.add_argument('--n-images', type=int, default=3)
    parser.add_argument('--extra-path', default=None)
    parser.add_argument('--runner-dir', required=True)
    args = parser.parse_args()
    if args.extra_path:
        sys.path.insert(0, args.extra_path)

    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from transformers import DonutProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    bundle = Path(args.bundle_dir)
    expected = json.loads(Path(args.graphs_manifest).read_text())['graphs']
    graph_sha = {}
    for g in GRAPHS:
        graph_sha[g] = digest(bundle / g)
        assert graph_sha[g] == expected[g]['sha256'], ('ONNX graph SHA256 mismatch', g)
    torch.set_num_threads(2)

    processor = DonutProcessor.from_pretrained(args.model_dir)
    tok = processor.tokenizer
    prompt_ids = tok(TASK_PROMPT, add_special_tokens=False).input_ids
    items = json.loads(Path(args.manifest).read_text())['images'][:args.n_images]
    images = []
    for item in items:
        assert digest(item['path']) == item['sha256'], ('Image SHA256 mismatch', item['path'])
        with Image.open(item['path']) as im:
            images.append((item['file'], im.convert('RGB')))

    sys.path.insert(0, args.runner_dir)
    import donut_onnx_runner as runner_mod

    def opts(prefix=None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        if prefix:
            so.enable_profiling = True
            so.profile_file_prefix = prefix
        return so

    def build(prefixes=None):
        orch = runner_mod.DonutOnnxOrchestrator(
            bundle_dir=str(bundle), max_length=768, prompt_token_id=prompt_ids[0], pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id, bad_words_ids=[[tok.unk_token_id]], forced_eos_token_id=None,
            num_decoder_layers=4, sess_options=opts(), providers=['CPUExecutionProvider'])
        if prefixes:  # Release old sessions before creating profiled sessions with per-graph prefixes.
            for attr, tag in (('_enc', 'enc'), ('_dec', 'dec'), ('_dec_p', 'dec_p')):
                holder = getattr(orch, attr)
                holder.sess = None
                gc.collect()
                holder.sess = ort.InferenceSession(holder.path, opts(prefixes[tag]), providers=['CPUExecutionProvider'])
        rec = {}

        def wrap_run(holder, tag):
            orig = holder.run

            def run(names, feed):
                t0 = time.perf_counter()
                r = orig(names, feed)
                rec['runs'].append((tag, t0, time.perf_counter()))
                return r
            holder.run = run

        for attr, tag in (('_enc', 'enc'), ('_dec', 'dec'), ('_dec_p', 'dec_p')):
            wrap_run(getattr(orch, attr), tag)
        return orch, rec

    def one(orch, rec, image):
        rec['runs'] = []
        t0 = time.perf_counter()
        pv = processor(image, return_tensors='pt').pixel_values
        t1 = time.perf_counter()
        res = orch(pv)
        t2 = time.perf_counter()
        text = processor.batch_decode(res['sequence'])[0]
        text = re.sub(r'<.*?>', '', text.replace(tok.eos_token, '').replace(tok.pad_token, ''), count=1).strip()
        parsed = processor.token2json(text)
        t3 = time.perf_counter()
        runs = rec['runs']
        d = collections.defaultdict(float)
        n = collections.Counter()
        for tag, a, b in runs:
            d[tag] += b - a
            n[tag] += 1
        ms = lambda s: s * 1000
        r = {'total_ms': ms(t3 - t0), 'preprocess_ms': ms(t1 - t0), 'parse_ms': ms(t3 - t2), 'forward_ms': ms(t2 - t1),
             'enc_run_ms': ms(d['enc']), 'dec_run_ms': ms(d['dec']), 'dec_p_run_ms': ms(d['dec_p']),
             'dec_runs': n['dec'], 'dec_p_runs': n['dec_p'], 'enc_runs': n['enc'],
             'head_ms': ms(runs[0][1] - t1), 'post_loop_ms': ms(t2 - runs[-1][2]),
             'seq_len': int(res['sequence'].shape[1])}
        r['loop_python_ms'] = r['forward_ms'] - r['enc_run_ms'] - r['dec_run_ms'] - r['dec_p_run_ms'] - r['post_loop_ms'] - r['head_ms']
        r['decode_steps'] = n['dec'] + n['dec_p']
        return r, res['sequence'][0].tolist(), parsed

    # ---------------- Pass A: profiling disabled
    orch, rec = build()
    one(orch, rec, images[0][1])  # Warm up.
    rows_a, seq_a = [], {}
    for name, image in images:
        r, seq, _ = one(orch, rec, image)
        r['file'] = name
        rows_a.append(r)
        seq_a[name] = seq
        print('A', name, {k: round(r[k], 1) for k in ('total_ms', 'enc_run_ms', 'dec_p_run_ms', 'loop_python_ms')},
              'steps', r['decode_steps'], flush=True)
    # Release the runner and records together; timing closures retain pass A sessions.
    del orch, rec
    gc.collect()
    maxrss_after_a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ---------------- Pass B: profiling enabled
    prefixes = {t: str(out / f'prof_{t}') for t in ('enc', 'dec', 'dec_p')}
    orch, rec = build(prefixes)
    warm, _, _ = one(orch, rec, images[0][1])
    rows_b, same = [], 0
    for name, image in images:
        r, seq, _ = one(orch, rec, image)
        r['file'] = name
        rows_b.append(r)
        same += seq == seq_a[name]
        print('B', name, {k: round(r[k], 1) for k in ('total_ms', 'enc_run_ms', 'dec_p_run_ms')}, flush=True)
    prof_files = {t: getattr(orch, a).sess.end_profiling() for a, t in (('_enc', 'enc'), ('_dec', 'dec'), ('_dec_p', 'dec_p'))}

    def med(rows, k):
        return statistics.median(r[k] for r in rows)

    keys = ['total_ms', 'preprocess_ms', 'enc_run_ms', 'dec_run_ms', 'dec_p_run_ms', 'loop_python_ms', 'post_loop_ms',
            'head_ms', 'parse_ms', 'decode_steps', 'seq_len']
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'backend': args.backend, 'bundle_dir': str(bundle),
        'graph_sha256': graph_sha, 'runner_sha256': digest(Path(args.runner_dir) / 'donut_onnx_runner.py'),
        'script_sha256': digest(Path(__file__)), 'images': [n for n, _ in images],
        'versions': {'onnxruntime': ort.__version__, 'torch': torch.__version__},
        'threads': {'torch_intra': torch.get_num_threads(), 'ort_intra': 2, 'ort_inter': 1},
        'cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        'dec_p_emits_cross_kv': orch._dec_p_emits_cross_kv,
        'pass_a_median': {k: med(rows_a, k) for k in keys},
        'pass_a_sum': {k: sum(r[k] for r in rows_a) for k in keys},
        'pass_b_median': {k: med(rows_b, k) for k in keys},
        'pass_b_sequences_same_as_a': f'{same}/{len(images)}',
        'warmup_runs': {'enc': warm['enc_runs'], 'dec': warm['dec_runs'], 'dec_p': warm['dec_p_runs']},
        'measured_runs': {'enc': sum(r['enc_runs'] for r in rows_b), 'dec': sum(r['dec_runs'] for r in rows_b),
                          'dec_p': sum(r['dec_p_runs'] for r in rows_b)},
        'profile_files': prof_files, 'rows_a': rows_a, 'rows_b': rows_b,
        'maxrss_kib_after_pass_a': maxrss_after_a,
        'process_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    A = summary['pass_a_sum']
    print('A_SUM_MS', {k: round(v, 1) for k, v in A.items()})
    for k in ['preprocess_ms', 'enc_run_ms', 'dec_run_ms', 'dec_p_run_ms', 'loop_python_ms', 'post_loop_ms', 'head_ms', 'parse_ms']:
        print('A_SHARE', k, f'{100 * A[k] / A["total_ms"]:.1f}%')
    print('B_SEQ_SAME', summary['pass_b_sequences_same_as_a'], 'MEASURED_RUNS', summary['measured_runs'])
    assert same == len(images), 'Pass B sequences differ from pass A'
    print('PROFILE_DONE', out)


if __name__ == '__main__':
    main()

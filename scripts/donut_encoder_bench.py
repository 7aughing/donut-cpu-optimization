"""Compare and benchmark the PyTorch DonutSwin and ONNX Runtime encoders.

--mode parity loads both backends and compares encoder_hidden_states for the
first N images using maximum/mean absolute error and cosine similarity.
--mode time loads one backend, warms up once per image, and records --repeats
runs. Inputs are preprocessed pixel_values; preprocessing is not timed.
Use 2 PyTorch threads and ONNX Runtime intra/inter-op thread counts of 2/1.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_WEIGHT_SHA256 = '31b78e3d3891072de8e2bf3553b71782242a1f3b589b914ec2b03feff7b14c54'


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('parity', 'time'))
    parser.add_argument('--backend', choices=('torch', 'ort'))
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--encoder-sha256', required=True, help='SHA256 of encoder.onnx from the export manifest')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--n-images', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--extra-path', default=None)
    args = parser.parse_args()
    if args.extra_path:
        sys.path.insert(0, args.extra_path)

    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from transformers import DonutProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir, bundle = Path(args.model_dir), Path(args.bundle_dir)
    torch.set_num_threads(2)
    processor = DonutProcessor.from_pretrained(str(model_dir))
    items = json.loads(Path(args.manifest).read_text())['images'][:args.n_images]
    pixels = []
    for item in items:
        assert digest(item['path']) == item['sha256']
        with Image.open(item['path']) as im:
            pixels.append((item['file'], processor(im.convert('RGB'), return_tensors='pt').pixel_values))

    meta = {'created_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode, 'backend': args.backend,
            'round': args.round, 'hostname': platform.node(), 'executable': sys.executable,
            'versions': {k: version(k) for k in ('torch', 'transformers', 'onnxruntime', 'numpy')},
            'torch_intra_threads': torch.get_num_threads(), 'ort_intra': 2, 'ort_inter': 1,
            'load_avg_start': os.getloadavg(), 'script_sha256': digest(Path(__file__)),
            'n_images': len(pixels), 'repeats': args.repeats}

    def load_torch():
        from transformers import VisionEncoderDecoderModel
        assert digest(model_dir / 'pytorch_model.bin') == EXPECTED_WEIGHT_SHA256
        model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).eval()
        enc = model.encoder

        def run(pv):
            with torch.no_grad():
                return enc(pixel_values=pv, return_dict=True).last_hidden_state.numpy()
        return run

    def load_ort():
        p = bundle / 'encoder.onnx'
        assert digest(p) == args.encoder_sha256, 'encoder.onnx SHA256 mismatch'
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(p), sess_options=so, providers=['CPUExecutionProvider'])
        meta['ort_graph_optimization_level'] = str(so.graph_optimization_level)

        def run(pv):
            return sess.run(['encoder_hidden_states'], {'image': pv.numpy()})[0]
        return run

    if args.mode == 'parity':
        run_t, run_o = load_torch(), load_ort()
        rows = []
        for name, pv in pixels:
            a, b = run_t(pv).astype(np.float64), run_o(pv).astype(np.float64)
            diff = np.abs(a - b)
            cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
            tok_cos = (a[0] * b[0]).sum(-1) / (np.linalg.norm(a[0], axis=-1) * np.linalg.norm(b[0], axis=-1))
            rows.append({'file': name, 'shape': list(a.shape), 'max_abs_diff': float(diff.max()),
                         'mean_abs_diff': float(diff.mean()), 'ref_max_abs': float(np.abs(a).max()),
                         'cosine_all': cos, 'cosine_min_per_position': float(tok_cos.min())})
            print('PARITY', rows[-1], flush=True)
        meta['results'] = rows
        meta['summary'] = {'max_abs_diff': max(r['max_abs_diff'] for r in rows),
                           'mean_abs_diff_mean': float(np.mean([r['mean_abs_diff'] for r in rows])),
                           'cosine_min_per_position': min(r['cosine_min_per_position'] for r in rows)}
        target = out / 'encoder_parity.json'
    else:
        t0 = time.perf_counter()
        run = load_torch() if args.backend == 'torch' else load_ort()
        meta['load_s'] = round(time.perf_counter() - t0, 3)
        rows = []
        for name, pv in pixels:
            run(pv)  # Warm up once per image.
            for rep in range(args.repeats):
                s = time.perf_counter()
                run(pv)
                rows.append({'file': name, 'repeat': rep, 'ms': (time.perf_counter() - s) * 1000})
            print('TIME', args.backend, name, [round(r['ms']) for r in rows[-args.repeats:]], flush=True)
        meta['results'] = rows
        target = out / f'encoder_time_{args.backend}_round{args.round}.json'
    meta['load_avg_end'] = os.getloadavg()
    meta['process_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print('WROTE', target, flush=True)


if __name__ == '__main__':
    main()

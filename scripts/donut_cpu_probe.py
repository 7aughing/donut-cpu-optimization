"""Benchmark Donut CORD-v2 one image at a time in a single process (batch size 1).

torch_fp32 uses generate with <s_cord-v2>, the decoder position limit,
use_cache, bad_words_ids=[[unk]], return_dict_in_generate, and output_scores.
onnx_raw, onnx_mha2, and onnx_mha2_xkv use the raw, fused, and cross-attention
K/V reuse bundles through DonutOnnxOrchestrator and its Python greedy loop.
The checkpoint has no top-level forced_eos_token_id.

Timing starts with a decoded RGB image entering the processor and ends after
token2json; token probability computation is included. Warm up on the first
image, then time each image once. Append results to results_<tag>.jsonl and
skip completed images when resuming.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_WEIGHT_SHA256 = '31b78e3d3891072de8e2bf3553b71782242a1f3b589b914ec2b03feff7b14c54'
TASK_PROMPT = '<s_cord-v2>'
MAXRSS_LIMIT_KIB = int(6.5 * 1024 * 1024)
GRAPHS = ('encoder.onnx', 'decoder.onnx', 'decoder_with_past.onnx')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cgroup(name):
    p = Path('/sys/fs/cgroup') / name
    return p.read_text().strip() if p.exists() else None


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='torch_fp32', choices=('torch_fp32', 'onnx_raw', 'onnx_mha2', 'onnx_mha2_xkv'))
    parser.add_argument('--model-dir', required=True, help='Checkpoint directory containing PyTorch weights and processor files')
    parser.add_argument('--manifest', required=True, help='manifest.json produced by cord_prepare.py')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag', default=None, help='Defaults to --backend')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--extra-path', default=None, help='Prepend a dependency directory, such as a separate sentencepiece installation')
    parser.add_argument('--bundle-dir', default=None, help='ONNX backend: directory containing the three graphs')
    parser.add_argument('--export-manifest', default=None, help='ONNX backend: export or optimization manifest with graph SHA256 values')
    parser.add_argument('--runner-dir', default=None, help='ONNX backend: directory containing donut_onnx_runner.py')
    parser.add_argument('--ort-intra', type=int, default=2)
    parser.add_argument('--ort-inter', type=int, default=1)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--position', type=int, default=0)
    args = parser.parse_args()
    tag = args.tag or args.backend

    if args.extra_path:
        sys.path.insert(0, args.extra_path)

    import torch
    from PIL import Image
    from transformers import DonutProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    manifest = json.loads(Path(args.manifest).read_text())
    items = manifest['images'][args.start:args.end]
    assert items, 'Empty image slice'
    for item in items:
        assert digest(item['path']) == item['sha256'], ('Image SHA256 mismatch', item['path'])

    results_path = out / f'results_{tag}.jsonl'
    done = set()
    if results_path.exists():
        done = {json.loads(line)['file'] for line in results_path.read_text().splitlines() if line.strip()}

    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    processor = DonutProcessor.from_pretrained(str(model_dir))
    tok = processor.tokenizer
    prompt_ids = tok(TASK_PROMPT, add_special_tokens=False).input_ids
    meta = {}
    probe = {'enc': [], 'dec': [], 'dec_p': [], 'enc_shape': None}

    def decode(sequence):
        text = processor.batch_decode(sequence)[0]
        text = text.replace(tok.eos_token, '').replace(tok.pad_token, '')
        text = re.sub(r'<.*?>', '', text, count=1).strip()
        return text, processor.token2json(text)

    if args.backend == 'torch_fp32':
        from transformers import VisionEncoderDecoderModel
        weight_sha = digest(model_dir / 'pytorch_model.bin')
        assert weight_sha == EXPECTED_WEIGHT_SHA256, ('Model weight SHA256 mismatch', weight_sha)
        model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
        model.eval()
        max_length = model.decoder.config.max_position_embeddings

        # Record timings and call counts with hooks, without extra inference calls.
        def enc_pre(module, inputs):
            probe['_enc_t'] = time.perf_counter()

        def enc_post(module, inputs, output):
            probe['enc'].append(time.perf_counter() - probe['_enc_t'])
            probe['enc_shape'] = list(output[0].shape)

        def dec_pre(module, inputs):
            probe['_dec_t'] = time.perf_counter()

        def dec_post(module, inputs, output):
            probe['dec'].append(time.perf_counter() - probe['_dec_t'])

        model.encoder.register_forward_pre_hook(enc_pre)
        model.encoder.register_forward_hook(enc_post)
        model.decoder.register_forward_pre_hook(dec_pre)
        model.decoder.register_forward_hook(dec_post)

        def run(image):
            for k in ('enc', 'dec', 'dec_p'):
                probe[k].clear()
            a = time.perf_counter()
            pixel_values = processor(image, return_tensors='pt').pixel_values
            decoder_input_ids = tok(TASK_PROMPT, add_special_tokens=False, return_tensors='pt').input_ids
            b = time.perf_counter()
            outputs = model.generate(
                pixel_values, decoder_input_ids=decoder_input_ids, max_length=max_length, early_stopping=True,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True, num_beams=1,
                bad_words_ids=[[tok.unk_token_id]], return_dict_in_generate=True, output_scores=True)
            c = time.perf_counter()
            probs = torch.stack(outputs.scores, dim=1).softmax(-1).max(-1).values[0]
            d = time.perf_counter()
            text, parsed = decode(outputs.sequences)
            e = time.perf_counter()
            return {
                'elapsed_ms': (e - a) * 1000, 'preprocess_ms': (b - a) * 1000, 'generate_ms': (c - b) * 1000,
                'encoder_ms': sum(probe['enc']) * 1000, 'encoder_calls': len(probe['enc']),
                'decoder_ms': sum(probe['dec']) * 1000, 'decoder_calls': len(probe['dec']),
                'prob_ms': (d - c) * 1000, 'postprocess_ms': (e - d) * 1000,
                'pixel_values_shape': list(pixel_values.shape), 'prompt_len': decoder_input_ids.shape[1],
                'sequence': outputs.sequences[0].tolist(), 'token_probabilities': probs.tolist(),
                'text': text, 'parsed': parsed,
            }

        meta.update({
            'weight_sha256': weight_sha,
            'forced_eos_token_id_top': model.config.forced_eos_token_id,
            'forced_eos_token_id_decoder': model.decoder.config.forced_eos_token_id,
            'decoder_start_token_id_top': model.config.decoder_start_token_id,
            'n_params': {'encoder': sum(p.numel() for p in model.encoder.parameters()),
                         'decoder': sum(p.numel() for p in model.decoder.parameters()),
                         'total': sum(p.numel() for p in model.parameters())},
            'lm_head_shape': list(model.decoder.lm_head.weight.shape),
            'lm_head_tied_to_embed': model.decoder.lm_head.weight.data_ptr() == model.decoder.model.decoder.embed_tokens.weight.data_ptr(),
            'scope': 'PyTorch FP32, batch size 1, single process. Timing includes preprocessing, generate, token probabilities, batch_decode, and token2json; excludes image loading and decoding.',
        })
    else:
        import onnxruntime as ort
        bundle = Path(args.bundle_dir)
        export = json.loads(Path(args.export_manifest).read_text())
        graph_sha = {}
        for g in GRAPHS:
            graph_sha[g] = digest(bundle / g)
            assert graph_sha[g] == export['graphs'][g]['sha256'], ('ONNX graph SHA256 mismatch', g)
        runner_path = Path(args.runner_dir) / 'donut_onnx_runner.py'
        sys.path.insert(0, args.runner_dir)
        from donut_onnx_runner import DonutOnnxOrchestrator  # Load sessions from file paths without retaining ModelProto objects.
        max_length = 768
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.ort_intra
        options.inter_op_num_threads = args.ort_inter
        orch = DonutOnnxOrchestrator(
            bundle_dir=str(bundle), max_length=max_length, prompt_token_id=prompt_ids[0],
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, bad_words_ids=[[tok.unk_token_id]],
            forced_eos_token_id=None, num_decoder_layers=4, sess_options=options,
            providers=['CPUExecutionProvider'])

        # Time each session.run call without changing the decoding loop.
        def timed(holder, key):
            original = holder.run

            def wrapper(output_names, input_feed):
                s = time.perf_counter()
                result = original(output_names, input_feed)
                probe[key].append(time.perf_counter() - s)
                if key == 'enc':
                    probe['enc_shape'] = list(result[0].shape)
                return result
            holder.run = wrapper

        timed(orch._enc, 'enc')
        timed(orch._dec, 'dec')
        timed(orch._dec_p, 'dec_p')

        def run(image):
            for k in ('enc', 'dec', 'dec_p'):
                probe[k].clear()
            a = time.perf_counter()
            pixel_values = processor(image, return_tensors='pt').pixel_values
            b = time.perf_counter()
            res = orch(pixel_values)
            c = time.perf_counter()
            text, parsed = decode(res['sequence'])
            e = time.perf_counter()
            dec_ms = (sum(probe['dec']) + sum(probe['dec_p'])) * 1000
            return {
                'elapsed_ms': (e - a) * 1000, 'preprocess_ms': (b - a) * 1000, 'generate_ms': (c - b) * 1000,
                'encoder_ms': sum(probe['enc']) * 1000, 'encoder_calls': len(probe['enc']),
                'decoder_ms': dec_ms, 'decoder_calls': len(probe['dec']) + len(probe['dec_p']),
                'first_decoder_ms': sum(probe['dec']) * 1000, 'decoder_past_ms': sum(probe['dec_p']) * 1000,
                'decoder_past_calls': len(probe['dec_p']),
                'prob_ms': 0.0, 'postprocess_ms': (e - c) * 1000,
                'pixel_values_shape': list(pixel_values.shape), 'prompt_len': 1,
                'sequence': res['sequence'][0].tolist(), 'token_probabilities': res['probabilities'][0].tolist(),
                'text': text, 'parsed': parsed,
            }

        meta.update({
            'bundle_dir': str(bundle), 'graph_sha256': graph_sha, 'runner_sha256': digest(runner_path),
            'ort_intra_threads': args.ort_intra, 'ort_inter_threads': args.ort_inter,
            'ort_graph_optimization_level': str(options.graph_optimization_level),
            'dec_p_emits_cross_kv': orch._dec_p_emits_cross_kv, 'forced_eos_token_id': None,
            'session_loading': 'Load InferenceSession from file paths without retaining ModelProto objects',
            'maxrss_kib_after_load': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope': f'Three ONNX graphs ({args.backend}) with a Python greedy loop, batch size 1, single process. Timing includes preprocessing, the runner with token probabilities, batch_decode, and token2json; excludes image loading and decoding. prob_ms is 0 because probability computation is included in generate_ms.',
        })
    load_s = time.perf_counter() - t0

    def load_image(item):
        with Image.open(item['path']) as im:
            return im.convert('RGB')

    meta.update({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'tag': tag, 'backend': args.backend,
        'round': args.round, 'position': args.position, 'slice': [args.start, args.end],
        'python': sys.version.split()[0], 'executable': sys.executable, 'hostname': platform.node(),
        'versions': {k: version(k) for k in ('torch', 'transformers', 'tokenizers', 'sentencepiece', 'numpy',
                                             'Pillow', 'onnx', 'onnxruntime')},
        'cpu_quota': cgroup('cpu.max'), 'memory_max': cgroup('memory.max'),
        'sched_affinity': sorted(os.sched_getaffinity(0)), 'load_avg_start': os.getloadavg(),
        'torch_intra_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'cpu_stat_start': cgroup('cpu.stat'), 'manifest': args.manifest, 'model_dir': str(model_dir),
        'script_sha256': digest(Path(__file__)), 'load_s': round(load_s, 3),
        'extra_path': args.extra_path,
        'sentencepiece_file': getattr(sys.modules.get('sentencepiece'), '__file__', None),
        'tokenizer_class': type(tok).__name__, 'task_prompt': TASK_PROMPT, 'task_prompt_ids': prompt_ids,
        'special_ids': {'pad': tok.pad_token_id, 'eos': tok.eos_token_id, 'unk': tok.unk_token_id, 'bos': tok.bos_token_id},
        'len_tokenizer': len(tok), 'max_length': max_length,
    })

    warm = run(load_image(items[0]))  # Warm up; exclude this result from the measured samples.
    meta.update({'warmup_file': items[0]['file'], 'warmup_elapsed_ms': warm['elapsed_ms'],
                 'encoder_output_shape': probe['enc_shape'], 'pixel_values_shape': warm['pixel_values_shape']})

    n = 0
    with results_path.open('a') as stream:
        for item in items:
            if item['file'] in done:
                continue
            image = load_image(item)
            r = run(image)
            seq = r['sequence']
            record = {'file': item['file'], 'index': item['index'], 'image_sha256': item['sha256'],
                      'image_size': list(image.size), 'tag': tag, 'backend': args.backend,
                      'round': args.round, 'position': args.position,
                      'n_generated': len(seq) - r['prompt_len'],
                      'hit_max_length': len(seq) >= max_length, 'ended_with_eos': seq[-1] == tok.eos_token_id,
                      'encoder_output_shape': probe['enc_shape'], **r}
            assert math.isfinite(record['elapsed_ms']) and record['elapsed_ms'] > 0 and seq
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            n += 1
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print('IMG', tag, item['file'], f"{record['elapsed_ms']:.0f}ms", 'gen', record['n_generated'],
                  f"enc {record['encoder_ms']:.0f}ms", 'maxrss_MiB', rss // 1024, flush=True)
            if rss > MAXRSS_LIMIT_KIB:
                meta['aborted'] = f'Peak process memory {rss} KiB exceeds 6.5 GiB'
                break

    meta.update({'records_written': n, 'skipped_done': len(done), 'cpu_stat_end': cgroup('cpu.stat'),
                 'load_avg_end': os.getloadavg(),
                 'process_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'cgroup_memory_current_end': cgroup('memory.current'),
                 'finished_utc': datetime.now(timezone.utc).isoformat()})
    with (out / f'manifest_{tag}.jsonl').open('a') as stream:
        stream.write(json.dumps(meta, ensure_ascii=False) + '\n')
    print('DONE', tag, 'records', n, 'skipped', len(done), 'maxrss_kib', meta['process_maxrss_kib'], flush=True)
    if 'aborted' in meta:
        sys.exit(2)


if __name__ == '__main__':
    main()

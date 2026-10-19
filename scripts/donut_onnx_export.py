"""Export Donut CORD-v2 as encoder, first-step decoder, and cached decoder graphs.

Each --only action runs in a separate process to limit peak memory:
  selfcheck          Compare greedy decoding through the PyTorch wrappers
                     with generate; token sequences must match exactly.
  encoder            Export encoder.onnx.
  decoder            Export decoder.onnx without a past cache.
  decoder_with_past  Export decoder_with_past.onnx with 16 cache tensors.
  check              Run onnx.checker, record node counts, sizes, and SHA256,
                     load all graphs in ORT, inspect input/output names and
                     encoder output shape, and write export_manifest.json.
Cache names follow past_{L}_{self_k|self_v|cross_k|cross_v} and present_* as
expected by donut_onnx_runner.py. Existing graph files are not overwritten.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

EXPECTED_WEIGHT_SHA256 = '31b78e3d3891072de8e2bf3553b71782242a1f3b589b914ec2b03feff7b14c54'
TASK_PROMPT = '<s_cord-v2>'
KV_KINDS = ('self_k', 'self_v', 'cross_k', 'cross_v')
GRAPHS = ('encoder.onnx', 'decoder.onnx', 'decoder_with_past.onnx')


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', required=True, choices=('selfcheck', 'encoder', 'decoder', 'decoder_with_past', 'check'))
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True, help='Export directory; graphs are written to <out>/bundle/')
    parser.add_argument('--extra-path', default=None)
    parser.add_argument('--selfcheck-images', type=int, default=3)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--no-constant-folding', action='store_true')
    args = parser.parse_args()
    if args.extra_path:
        sys.path.insert(0, args.extra_path)

    import torch
    import torch.nn as nn
    from PIL import Image
    from transformers import DonutProcessor

    out = Path(args.out)
    bundle = out / 'bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    folding = not args.no_constant_folding
    report = {'step': args.only, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': digest(Path(__file__)), 'python': sys.version.split()[0], 'executable': sys.executable,
              'versions': {k: version(k) for k in ('torch', 'transformers', 'tokenizers', 'sentencepiece',
                                                   'onnx', 'onnxruntime', 'numpy')},
              'opset': args.opset, 'do_constant_folding': folding}
    step_file = out / f'export_step_{args.only}.json'

    def save():
        step_file.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    torch.set_num_threads(2)
    model_dir = Path(args.model_dir)
    processor = DonutProcessor.from_pretrained(str(model_dir))
    tok = processor.tokenizer
    manifest = json.loads(Path(args.manifest).read_text())
    pixel = {}
    for item in manifest['images'][:args.selfcheck_images]:
        assert digest(item['path']) == item['sha256']
        with Image.open(item['path']) as im:
            pixel[item['file']] = processor(im.convert('RGB'), return_tensors='pt').pixel_values
    image0 = next(iter(pixel.values()))

    if args.only == 'check':
        import onnx
        import onnxruntime as ort
        graphs = {}
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        for name in GRAPHS:
            p = bundle / name
            onnx.checker.check_model(str(p))
            m = onnx.load(str(p))
            ops = {}
            for n in m.graph.node:
                ops[n.op_type] = ops.get(n.op_type, 0) + 1
            info = {'bytes': p.stat().st_size, 'sha256': digest(p), 'n_nodes': len(m.graph.node),
                    'n_initializers': len(m.graph.initializer),
                    'opset_imports': {o.domain or 'ai.onnx': o.version for o in m.opset_import},
                    'top_ops': sorted(ops.items(), key=lambda kv: -kv[1])[:12]}
            del m
            t0 = time.perf_counter()
            sess = ort.InferenceSession(str(p), sess_options=so, providers=['CPUExecutionProvider'])
            info['ort_load_s'] = round(time.perf_counter() - t0, 2)
            info['inputs'] = [(i.name, i.shape) for i in sess.get_inputs()]
            info['outputs'] = [(o.name, o.shape) for o in sess.get_outputs()]
            if name == 'encoder.onnx':
                y = sess.run(None, {'image': image0.numpy()})[0]
                info['ort_encoder_output_shape'] = list(y.shape)
                assert list(y.shape) == [1, 1200, 1024], y.shape
            del sess
            graphs[name] = info
            print('CHECKED', name, {k: info[k] for k in ('bytes', 'n_nodes', 'ort_load_s')}, flush=True)
        steps = {s: json.loads((out / f'export_step_{s}.json').read_text())
                 for s in ('selfcheck', 'encoder', 'decoder', 'decoder_with_past')
                 if (out / f'export_step_{s}.json').exists()}
        merged = {'created_utc': report['created_utc'], 'script_sha256': report['script_sha256'],
                  'versions': report['versions'], 'opset': args.opset, 'steps': steps, 'graphs': graphs,
                  'check_maxrss_mib': maxrss_mib(), 'finished_utc': datetime.now(timezone.utc).isoformat()}
        (out / 'export_manifest.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2, default=str))
        report['graphs'] = graphs
        report['maxrss_mib'] = maxrss_mib()
        save()
        print('DONE check maxrss_mib', report['maxrss_mib'], flush=True)
        return

    from transformers import VisionEncoderDecoderModel
    report['weight_sha256'] = digest(model_dir / 'pytorch_model.bin')
    assert report['weight_sha256'] == EXPECTED_WEIGHT_SHA256
    model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).eval()
    # Disable gradients before export to avoid retaining intermediate activations during tracing.
    model.requires_grad_(False)
    assert not hasattr(model, 'enc_to_dec_proj'), 'enc_to_dec_proj is present; the wrappers need a projection'
    assert model.config.forced_eos_token_id is None
    num_layers = model.decoder.config.decoder_layers
    max_length = model.decoder.config.max_position_embeddings
    prompt_id = tok(TASK_PROMPT, add_special_tokens=False).input_ids
    assert prompt_id == [57579], prompt_id
    prompt_id = prompt_id[0]
    pad_id, eos_id, unk_id = tok.pad_token_id, tok.eos_token_id, tok.unk_token_id
    report['maxrss_mib_after_load'] = maxrss_mib()

    # ---------- Export wrappers ----------
    class EncoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.encoder

        def forward(self, image):
            return self.encoder(pixel_values=image, return_dict=True).last_hidden_state

    class DecoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.bart_decoder = m.decoder.model.decoder
            self.lm_head = m.decoder.lm_head

        def forward(self, input_ids, attention_mask, encoder_hidden_states):
            o = self.bart_decoder(input_ids=input_ids, attention_mask=attention_mask,
                                  encoder_hidden_states=encoder_hidden_states, past_key_values=None,
                                  use_cache=True, return_dict=True)
            flat: List[torch.Tensor] = []
            for layer_kv in o.past_key_values:
                flat.extend(layer_kv)
            return (self.lm_head(o.last_hidden_state), *flat)

    class DecoderWithPastWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.bart_decoder = m.decoder.model.decoder
            self.lm_head = m.decoder.lm_head

        def forward(self, input_ids, attention_mask, encoder_hidden_states, *flat_past_kv):
            assert len(flat_past_kv) == num_layers * 4
            past = tuple(tuple(flat_past_kv[i * 4:(i + 1) * 4]) for i in range(num_layers))
            o = self.bart_decoder(input_ids=input_ids, attention_mask=attention_mask,
                                  encoder_hidden_states=encoder_hidden_states, past_key_values=past,
                                  use_cache=True, return_dict=True)
            flat: List[torch.Tensor] = []
            for layer_kv in o.past_key_values:
                flat.extend(layer_kv)
            return (self.lm_head(o.last_hidden_state), *flat)

    enc_w, dec_w, dec_p_w = EncoderWrapper(model).eval(), DecoderWrapper(model).eval(), DecoderWithPastWrapper(model).eval()
    kv_out = [f'present_{L}_{k}' for L in range(num_layers) for k in KV_KINDS]
    past_in = [f'past_{L}_{k}' for L in range(num_layers) for k in KV_KINDS]

    if args.only == 'selfcheck':
        @torch.no_grad()
        def greedy_with_wrappers(image) -> Tuple[List[int], List[float]]:
            # Greedy decoding; the checkpoint has no top-level forced EOS token.
            enc_hs = enc_w(image)
            seq = torch.tensor([[prompt_id]])
            scores = []

            def pick(last):
                ll = last.clone()
                ll[:, unk_id] = float('-inf')
                scores.append(ll)
                return ll.argmax(-1, keepdim=True)

            o = dec_w(seq, torch.ones_like(seq), enc_hs)
            kv = list(o[1:])
            nxt = pick(o[0][:, -1, :])
            seq = torch.cat([seq, nxt], 1)
            while seq.shape[1] < max_length and nxt.item() != eos_id:
                o = dec_p_w(nxt, torch.ones_like(seq), enc_hs, *kv)
                kv = list(o[1:])
                nxt = pick(o[0][:, -1, :])
                seq = torch.cat([seq, nxt], 1)
            probs = torch.stack([s.softmax(-1).max(-1).values for s in scores]).T[0]
            return seq[0].tolist(), probs.tolist()

        t0 = time.perf_counter()
        checks = []
        for name, pv in pixel.items():
            with torch.no_grad():
                g = model.generate(pv, decoder_input_ids=torch.tensor([[prompt_id]]), max_length=max_length,
                                   early_stopping=True, pad_token_id=pad_id, eos_token_id=eos_id, use_cache=True,
                                   num_beams=1, bad_words_ids=[[unk_id]], return_dict_in_generate=True,
                                   output_scores=True)
            ref_seq = g.sequences[0].tolist()
            ref_p = torch.stack(g.scores, 1).softmax(-1).max(-1).values[0].tolist()
            seq, p = greedy_with_wrappers(pv)
            same = seq == ref_seq
            checks.append({'file': name, 'n_generated': len(ref_seq) - 1, 'sequence_identical': same,
                           'max_prob_delta': max(abs(a - b) for a, b in zip(p, ref_p)) if same else None})
            print('SELFCHECK', checks[-1], flush=True)
        report.update({'images': checks, 'elapsed_s': round(time.perf_counter() - t0, 2), 'maxrss_mib': maxrss_mib(),
                       'all_identical': all(c['sequence_identical'] for c in checks)})
        save()
        assert report['all_identical'], 'Wrapper sequences differ from generate'
        print('DONE selfcheck', flush=True)
        return

    target = bundle / f'{args.only}.onnx'
    assert not target.exists(), ('Refusing to overwrite an existing graph', str(target))
    t0 = time.perf_counter()
    if args.only == 'encoder':
        with torch.no_grad():
            torch.onnx.export(enc_w, image0, str(target), input_names=['image'], output_names=['encoder_hidden_states'],
                              dynamic_axes={'image': {0: 'batch'}, 'encoder_hidden_states': {0: 'batch'}},
                              opset_version=args.opset, do_constant_folding=folding)
    else:
        with torch.no_grad():
            enc_out = enc_w(image0)
        ids = torch.full((1, 1), prompt_id, dtype=torch.long)
        attn = torch.ones(1, 1, dtype=torch.long)
        if args.only == 'decoder':
            dyn = {'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0: 'batch', 1: 'seq'},
                   'encoder_hidden_states': {0: 'batch'}, 'logits': {0: 'batch', 1: 'seq'}}
            for L in range(num_layers):
                dyn[f'present_{L}_self_k'] = {0: 'batch', 2: 'seq'}
                dyn[f'present_{L}_self_v'] = {0: 'batch', 2: 'seq'}
                dyn[f'present_{L}_cross_k'] = {0: 'batch'}
                dyn[f'present_{L}_cross_v'] = {0: 'batch'}
            with torch.no_grad():
                torch.onnx.export(dec_w, (ids, attn, enc_out), str(target),
                                  input_names=['input_ids', 'attention_mask', 'encoder_hidden_states'],
                                  output_names=['logits'] + kv_out, dynamic_axes=dyn,
                                  opset_version=args.opset, do_constant_folding=folding)
        else:
            with torch.no_grad():
                d = dec_w(ids, attn, enc_out)
            flat_kv = list(d[1:])
            nxt = d[0][:, -1].argmax(-1, keepdim=True).long()
            attn2 = torch.ones(1, 2, dtype=torch.long)
            dyn = {'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch', 1: 'total_seq'},
                   'encoder_hidden_states': {0: 'batch'}, 'logits': {0: 'batch'}}
            for L in range(num_layers):
                dyn[f'past_{L}_self_k'] = {0: 'batch', 2: 'past_seq'}
                dyn[f'past_{L}_self_v'] = {0: 'batch', 2: 'past_seq'}
                dyn[f'past_{L}_cross_k'] = {0: 'batch'}
                dyn[f'past_{L}_cross_v'] = {0: 'batch'}
                dyn[f'present_{L}_self_k'] = {0: 'batch', 2: 'total_seq'}
                dyn[f'present_{L}_self_v'] = {0: 'batch', 2: 'total_seq'}
                dyn[f'present_{L}_cross_k'] = {0: 'batch'}
                dyn[f'present_{L}_cross_v'] = {0: 'batch'}
            with torch.no_grad():
                torch.onnx.export(dec_p_w, (nxt, attn2, enc_out, *flat_kv), str(target),
                                  input_names=['input_ids', 'attention_mask', 'encoder_hidden_states'] + past_in,
                                  output_names=['logits'] + kv_out, dynamic_axes=dyn,
                                  opset_version=args.opset, do_constant_folding=folding)
    report.update({'graph': target.name, 'bytes': target.stat().st_size,
                   'elapsed_s': round(time.perf_counter() - t0, 2), 'maxrss_mib': maxrss_mib()})
    save()
    print('EXPORTED', report['graph'], {k: report[k] for k in ('bytes', 'elapsed_s', 'maxrss_mib')}, flush=True)


if __name__ == '__main__':
    main()

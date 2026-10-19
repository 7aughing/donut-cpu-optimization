"""Run three Donut ONNX graphs with a Python greedy decoding loop.

Bundle layout:
  encoder.onnx            image -> encoder_hidden_states
  decoder.onnx            First step without cache: input_ids, attention_mask,
                          encoder_hidden_states -> logits and 16 present_*
                          self/cross-attention key/value cache tensors.
  decoder_with_past.onnx  Cached steps: input_ids, attention_mask, 16 past_*
                          tensors -> logits and present_* tensors. If the
                          cross-attention outputs are absent, reuse the K/V
                          tensors from the first decoder step.

Read outputs by name. Load sessions from file paths without retaining a
ModelProto. Return sequence: LongTensor[B, T] and probabilities:
FloatTensor[B, T-1], the maximum softmax probability after vocabulary masking
at each step.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import onnxruntime
import torch

KV_KINDS = ('self_k', 'self_v', 'cross_k', 'cross_v')
GRAPHS = ('encoder.onnx', 'decoder.onnx', 'decoder_with_past.onnx')


class _Session:
    """Wrap an ORT session and retain its path for rebuilding with options such as profiling."""

    def __init__(self, path: str, sess_options: Optional[onnxruntime.SessionOptions] = None,
                 providers: Optional[List[str]] = None):
        self.path = path
        self.sess = onnxruntime.InferenceSession(path, sess_options=sess_options,
                                                 providers=providers or ['CPUExecutionProvider'])

    def run(self, output_names, input_feed):
        return self.sess.run(output_names, input_feed)


class DonutOnnxOrchestrator:
    def __init__(self, bundle_dir: str, max_length: int, prompt_token_id: int, pad_token_id: int,
                 eos_token_id: int, bad_words_ids: Optional[List[List[int]]] = None,
                 forced_eos_token_id: Optional[int] = None, num_decoder_layers: int = 4,
                 sess_options: Optional[onnxruntime.SessionOptions] = None, providers: Optional[List[str]] = None):
        self.max_length = int(max_length)
        self.prompt_token_id = int(prompt_token_id)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.bad_words_ids = bad_words_ids or []
        self.forced_eos_token_id = forced_eos_token_id
        self.num_layers = int(num_decoder_layers)
        for name in GRAPHS:
            if not os.path.isfile(os.path.join(bundle_dir, name)):
                raise FileNotFoundError(f'{bundle_dir} is missing {name}')
        self._enc = _Session(os.path.join(bundle_dir, 'encoder.onnx'), sess_options, providers)
        self._dec = _Session(os.path.join(bundle_dir, 'decoder.onnx'), sess_options, providers)
        self._dec_p = _Session(os.path.join(bundle_dir, 'decoder_with_past.onnx'), sess_options, providers)
        self._dec_p_input_names = {i.name for i in self._dec_p.sess.get_inputs()}
        self._dec_output_names = [o.name for o in self._dec.sess.get_outputs()]
        self._dec_p_output_names = [o.name for o in self._dec_p.sess.get_outputs()]
        self._dec_p_emits_cross_kv = all(
            f'present_{L}_{kind}' in self._dec_p_output_names
            for L in range(self.num_layers) for kind in ('cross_k', 'cross_v'))

    def __call__(self, image) -> Dict[str, torch.Tensor]:
        return self.forward(image)

    @torch.no_grad()
    def forward(self, image) -> Dict[str, torch.Tensor]:
        np_image = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
        np_image = np_image.astype(np.float32)
        if np_image.ndim != 4:
            raise ValueError(f'Expected (B, 3, H, W), got {np_image.shape}')
        B = np_image.shape[0]

        enc_hs = self._enc.run(['encoder_hidden_states'], {'image': np_image})[0]

        prompt = np.full((B, 1), self.prompt_token_id, dtype=np.int64)
        first = dict(zip(self._dec_output_names, self._dec.run(None, {
            'input_ids': prompt, 'attention_mask': np.ones((B, 1), dtype=np.int64), 'encoder_hidden_states': enc_hs})))
        kv = {f'{L}_{kind}': first[f'present_{L}_{kind}'] for L in range(self.num_layers) for kind in KV_KINDS}

        bad_token_ids = [bw[0] for bw in self.bad_words_ids if len(bw) == 1]
        scores: List[np.ndarray] = []

        def pick(last_logits: np.ndarray, cur_len: int) -> np.ndarray:
            ll = last_logits.copy()
            if bad_token_ids:
                ll[:, bad_token_ids] = -np.inf
            if self.forced_eos_token_id is not None and cur_len == self.max_length - 1:
                forced = np.full_like(ll, -np.inf)
                forced[:, self.forced_eos_token_id] = 0.0
                ll = forced
            scores.append(ll)
            return ll.argmax(axis=-1, keepdims=True)

        seq = prompt.copy()
        finished = np.zeros(B, dtype=bool)
        next_tok = pick(first['logits'][:, -1, :], seq.shape[1])
        next_tok = np.where(finished[:, None], np.full_like(next_tok, self.pad_token_id), next_tok)
        seq = np.concatenate([seq, next_tok], axis=1)
        finished |= next_tok[:, 0] == self.eos_token_id

        kinds = KV_KINDS if self._dec_p_emits_cross_kv else ('self_k', 'self_v')
        while seq.shape[1] < self.max_length and not finished.all():
            feed = {'input_ids': next_tok, 'attention_mask': (seq != self.pad_token_id).astype(np.int64)}
            if 'encoder_hidden_states' in self._dec_p_input_names:
                feed['encoder_hidden_states'] = enc_hs
            for L in range(self.num_layers):
                for kind in KV_KINDS:
                    feed[f'past_{L}_{kind}'] = kv[f'{L}_{kind}']
            step = dict(zip(self._dec_p_output_names, self._dec_p.run(None, feed)))
            for L in range(self.num_layers):
                for kind in kinds:
                    kv[f'{L}_{kind}'] = step[f'present_{L}_{kind}']
            next_tok = pick(step['logits'][:, -1, :], seq.shape[1])
            next_tok = np.where(finished[:, None], np.full_like(next_tok, self.pad_token_id), next_tok)
            seq = np.concatenate([seq, next_tok], axis=1)
            finished |= next_tok[:, 0] == self.eos_token_id

        probs = torch.stack([torch.softmax(torch.from_numpy(s.astype(np.float32)), dim=-1).max(dim=-1).values
                             for s in scores]).T
        return {'sequence': torch.from_numpy(seq).long(), 'probabilities': probs}

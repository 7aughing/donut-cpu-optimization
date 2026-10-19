"""Compare per-image results from two results_*.jsonl files.

Match images by filename and compare token sequences, token probabilities,
decoded text, parsed JSON, and paired latency. For a changed sequence, record
the first differing position and the corresponding probabilities.
Usage: python donut_compare_runs.py <reference.jsonl> <new.jsonl> [--out <json>]
"""
import argparse
import json
import statistics


def load(p):
    rows = [json.loads(line) for line in open(p) if line.strip()]
    by = {r['file']: r for r in rows}
    assert len(by) == len(rows), ('Duplicate image records', p)
    return by


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref')
    parser.add_argument('new')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    ref, new = load(args.ref), load(args.new)
    common = sorted(set(ref) & set(new))
    seq_diff, text_diff, parsed_diff = [], 0, 0
    max_tok = 0.0
    for k in common:
        a, b = ref[k], new[k]
        assert a['image_sha256'] == b['image_sha256'], ('Image hashes differ', k)
        if a['sequence'] != b['sequence']:
            sa, sb = a['sequence'], b['sequence']
            i = next((j for j in range(min(len(sa), len(sb))) if sa[j] != sb[j]), min(len(sa), len(sb)))
            p = i - 1  # Probabilities exclude the initial task token in the generated sequence.
            seq_diff.append({'file': k, 'first_diff_index': i, 'ref_len': len(sa), 'new_len': len(sb),
                             'ref_token': sa[i] if i < len(sa) else None, 'new_token': sb[i] if i < len(sb) else None,
                             'ref_prob': a['token_probabilities'][p] if 0 <= p < len(a['token_probabilities']) else None,
                             'new_prob': b['token_probabilities'][p] if 0 <= p < len(b['token_probabilities']) else None})
        else:
            max_tok = max(max_tok, max(abs(x - y) for x, y in zip(a['token_probabilities'], b['token_probabilities'])))
        text_diff += a['text'] != b['text']
        parsed_diff += a['parsed'] != b['parsed']

    def ratio(key):
        return statistics.median(new[k][key] / ref[k][key] for k in common)

    result = {
        'ref': args.ref, 'new': args.new, 'ref_images': len(ref), 'new_images': len(new), 'common': len(common),
        'sequence_identical': len(common) - len(seq_diff), 'sequence_changed': len(seq_diff),
        'text_changed': text_diff, 'parsed_changed': parsed_diff,
        'max_token_prob_delta_on_identical': max_tok,
        'sequence_diffs': seq_diff,
        'ref_median_ms': statistics.median(ref[k]['elapsed_ms'] for k in common),
        'new_median_ms': statistics.median(new[k]['elapsed_ms'] for k in common),
        'median_paired_time_ratio': ratio('elapsed_ms'),
        'ref_encoder_median_ms': statistics.median(ref[k]['encoder_ms'] for k in common),
        'new_encoder_median_ms': statistics.median(new[k]['encoder_ms'] for k in common),
        'median_paired_encoder_ratio': ratio('encoder_ms'),
        'ref_decoder_median_ms': statistics.median(ref[k]['decoder_ms'] for k in common),
        'new_decoder_median_ms': statistics.median(new[k]['decoder_ms'] for k in common),
        'median_paired_decoder_ratio': ratio('decoder_ms'),
        'new_faster_count': sum(new[k]['elapsed_ms'] < ref[k]['elapsed_ms'] for k in common),
        'BITWISE_IDENTICAL': not seq_diff and text_diff == 0 and parsed_diff == 0 and max_tok == 0.0,
    }
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        open(args.out, 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()

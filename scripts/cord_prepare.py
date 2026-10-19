"""Extract CORD-v2 test images and manifest.json from a Parquet file.

Requires PyArrow and the standard library. Write the original image bytes
without re-encoding them.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--out', required=True, help='Output directory; must not already exist')
    args = parser.parse_args()

    src = Path(args.parquet)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()

    table = pq.read_table(src)
    print('rows', table.num_rows, 'columns', table.column_names, flush=True)
    assert table.column_names == ['image', 'ground_truth'], table.column_names

    items = []
    for idx, row in enumerate(table.to_pylist()):
        data = row['image']['bytes']
        assert data[:8] == b'\x89PNG\r\n\x1a\n', ('Not a PNG image', idx)
        gt = json.loads(row['ground_truth'])
        meta = gt.get('meta', {})
        name = f'cord_test_{idx:03d}.png'
        (out / 'images' / name).write_bytes(data)
        items.append({'index': idx, 'file': name, 'path': str(out / 'images' / name),
                      'bytes': len(data), 'sha256': sha256(data),
                      'image_id': meta.get('image_id'), 'image_size': meta.get('image_size'),
                      'ground_truth': row['ground_truth']})

    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_parquet': str(src), 'source_sha256': sha256(src.read_bytes()),
        'dataset': 'naver-clova-ix/cord-v2', 'revision': '7f0115a4b758a71d6473b8d085751692da2fef98',
        'split': 'test', 'license': 'CC BY 4.0', 'pyarrow': pyarrow.__version__,
        'count': len(items), 'images': items,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('WROTE', len(items), 'images to', out / 'images', flush=True)


if __name__ == '__main__':
    main()

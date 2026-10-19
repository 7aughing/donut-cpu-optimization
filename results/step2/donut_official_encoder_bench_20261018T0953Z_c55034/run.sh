#!/bin/bash
cd <HOME>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=<VENV>/bin/python
COMMON="--model-dir <WORK_DIR>/donut_official_assets/model --bundle-dir <WORK_DIR>/donut_official_onnx_raw_20261018T0950Z_967829/bundle --encoder-sha256 fa5c2951233a063e3ffc498dfc1c21661757ba7df1ed0ee093e27d6d9768169b --manifest <WORK_DIR>/donut_official_assets/cord_v2/test/manifest.json --n-images 10 --out <WORK_DIR>/donut_official_encoder_bench_20261018T0953Z_c55034 --extra-path <WORK_DIR>/donut_official_assets/pylib_sentencepiece"
echo "START parity $(date -u +%FT%TZ)"
$PY -I <WORK_DIR>/donut_official_assets/scripts/donut_encoder_bench.py --mode parity $COMMON || { echo FAIL parity; exit 1; }
for r in 0 1; do for b in torch ort; do
  echo "START time $b round $r $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
  $PY -I <WORK_DIR>/donut_official_assets/scripts/donut_encoder_bench.py --mode time --backend $b --round $r --repeats 3 $COMMON || { echo FAIL $b $r; exit 1; }
done; done
echo "ALL_DONE $(date -u +%FT%TZ)"

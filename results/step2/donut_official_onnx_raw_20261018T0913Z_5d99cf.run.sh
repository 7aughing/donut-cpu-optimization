#!/bin/bash
cd <HOME>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
echo "START $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
<VENV>/bin/python -I <WORK_DIR>/donut_official_assets/scripts/donut_onnx_export.py   --model-dir <WORK_DIR>/donut_official_assets/model --manifest <WORK_DIR>/donut_official_assets/cord_v2/test/manifest.json --out <WORK_DIR>/donut_official_onnx_raw_20261018T0913Z_5d99cf --extra-path <WORK_DIR>/donut_official_assets/pylib_sentencepiece
echo "EXIT $? $(date -u +%FT%TZ)"

#!/bin/bash
cd <HOME>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
echo "START $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
<VENV>/bin/python -I <WORK_DIR>/donut_official_assets/scripts/donut_cpu_probe.py   --model-dir <WORK_DIR>/donut_official_assets/model --manifest <WORK_DIR>/donut_official_assets/cord_v2/test/manifest.json --start 0 --end 100 --out <WORK_DIR>/donut_official_baseline_20261018T0816Z_9d24e0 --tag torch_fp32   --extra-path <WORK_DIR>/donut_official_assets/pylib_sentencepiece
echo "EXIT $? $(date -u +%FT%TZ)"

#!/usr/bin/env bash
# Benchmark two ONNX backends in 4 rounds of 25 images, alternating process order.
# Use the experiment venv and ORT threads 2/1; the probe loads graphs from file paths.
# Usage: bash donut_run_pair_onnx.sh <OUT_DIR> <A_NAME> <A_BUNDLE> <A_MANIFEST> <B_NAME> <B_BUNDLE> <B_MANIFEST>
set -u
OUT="$1"
NAMES=("$2" "$5"); BUNDLES=("$3" "$6"); MANIFESTS=("$4" "$7")
A=${WORK_DIR:?Set WORK_DIR to the experiment workspace}/donut_official_assets
PY=${VENV:?Set VENV to the experiment virtual environment}/bin/python
CHUNK=25
mkdir -p "$OUT"
cd "$HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
echo "START $(date -u +%FT%TZ) OUT=$OUT A=${NAMES[0]}:${BUNDLES[0]} B=${NAMES[1]}:${BUNDLES[1]}" | tee -a "$OUT/driver.log"
for round in 0 1 2 3; do
  start=$((round * CHUNK)); end=$((start + CHUNK))
  for pos in 0 1; do
    i=$(( (round + pos) % 2 ))
    backend=${NAMES[$i]}
    echo "RUN round=$round pos=$pos backend=$backend slice=$start-$end $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/driver.log"
    /usr/bin/timeout 1800s "$PY" -I "$A/scripts/donut_cpu_probe.py" --backend "$backend" \
      --model-dir "$A/model" --manifest "$A/cord_v2/test/manifest.json" --start "$start" --end "$end" \
      --out "$OUT" --extra-path "$A/pylib_sentencepiece" --round "$round" --position "$pos" \
      --bundle-dir "${BUNDLES[$i]}" --export-manifest "${MANIFESTS[$i]}" --runner-dir "$A/scripts" \
      --threads 2 --ort-intra 2 --ort-inter 1 >> "$OUT/log_${backend}.txt" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "FAILED round=$round backend=$backend rc=$rc $(date -u +%FT%TZ)" | tee -a "$OUT/driver.log"
      touch "$OUT/FAILED"; exit $rc
    fi
    tail -1 "$OUT/log_${backend}.txt" | tee -a "$OUT/driver.log"
  done
done
echo "END $(date -u +%FT%TZ)" | tee -a "$OUT/driver.log"
touch "$OUT/DONE"

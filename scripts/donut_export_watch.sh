#!/bin/bash
# Run the exporter with a watchdog sampling cgroup anonymous memory every 0.2 s.
# Kill the exporter above 7 GiB to avoid container OOM; trace to <OUT>/memtrace_<ONLY>.log.
# Usage: donut_export_watch.sh <only> <out_dir> [extra exporter arguments...]
ONLY=$1; OUT=$2; shift 2
A=${WORK_DIR:?Set WORK_DIR to the experiment workspace}/donut_official_assets
PY=${VENV:?Set VENV to the experiment virtual environment}/bin/python
LIMIT=$((7 * 1024 * 1024 * 1024))
mkdir -p "$OUT"
cd "$HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
echo "START $ONLY $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
"$PY" -I "$A/scripts/donut_onnx_export.py" --only "$ONLY" --model-dir "$A/model" \
  --manifest "$A/cord_v2/test/manifest.json" --out "$OUT" --extra-path "$A/pylib_sentencepiece" "$@" &
PID=$!
PEAK=0; KILLED=0
while kill -0 "$PID" 2>/dev/null; do
  ANON=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat)
  RSS=$(awk '/^VmRSS/{print $2 * 1024}' "/proc/$PID/status" 2>/dev/null)
  CUR=$(cat /sys/fs/cgroup/memory.current)
  [ "$ANON" -gt "$PEAK" ] && PEAK=$ANON
  echo "$(date +%s.%N) anon=$ANON rss=${RSS:-0} current=$CUR" >> "$OUT/memtrace_$ONLY.log"
  if [ "$ANON" -gt "$LIMIT" ]; then
    kill -9 "$PID"; KILLED=1
    echo "WATCHDOG_KILL anon=$ANON $(date -u +%FT%TZ)"
    break
  fi
  sleep 0.2
done
wait "$PID"; RC=$?
echo "EXIT $ONLY rc=$RC killed=$KILLED peak_anon=$PEAK $(date -u +%FT%TZ)"

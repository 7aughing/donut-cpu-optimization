#!/bin/bash
# Run a command and sample cgroup anonymous memory every 0.2 s; kill it above 7 GiB.
# Trace to <OUT>/memtrace_<TAG>.log using the same guard as donut_export_watch.sh.
# Usage: donut_mem_watch.sh <tag> <out_dir> <command> [arguments...]
TAG=$1; OUT=$2; shift 2
LIMIT=$((7 * 1024 * 1024 * 1024))
mkdir -p "$OUT"
cd "$HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
echo "START $TAG $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
"$@" &
PID=$!
PEAK=0; KILLED=0
while kill -0 "$PID" 2>/dev/null; do
  ANON=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat)
  RSS=$(awk '/^VmRSS/{print $2 * 1024}' "/proc/$PID/status" 2>/dev/null)
  [ "$ANON" -gt "$PEAK" ] && PEAK=$ANON
  echo "$(date +%s.%N) anon=$ANON rss=${RSS:-0}" >> "$OUT/memtrace_$TAG.log"
  if [ "$ANON" -gt "$LIMIT" ]; then
    kill -9 "$PID"; KILLED=1
    echo "WATCHDOG_KILL anon=$ANON $(date -u +%FT%TZ)"
    break
  fi
  sleep 0.2
done
wait "$PID"; RC=$?
echo "EXIT $TAG rc=$RC killed=$KILLED peak_anon=$PEAK $(date -u +%FT%TZ)"

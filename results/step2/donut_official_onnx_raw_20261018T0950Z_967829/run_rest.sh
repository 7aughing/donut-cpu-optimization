#!/bin/bash
for s in decoder decoder_with_past check; do
  <WORK_DIR>/donut_official_assets/scripts/donut_export_watch.sh $s <WORK_DIR>/donut_official_onnx_raw_20261018T0950Z_967829 > <WORK_DIR>/donut_official_onnx_raw_20261018T0950Z_967829/watch_$s.log 2>&1
  grep -q "rc=0 killed=0" <WORK_DIR>/donut_official_onnx_raw_20261018T0950Z_967829/watch_$s.log || { echo "STOP at $s"; exit 1; }
done
echo ALL_DONE

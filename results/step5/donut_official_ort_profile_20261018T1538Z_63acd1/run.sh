#!/bin/bash
W=<WORK_DIR>/donut_official_assets/scripts/donut_mem_watch.sh
for spec in "onnx_mha2 <WORK_DIR>/donut_official_onnx_mha2_20261018T1250Z_26c1ed <WORK_DIR>/donut_official_onnx_mha2_20261018T1250Z_26c1ed/fuse_manifest.json" "onnx_mha2_xkv <WORK_DIR>/donut_official_onnx_mha2_xkv_20261018T1337Z_1ad308 <WORK_DIR>/donut_official_onnx_mha2_xkv_20261018T1337Z_1ad308/xkv_manifest.json"; do
  set -- $spec
  $W prof_$1 <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1 <VENV>/bin/python -I <WORK_DIR>/donut_official_assets/scripts/donut_ort_profile.py --backend $1 --bundle-dir $2/bundle --graphs-manifest $3     --out <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1/$1 --model-dir <WORK_DIR>/donut_official_assets/model --manifest <WORK_DIR>/donut_official_assets/cord_v2/test/manifest.json --n-images 3     --extra-path <WORK_DIR>/donut_official_assets/pylib_sentencepiece --runner-dir <WORK_DIR>/donut_official_assets/scripts > <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1/watch_prof_$1.log 2>&1
  grep -q "rc=0 killed=0" <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1/watch_prof_$1.log || { echo STOP prof $1; exit 1; }
  $W window_$1 <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1 <VENV>/bin/python -I <WORK_DIR>/donut_official_assets/scripts/donut_prof_window.py <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1/$1 > <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1/watch_window_$1.log 2>&1
  grep -q "rc=0 killed=0" <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1/watch_window_$1.log || { echo STOP window $1; exit 1; }
  grep -q "ALL_COMPLETE True" <WORK_DIR>/donut_official_ort_profile_20261018T1538Z_63acd1/watch_window_$1.log || { echo STOP incomplete $1; exit 1; }
done
echo ALL_DONE

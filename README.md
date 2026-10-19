# Donut CPU Inference Optimization

English | [简体中文](README.zh-CN.md)

CPU inference optimization scripts and archived experiment results for
[`naver-clova-ix/donut-base-finetuned-cord-v2`](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2).
The workflow exports three ONNX graphs, fuses cached-decoder attention, and
reuses cross-attention key/value (K/V) tensors across decoding steps.
This is a community project using the official checkpoint.

[Results](#benchmark-results) · [Quick start](#quick-start-inspect-the-results) ·
[Reproduction](#reproduce-the-experiment) · [Scripts](#script-guide) ·
[Scope](#scope-and-reproducibility) · [Comparison with onnx-donut](#comparison-with-ikomia-devonnx-donut)

## Benchmark results

The final ONNX variant recorded **5,063 ms median latency per image**, compared
with **7,153 ms for PyTorch FP32**: a **29.2% reduction**, or **1.41x speedup**,
calculated from the two medians. All 100 images retained the same generated
token sequences, decoded text, and parsed JSON as the PyTorch reference.
All full-dataset performance numbers come from the original archived experiment.
The current scripts have passed a separate [three-image minimal reproduction
on Mac with Docker](#scope-and-reproducibility); the 100-image performance
benchmark was not rerun during that check.

### Test conditions

The runs used all 100 CORD-v2 test images on October 18, 2026 (UTC), batch
size 1, greedy decoding, and one process in a Linux CPU container with a
4-core CPU quota and an 8 GiB memory limit. The recorded software was Python
3.9.6, PyTorch 2.0.1+cpu, Transformers 4.25.1, and ONNX Runtime 1.18.1.
PyTorch used 2 intra-op threads; ONNX Runtime used 2 intra-op and 1 inter-op
thread. Timing includes preprocessing, generation, token probabilities,
and JSON parsing; image loading and decoding are excluded.

Each optimization stage was compared with its predecessor across four rounds
of 25 images, alternating process order. The overview below selects results
from different stages; it is not one simultaneous four-backend benchmark.

| Backend | Median latency | P95 latency | Token sequences matching PyTorch | Source |
|---|---:|---:|---:|---|
| PyTorch FP32 | 7,153 ms | 10,327 ms | Reference | [Step 2](results/step2/donut_official_pair_raw_20261018T1012Z_c11161/summary_torch_fp32.json) |
| Raw ONNX | 5,331 ms | 7,228 ms | 100 / 100 | [Step 2](results/step2/donut_official_pair_raw_20261018T1012Z_c11161/summary_onnx_raw.json) |
| Fused ONNX | 5,290 ms | 6,965 ms | 100 / 100 | [Step 3](results/step3/donut_official_pair_mha2_20261018T1253Z_512992/summary_onnx_mha2.json) |
| Fused ONNX + cross-attention K/V reuse | 5,063 ms | 6,382 ms | 100 / 100 | [Step 4](results/step4/donut_official_pair_xkv_20261018T1339Z_494003/summary_onnx_mha2_xkv.json) |

### Gains at each optimization stage

The largest measured gain came from running the exported ONNX graphs.
Fusion and K/V reuse further reduced decoder time. Each row below uses the
reference rerun from that same stage, so its reference median can differ
from the overview table.

| Paired comparison | Median latency, reference → optimized | Paired latency reduction | Paired decoder reduction | Images faster | Evidence |
|---|---:|---:|---:|---:|---|
| PyTorch FP32 → raw ONNX | 7,153 → 5,331 ms | 26.1% | 40.4% | 100 / 100 | [Comparison](results/step2/donut_official_pair_raw_20261018T1012Z_c11161/compare_onnx_raw_vs_torch_fp32.json) |
| Raw ONNX → fused ONNX | 5,349 → 5,290 ms | 1.6% | 7.6% | 97 / 100 | [Comparison](results/step3/donut_official_pair_mha2_20261018T1253Z_512992/compare_onnx_mha2_vs_onnx_raw.json) |
| Fused ONNX → K/V reuse | 5,271 → 5,063 ms | 4.1% | 20.3% | 99 / 100 | [Comparison](results/step4/donut_official_pair_xkv_20261018T1339Z_494003/compare_onnx_mha2_xkv_vs_onnx_mha2.json) |

Paired reduction is `100 × (1 − median(optimized_ms[i] / reference_ms[i]))`
over the same 100 images. It is distinct from the ratio of aggregate medians
used for the 29.2% headline; the stage percentages should not be added together.

The graph changes are:

- **Raw ONNX:** separate encoder, first-step decoder, and cached decoder graphs,
  driven by a Python greedy decoding loop.
- **Fusion:** fuse four cross-attention `MultiHeadAttention` operators in the
  cached decoder, along with normalization and activation patterns, then simplify
  the graph. See the [fusion manifest](results/step3/donut_official_onnx_mha2_20261018T1250Z_26c1ed/fuse_manifest.json).
- **K/V reuse:** retain the cross-attention keys and values produced at the first
  decoding step, and omit their repeated outputs from the cached decoder.
  The encoder, first-step decoder, and weight initializers are unchanged.
  See the [K/V manifest](results/step4/donut_official_onnx_mha2_xkv_20261018T1337Z_1ad308/xkv_manifest.json).

### Output agreement and remaining bottleneck

The [final comparison with PyTorch](results/step4/donut_official_pair_xkv_20261018T1339Z_494003/compare_onnx_mha2_xkv_vs_step2_torch_fp32.json)
reports 100/100 identical token sequences, zero changed texts, and zero changed
parsed JSON outputs. The maximum absolute token-probability difference is
`1.9073486328125e-05`; this is output agreement, not bitwise numerical identity
or an accuracy score against ground truth. K/V reuse versus fused ONNX has
zero token-probability difference on the same 100 images.

The final [100-image summary](results/step4/donut_official_pair_xkv_20261018T1339Z_494003/summary_onnx_mha2_xkv.json) records
4,138 ms median encoder time and 814 ms median decoder time. Encoder time
remains the largest component; segment medians do not sum to total median latency.
A separate profile of the first three test images attributes 74.3% of encoder
`model_run` time to `MatMul`, and 32.7% of cached-decoder `model_run` time to
the output vocabulary projection (`lm_head`). See the [operator breakdown](results/step5/donut_official_ort_profile_20261018T1543Z_f7db2a/onnx_mha2_xkv/window.json).
The profiler covered 3 encoder, 3 first-step decoder, and 160 cached-decoder
calls after excluding warmup. These small-sample operator shares describe
where time was spent; they are separate from the 100-image latency benchmark.

## Comparison with Ikomia-dev/onnx-donut

[Ikomia-dev/onnx-donut](https://github.com/Ikomia-dev/onnx-donut/blob/bbde423d4cc63d5d6a727f20c483adbe8467bc9c/README.md) provides an installable package
for Donut ONNX export and inference. This repository focuses on CPU optimization
and measurements for one pinned CORD-v2 checkpoint. The comparison below refers
to Ikomia commit [`bbde423`](https://github.com/Ikomia-dev/onnx-donut/commit/bbde423d4cc63d5d6a727f20c483adbe8467bc9c),
reviewed on October 19, 2026.

| Area | Ikomia-dev/onnx-donut | This repository |
|---|---|---|
| Usage | `pip install onnx-donut`; `export_onnx` and `OnnxPredictor.generate`; README example uses DocVQA | Scripts for the pinned CORD-v2 checkpoint, export, optimization, and paired CPU measurements |
| Model loading | [Exporter](https://github.com/Ikomia-dev/onnx-donut/blob/bbde423d4cc63d5d6a727f20c483adbe8467bc9c/src/onnx_donut/exporter.py) uses `donut.DonutModel.from_pretrained` | [Exporter](scripts/donut_onnx_export.py) uses Transformers `VisionEncoderDecoderModel` and `DonutProcessor`, with weight-hash validation |
| Graphs and caches | Three graphs; [predictor](https://github.com/Ikomia-dev/onnx-donut/blob/bbde423d4cc63d5d6a727f20c483adbe8467bc9c/src/onnx_donut/predictor.py) passes self- and cross-attention caches between steps and reads cache outputs each step | Same three-graph pattern; the final variant omits repeated cross-attention K/V outputs from the cached decoder and retains the first-step values in the [runner](scripts/donut_onnx_runner.py) |
| Optimization and precision | Optional [dynamic INT8 quantization](https://github.com/Ikomia-dev/onnx-donut/blob/bbde423d4cc63d5d6a727f20c483adbe8467bc9c/src/onnx_donut/quantizer.py), excluding `Conv` | FP32 attention fusion, graph simplification, and removal of repeated cross-attention K/V outputs; no quantization |

Three-graph export and cached decoding are shared techniques. The work documented
here is the staged FP32 optimization, paired 100-image measurements, per-image
output checks, and operator profiling for the pinned checkpoint. The reported
**1.41x** speedup compares our final ONNX variant with our archived PyTorch FP32
reference. Neither the original experiment nor the Mac check benchmarked
Ikomia's implementation, so these results do not establish a speed advantage
over `onnx-donut`.

## Quick start: inspect the results

The archived results can be compared with Python 3 and the standard library.
No model download or inference dependencies are needed. Set `REPO` to the
absolute path of this directory:

```bash
export REPO=/absolute/path/to/donut-cpu-optimization
python3 "$REPO/scripts/donut_compare_runs.py" \
  "$REPO/results/step2/donut_official_pair_raw_20261018T1012Z_c11161/results_torch_fp32.jsonl" \
  "$REPO/results/step4/donut_official_pair_xkv_20261018T1339Z_494003/results_onnx_mha2_xkv.jsonl"
```
The output should report `common: 100`, `sequence_identical: 100`,
`text_changed: 0`, and `parsed_changed: 0`. This checks recorded outputs; it
does not run the model.

## Repository layout

| Path | Contents |
|---|---|
| [scripts/](scripts/) | Export, optimization, inference, and timing scripts; English comments and CLI messages |
| [results/](results/) | Per-image JSONL, run manifests, summaries, comparisons, and logs |
| [requirements.txt](requirements.txt) | Pinned direct dependencies based on the recorded Linux experiment environment |

## Reproduce the experiment

### 1. Prepare the environment

The recorded environment used Linux and Python 3.9.6. The shell drivers use
`/proc`, cgroup v2 files, and GNU `timeout`; run them in a Linux environment.
The pinned dependencies and the three-image model workflow have been verified
in Docker Desktop on an Apple Silicon Mac, using `python:3.9.6-slim` with
`linux/amd64` via Rosetta, a 4-CPU quota, and a 7 GiB container memory limit.
Run the commands below inside Linux, including when using Docker on Mac.
See [Scope and reproducibility](#scope-and-reproducibility) for the verified steps.

Use a **new workspace outside this repository** so generated models and runs
stay separate from the archived evidence. The following commands assume Bash:

```bash
export REPO=/absolute/path/to/donut-cpu-optimization
export WORK_DIR=/absolute/path/to/a/new/donut-experiment
export VENV="$WORK_DIR/venv"
export A="$WORK_DIR/donut_official_assets"

mkdir -p "$WORK_DIR" "$A"
python3.9 -m venv "$VENV"
export PY="$VENV/bin/python"
"$PY" -m pip install -r "$REPO/requirements.txt"
cp -R "$REPO/scripts" "$A/scripts"
```
The dependency file installs SentencePiece and PyArrow in the same virtual
environment. The scripts also accept `--extra-path` for a separate dependency
directory, as used in the original experiment.

### 2. Prepare the checkpoint and dataset

Download these pinned upstream assets into the layout below:

- [Donut CORD-v2 checkpoint at revision `8003d433`](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2/tree/8003d433113256b4ce3a0f5bf604b29ff78a7451): save the checkpoint's weights, configuration, processor, and tokenizer files under `$A/model/`.
- [CORD-v2 test Parquet at revision `7f0115a4`](https://huggingface.co/datasets/naver-clova-ix/cord-v2/blob/7f0115a4b758a71d6473b8d085751692da2fef98/data/test-00000-of-00001-9c204eb3f4e11791.parquet): save the actual Parquet file under `$A/cord_v2/`.

`pytorch_model.bin` contains the model weights. Keep it together with the
configuration, processor, and tokenizer files from the same revision in
`$A/model/`. Pass this directory to `--model-dir`; the scripts do not accept
the `.bin` file itself as that argument.

After installing the dependencies in step 1, run the following command with
network access. It downloads the complete PyTorch checkpoint assets at the
fixed revision and verifies the weight hash. `huggingface_hub` is already
included in `requirements.txt`; the download uses its
[`snapshot_download` API](https://huggingface.co/docs/huggingface_hub/v0.25.2/en/package_reference/file_download#huggingface_hub.snapshot_download).

```bash
"$PY" - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_dir = Path(os.environ["A"]) / "model"
snapshot_download(
    repo_id="naver-clova-ix/donut-base-finetuned-cord-v2",
    revision="8003d433113256b4ce3a0f5bf604b29ff78a7451",
    local_dir=str(model_dir),
    allow_patterns=["pytorch_model.bin", "*.json", "*.model", "README.md"],
)

sha256 = hashlib.sha256()
with (model_dir / "pytorch_model.bin").open("rb") as weights:
    for chunk in iter(lambda: weights.read(1024 * 1024), b""):
        sha256.update(chunk)
expected = "31b78e3d3891072de8e2bf3553b71782242a1f3b589b914ec2b03feff7b14c54"
if sha256.hexdigest() != expected:
    raise RuntimeError("Model weight SHA256 mismatch")
print(f"Checkpoint ready: {model_dir}")
PY
```

The required local files are shown below. The download also includes the
upstream model card and may create download metadata under `model/.cache/`.
Download the test Parquet from the dataset link above into `$A/cord_v2/` before
running the extraction command.

```text
$WORK_DIR/
├── venv/
└── donut_official_assets/
    ├── scripts/
    ├── model/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   ├── preprocessor_config.json
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   ├── sentencepiece.bpe.model
    │   ├── special_tokens_map.json
    │   └── added_tokens.json
    └── cord_v2/
        └── test-00000-of-00001-9c204eb3f4e11791.parquet
```
Expected SHA256 values:

| File | SHA256 |
|---|---|
| `pytorch_model.bin` | `31b78e3d3891072de8e2bf3553b71782242a1f3b589b914ec2b03feff7b14c54` |
| Test Parquet | `51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7` |

Extract the original PNG bytes and create the manifest; the output directory
must not already exist:

```bash
"$PY" -I "$A/scripts/cord_prepare.py" \
  --parquet "$A/cord_v2/test-00000-of-00001-9c204eb3f4e11791.parquet" \
  --out "$A/cord_v2/test"
```
### 3. Export and optimize

The [export driver](scripts/donut_export_watch.sh) passes `--model-dir "$A/model"`
to [donut_onnx_export.py](scripts/donut_onnx_export.py). The exporter checks
`pytorch_model.bin` against the expected SHA256 and uses Transformers to load
the model directory. Its loading calls are:

```python
processor = DonutProcessor.from_pretrained(str(model_dir))
model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).eval()
```

Transformers reads the configuration and weights to construct the model;
the processor loads the image preprocessing and tokenizer settings. The
exporter wraps the encoder and decoder modules and calls `torch.onnx.export`.
The source `.bin` file stays in place; the generated ONNX files go to the
separate output directory. The driver enables offline mode, so complete the
checkpoint and dataset downloads before starting it.

The export driver runs each action in a separate process. Inspect each
`EXIT` line for `rc=0` and `killed=0` and confirm that `export_manifest.json`
exists before continuing. The memory watchdog terminates its child when
cgroup anonymous memory exceeds 7 GiB; its final log line carries the child's
status, so the shell exit status alone is insufficient.

```bash
export RAW="$WORK_DIR/onnx_raw"
export FUSED="$WORK_DIR/onnx_fused"
export XKV="$WORK_DIR/onnx_cross_kv_reuse"

for step in selfcheck encoder decoder decoder_with_past check; do
  bash "$A/scripts/donut_export_watch.sh" "$step" "$RAW"
done
```
A successful run produces the following files:

```text
$RAW/
├── export_manifest.json
└── bundle/
    ├── encoder.onnx
    ├── decoder.onnx
    └── decoder_with_past.onnx
```

`encoder.onnx` converts an image tensor into encoder features;
`decoder.onnx` produces the first token's logits and initial caches;
`decoder_with_past.onnx` consumes those caches for subsequent decoding steps.
`export_manifest.json` records the graph hashes and checks. Subsequent fusion
and K/V-reuse steps operate on these ONNX graphs.

Apply attention fusion, graph simplification, and cross-attention K/V reuse:

```bash
for step in fuse simplify finalize; do
  bash "$A/scripts/donut_mem_watch.sh" "$step" "$FUSED" \
    "$PY" -I "$A/scripts/donut_fuse_mha.py" \
    --step "$step" --raw-bundle "$RAW/bundle" --out "$FUSED" \
    --model-dir "$A/model" --manifest "$A/cord_v2/test/manifest.json"
done

bash "$A/scripts/donut_mem_watch.sh" make_xkv "$XKV" \
  "$PY" -I "$A/scripts/donut_make_xkv_bundle.py" \
  --src "$FUSED/bundle" --out "$XKV"
```
Check each watchdog log, `fuse_manifest.json`, and `xkv_manifest.json` before
benchmarking. Fusion and K/V-reuse scripts enforce the archived input-graph
hashes. A new export can produce a different hash in a different environment;
investigate the mismatch instead of removing the assertions.

### 4. Run inference and benchmark

Start with the first three test images. Predictions, token sequences, parsed
JSON, and timings are written to `results_onnx_mha2_xkv.jsonl`:

```bash
"$PY" -I "$A/scripts/donut_cpu_probe.py" \
  --backend onnx_mha2_xkv \
  --model-dir "$A/model" --manifest "$A/cord_v2/test/manifest.json" \
  --start 0 --end 3 --out "$WORK_DIR/smoke_xkv" \
  --bundle-dir "$XKV/bundle" --export-manifest "$XKV/xkv_manifest.json" \
  --runner-dir "$A/scripts" --threads 2 --ort-intra 2 --ort-inter 1
```
For full paired comparisons, use fresh output directories. The probe resumes
existing result files by skipping completed images; reuse is not a clean rerun.

```bash
bash "$A/scripts/donut_run_pair.sh" "$WORK_DIR/bench_raw" "$RAW"

bash "$A/scripts/donut_run_pair_onnx.sh" "$WORK_DIR/bench_fused" \
  onnx_raw "$RAW/bundle" "$RAW/export_manifest.json" \
  onnx_mha2 "$FUSED/bundle" "$FUSED/fuse_manifest.json"

bash "$A/scripts/donut_run_pair_onnx.sh" "$WORK_DIR/bench_xkv" \
  onnx_mha2 "$FUSED/bundle" "$FUSED/fuse_manifest.json" \
  onnx_mha2_xkv "$XKV/bundle" "$XKV/xkv_manifest.json"

"$PY" -I "$A/scripts/donut_compare_runs.py" \
  "$WORK_DIR/bench_xkv/results_onnx_mha2.jsonl" \
  "$WORK_DIR/bench_xkv/results_onnx_mha2_xkv.jsonl" \
  --out "$WORK_DIR/bench_xkv/comparison.json"

"$PY" -I "$A/scripts/donut_summarize.py" \
  --run-dir "$WORK_DIR/bench_xkv" --tag onnx_mha2_xkv
```
## Script guide

| Task | Entry point |
|---|---|
| Extract CORD images and manifest | [cord_prepare.py](scripts/cord_prepare.py) |
| Export and check three ONNX graphs | [donut_onnx_export.py](scripts/donut_onnx_export.py) |
| Fuse and simplify cached-decoder attention | [donut_fuse_mha.py](scripts/donut_fuse_mha.py) |
| Omit repeated cross-attention K/V outputs | [donut_make_xkv_bundle.py](scripts/donut_make_xkv_bundle.py) |
| Run the three-graph greedy decoding loop | [donut_onnx_runner.py](scripts/donut_onnx_runner.py) (importable module) |
| Time per-image inference and record predictions | [donut_cpu_probe.py](scripts/donut_cpu_probe.py) |
| Compare outputs / summarize timings | [donut_compare_runs.py](scripts/donut_compare_runs.py) / [donut_summarize.py](scripts/donut_summarize.py) |
| Check and time the encoder separately | [donut_encoder_bench.py](scripts/donut_encoder_bench.py) |
| Collect ORT profiles / summarize operator windows | [donut_ort_profile.py](scripts/donut_ort_profile.py) / [donut_prof_window.py](scripts/donut_prof_window.py) |

## Scope and reproducibility

**A three-image minimal reproduction passed on Mac with Docker on October 19,
2026.** The Docker environment is described in [Prepare the environment](#1-prepare-the-environment).

- The README checkpoint download command ran on the Mac host. All nine downloaded
  files matched the archived hashes. The CORD-v2 test Parquet was reused from a
  local cache after SHA256 verification; dataset extraction was rerun.
- Inside Docker, pinned dependency installation, PyTorch checkpoint loading,
  three-image wrapper self-checks, three-graph ONNX export and loading, attention
  fusion, graph simplification, and cross-attention K/V reuse all passed.
  The exported and optimized graph hashes matched the original experiment.
- The current standalone `donut_onnx_runner.py` ran the first three CORD-v2 test
  images (`cord_test_000.png` through `cord_test_002.png`). All **3/3 token
  sequences, decoded texts, and parsed JSON outputs** matched the corresponding
  archived PyTorch results. The maximum absolute token-probability difference
  was `1.1205673217773438e-05`; numerical results were not bitwise identical.
  The result-summary script also passed on these three images.

**Full-dataset performance data remains from the original 100-image experiment.**
The Mac check validates the minimal workflow and output agreement for three
images. It did not rerun the full paired performance benchmark or operator
profiling, and its timings do not confirm or replace the archived latency and
speedup numbers.

- The scripts target this specific CORD-v2 checkpoint and greedy decoding at batch size 1. Other checkpoints, tasks, batch sizes, and concurrent serving are unverified.
- No quantization or accuracy evaluation against CORD ground truth is included.
- Archived `script_sha256` and `runner_sha256` values identify the scripts used in the original experiment. They do not identify the current source files or the Mac reproduction run.
- Recorded model, data, and graph hashes identify the experimental assets. Weights, ONNX graphs, and raw profiler traces are not included; operator summaries are included.
- Archived hostnames and paths use placeholders such as `<pod>`, `<HOME>`, `<WORK_DIR>`, and `<VENV>`. They are metadata, not executable local paths. Shell files under `results/` are archived command records containing these placeholders; use the entry points under `scripts/` and the reproduction steps above to run the workflow. Descriptive metadata is in English; experiment numbers, recorded hashes, and model outputs are preserved. The archived summaries omit an unrelated reference field.

## Sources and data

- [Official Donut project](https://github.com/clovaai/donut) and [CORD-v2 checkpoint](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2).
- [CORD dataset](https://github.com/clovaai/cord) and [CORD-v2 on Hugging Face](https://huggingface.co/datasets/naver-clova-ix/cord-v2), attributed under CC BY 4.0. Archived result files contain model-recognized receipt text and parsed JSON derived from CORD.

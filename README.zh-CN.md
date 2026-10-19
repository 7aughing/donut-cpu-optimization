# Donut CPU 推理优化

[English](README.md) | 简体中文

本项目提供官方
[`naver-clova-ix/donut-base-finetuned-cord-v2`](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2)
模型的 CPU 推理优化脚本与可追溯实验记录。流程包括导出三张 ONNX 图、融合带缓存解码器的注意力算子，以及在解码步骤之间复用交叉注意力键值缓存（K/V）。本项目是使用官方模型的社区项目。

[实验结果](#实验结果) · [快速查看结果](#快速查看结果) · [复现实验](#复现实验) ·
[脚本导航](#脚本导航) · [适用范围与复现状态](#适用范围与复现状态) ·
[与 onnx-donut 的区别](#与-ikomia-devonnx-donut-的区别)

## 实验结果

最终 ONNX 版本的**单图耗时中位数为 5,063 ms**，PyTorch FP32 为 **7,153 ms**：按两个中位数之比计算，**耗时降低 29.2%，加速比为 1.41 倍**。全部 100 张图的生成词元序列、解码文本和解析 JSON 均与 PyTorch 基线一致。全量性能数据均来自原始归档实验。当前脚本已在 Mac 的 Docker 环境完成[3 张图片的最小复现](#适用范围与复现状态)；这次验证没有重跑 100 张图片的性能基准。

### 测试条件

实验于 2026 年 10 月 18 日（UTC）运行，使用 CORD-v2 test 全部 100 张图，批大小 1、贪心解码、单进程。Linux CPU 容器具有 4 核 CPU 配额和 8 GiB 内存上限；记录的软件版本为 Python 3.9.6、PyTorch 2.0.1+cpu、Transformers 4.25.1、ONNX Runtime 1.18.1。PyTorch 使用 2 个算子内线程；ONNX Runtime 使用 2 个算子内线程、1 个算子间线程。计时包含预处理、生成、词元概率计算与 JSON 解析，不包含读图和图片解码。

每个优化阶段与前一阶段配对测试，分 4 轮、每轮 25 张，轮换进程启动顺序。以下总览取自不同阶段，并非四个后端同时运行的一轮实验。

| 后端 | 单图耗时中位数 | P95 耗时 | 与 PyTorch 生成词元序列一致 | 来源 |
|---|---:|---:|---:|---|
| PyTorch FP32 | 7,153 ms | 10,327 ms | 基线 | [第 2 步](results/step2/donut_official_pair_raw_20261018T1012Z_c11161/summary_torch_fp32.json) |
| 原始 ONNX 三图 | 5,331 ms | 7,228 ms | 100 / 100 | [第 2 步](results/step2/donut_official_pair_raw_20261018T1012Z_c11161/summary_onnx_raw.json) |
| 融合 ONNX 三图 | 5,290 ms | 6,965 ms | 100 / 100 | [第 3 步](results/step3/donut_official_pair_mha2_20261018T1253Z_512992/summary_onnx_mha2.json) |
| 融合 ONNX 三图 + 交叉注意力 K/V 复用 | 5,063 ms | 6,382 ms | 100 / 100 | [第 4 步](results/step4/donut_official_pair_xkv_20261018T1339Z_494003/summary_onnx_mha2_xkv.json) |

### 各优化阶段的收益

实测最大的收益来自运行导出的 ONNX 图，算子融合和 K/V 复用进一步减少了解码器耗时。下表每行均使用同一阶段重新运行的基线，因此基线中位数可能与总览表不同。

| 配对对比 | 耗时中位数：基线 → 优化后 | 逐图配对耗时降幅 | 逐图配对解码器耗时降幅 | 更快的图片数 | 证据 |
|---|---:|---:|---:|---:|---|
| PyTorch FP32 → 原始 ONNX | 7,153 → 5,331 ms | 26.1% | 40.4% | 100 / 100 | [对比结果](results/step2/donut_official_pair_raw_20261018T1012Z_c11161/compare_onnx_raw_vs_torch_fp32.json) |
| 原始 ONNX → 融合 ONNX | 5,349 → 5,290 ms | 1.6% | 7.6% | 97 / 100 | [对比结果](results/step3/donut_official_pair_mha2_20261018T1253Z_512992/compare_onnx_mha2_vs_onnx_raw.json) |
| 融合 ONNX → K/V 复用 | 5,271 → 5,063 ms | 4.1% | 20.3% | 99 / 100 | [对比结果](results/step4/donut_official_pair_xkv_20261018T1339Z_494003/compare_onnx_mha2_xkv_vs_onnx_mha2.json) |

逐图配对降幅按同一批 100 张图计算：`100 × (1 − median(optimized_ms[i] / reference_ms[i]))`。这个统计量不同于开头 29.2% 使用的两个总体中位数之比；各阶段百分比也不能直接相加。

各阶段的图改动如下：

- **原始 ONNX：** 分别导出编码器、首步解码器、带缓存解码器三张图，由 Python 贪心解码循环调度。
- **算子融合：** 在带缓存解码器中融合 4 个交叉注意力 `MultiHeadAttention` 算子，以及归一化和激活模式，再简化计算图。见[融合记录](results/step3/donut_official_onnx_mha2_20261018T1250Z_26c1ed/fuse_manifest.json)。
- **K/V 复用：** 保留首步解码产生的交叉注意力键值缓存，带缓存解码器不再逐步重复输出这组张量。编码器、首步解码器和权重初始值保持不变。见 [K/V 记录](results/step4/donut_official_onnx_mha2_xkv_20261018T1337Z_1ad308/xkv_manifest.json)。

### 输出一致性与剩余瓶颈

[最终版本与 PyTorch 的对比](results/step4/donut_official_pair_xkv_20261018T1339Z_494003/compare_onnx_mha2_xkv_vs_step2_torch_fp32.json)显示：100/100 词元序列一致，文本变化数和解析 JSON 变化数均为 0。词元概率的最大绝对差为 `1.9073486328125e-05`；这表示输出一致，不代表全部数值逐位相同，也不是真值准确率。K/V 复用版本与融合 ONNX 在同一批 100 张图上的词元概率差为 0。

最终版本的 [100 张汇总](results/step4/donut_official_pair_xkv_20261018T1339Z_494003/summary_onnx_mha2_xkv.json)记录了编码器中位耗时 4,138 ms、解码器中位耗时 814 ms，编码器仍是最大的耗时部分；各段中位数不能直接相加得到总耗时中位数。另一个只使用前 3 张测试图的性能分析显示，`MatMul` 占编码器 `model_run` 时间的 74.3%，输出词表投影（`lm_head`）占带缓存解码器 `model_run` 时间的 32.7%。见[逐算子拆分](results/step5/donut_official_ort_profile_20261018T1543Z_f7db2a/onnx_mha2_xkv/window.json)。排除预热后，该分析完整覆盖了 3 次编码器、3 次首步解码器和 160 次带缓存解码器调用。这些小样本算子占比用于说明耗时去向，与 100 张图的延迟基准分开统计。

## 与 Ikomia-dev/onnx-donut 的区别

[Ikomia-dev/onnx-donut](https://github.com/Ikomia-dev/onnx-donut/blob/bbde423d4cc63d5d6a727f20c483adbe8467bc9c/README.md) 提供可安装的 Donut ONNX 导出与推理包。本仓库聚焦一个固定版本的 CORD-v2 checkpoint 的 CPU 优化与实测。以下对比依据 2026 年 10 月 19 日核对的 Ikomia 提交 [`bbde423`](https://github.com/Ikomia-dev/onnx-donut/commit/bbde423d4cc63d5d6a727f20c483adbe8467bc9c)。

| 方面 | Ikomia-dev/onnx-donut | 本仓库 |
|---|---|---|
| 使用方式 | `pip install onnx-donut`；提供 `export_onnx` 和 `OnnxPredictor.generate`；README 以 DocVQA 为例 | 面向固定 CORD-v2 checkpoint 的脚本，覆盖导出、优化和 CPU 配对测速 |
| 模型加载 | [导出器](https://github.com/Ikomia-dev/onnx-donut/blob/bbde423d4cc63d5d6a727f20c483adbe8467bc9c/src/onnx_donut/exporter.py)使用 `donut.DonutModel.from_pretrained` | [导出器](scripts/donut_onnx_export.py)使用 Transformers 的 `VisionEncoderDecoderModel` 和 `DonutProcessor`，校验权重哈希 |
| 计算图与缓存 | 采用三张图；[推理器](https://github.com/Ikomia-dev/onnx-donut/blob/bbde423d4cc63d5d6a727f20c483adbe8467bc9c/src/onnx_donut/predictor.py)在解码步骤间传递自注意力和交叉注意力缓存，每步读取缓存输出 | 同样采用三张图；最终版本移除带缓存解码器重复输出的交叉注意力 K/V，由[推理器](scripts/donut_onnx_runner.py)保留首步产生的值 |
| 优化与精度 | 可选[动态 INT8 量化](https://github.com/Ikomia-dev/onnx-donut/blob/bbde423d4cc63d5d6a727f20c483adbe8467bc9c/src/onnx_donut/quantizer.py)，排除 `Conv` | FP32 注意力融合、计算图化简和移除重复的交叉注意力 K/V 输出；未量化 |

三图导出和带缓存解码是双方共有的技术。本仓库记录的工作是针对固定 checkpoint 的分阶段 FP32 优化、100 张图片的配对测速、逐图输出核对和算子性能分析。文中的 **1.41 倍**加速比来自本实验最终 ONNX 版本与归档 PyTorch FP32 基线的比较。原实验和 Mac 验证均未对 Ikomia 的实现测速，因此这些结果不能说明本仓库比 `onnx-donut` 更快。

## 快速查看结果

使用 Python 3 与标准库即可对比已有结果，无需下载模型或安装推理依赖。把 `REPO` 设置为本目录的绝对路径：

```bash
export REPO=/absolute/path/to/donut-cpu-optimization
python3 "$REPO/scripts/donut_compare_runs.py" \
  "$REPO/results/step2/donut_official_pair_raw_20261018T1012Z_c11161/results_torch_fp32.jsonl" \
  "$REPO/results/step4/donut_official_pair_xkv_20261018T1339Z_494003/results_onnx_mha2_xkv.jsonl"
```
输出应包含 `common: 100`、`sequence_identical: 100`、`text_changed: 0` 和 `parsed_changed: 0`。这条命令核对已有输出，不运行模型。

## 目录结构

| 路径 | 内容 |
|---|---|
| [scripts/](scripts/) | 导出、优化、推理和计时脚本；注释与命令行提示为英文 |
| [results/](results/) | 逐图 JSONL、运行记录、汇总、对比与日志 |
| [requirements.txt](requirements.txt) | 根据已记录的 Linux 实验环境固定直接依赖版本 |

## 复现实验

### 1. 准备环境

原实验使用 Linux 与 Python 3.9.6。Shell 驱动依赖 `/proc`、cgroup v2 文件和 GNU `timeout`，应在 Linux 环境运行。固定依赖安装和 3 张图片的模型流程已在 Apple Silicon Mac 的 Docker Desktop 中验证，使用 `python:3.9.6-slim` 镜像，通过 Rosetta 运行 `linux/amd64`，容器 CPU 配额为 4、内存上限为 7 GiB。在 Mac 上使用 Docker 时，下方命令也应在 Linux 容器内执行。已验证步骤见[适用范围与复现状态](#适用范围与复现状态)。

请使用**仓库外的新工作目录**，将生成的模型和运行结果与归档证据分开。下面的命令使用 Bash：

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
依赖文件将 SentencePiece 和 PyArrow 装入同一个虚拟环境。脚本也支持通过 `--extra-path` 指定单独安装的依赖目录，原实验采用了这种方式。

### 2. 准备模型与数据

下载以下固定版本的上游文件，按下方目录存放：

- [Donut CORD-v2 模型，版本 `8003d433`](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2/tree/8003d433113256b4ce3a0f5bf604b29ff78a7451)：将权重、配置、预处理器和分词器文件放入 `$A/model/`。
- [CORD-v2 test Parquet，版本 `7f0115a4`](https://huggingface.co/datasets/naver-clova-ix/cord-v2/blob/7f0115a4b758a71d6473b8d085751692da2fef98/data/test-00000-of-00001-9c204eb3f4e11791.parquet)：将实际 Parquet 文件放入 `$A/cord_v2/`。

`pytorch_model.bin` 保存模型权重，需要与同一版本的配置、预处理器和分词器文件一起放在 `$A/model/`。`--model-dir` 接收这个目录路径，不能传入 `.bin` 文件本身。

完成第 1 步依赖安装后，在可联网环境执行以下命令，下载固定版本的完整 PyTorch checkpoint 所需文件，并校验权重哈希。`requirements.txt` 已包含 `huggingface_hub`，命令使用其 [`snapshot_download` 接口](https://huggingface.co/docs/huggingface_hub/v0.25.2/en/package_reference/file_download#huggingface_hub.snapshot_download)。

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

所需本地文件如下。下载还会包含上游模型说明，并可能在 `model/.cache/` 下创建下载元数据。执行提取命令前，通过上方数据集链接将 test Parquet 下载到 `$A/cord_v2/`。

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
预期 SHA256：

| 文件 | SHA256 |
|---|---|
| `pytorch_model.bin` | `31b78e3d3891072de8e2bf3553b71782242a1f3b589b914ec2b03feff7b14c54` |
| Test Parquet | `51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7` |

提取原始 PNG 字节并生成数据清单；输出目录必须尚不存在：

```bash
"$PY" -I "$A/scripts/cord_prepare.py" \
  --parquet "$A/cord_v2/test-00000-of-00001-9c204eb3f4e11791.parquet" \
  --out "$A/cord_v2/test"
```
### 3. 导出与优化

[导出驱动](scripts/donut_export_watch.sh)将 `--model-dir "$A/model"` 传给 [donut_onnx_export.py](scripts/donut_onnx_export.py)。导出脚本校验 `pytorch_model.bin` 的 SHA256，并通过 Transformers 加载模型目录。对应的加载调用为：

```python
processor = DonutProcessor.from_pretrained(str(model_dir))
model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).eval()
```

Transformers 读取配置和权重来构建模型，processor 加载图像预处理和分词器配置。导出脚本封装编码器与解码器模块，再调用 `torch.onnx.export`。原始 `.bin` 文件保留在模型目录，生成的 ONNX 文件写入单独的输出目录。导出驱动启用离线模式，因此启动前应完成模型和数据下载。

导出驱动为每一步创建独立进程。继续之前，检查每个 `EXIT` 日志的 `rc=0`、`killed=0`，并确认 `export_manifest.json` 已生成。内存守护在 cgroup 匿名内存超过 7 GiB 时终止子进程；子进程状态记录在最后的日志中，不能只依据 Shell 退出码判断成功。

```bash
export RAW="$WORK_DIR/onnx_raw"
export FUSED="$WORK_DIR/onnx_fused"
export XKV="$WORK_DIR/onnx_cross_kv_reuse"

for step in selfcheck encoder decoder decoder_with_past check; do
  bash "$A/scripts/donut_export_watch.sh" "$step" "$RAW"
done
```
成功运行后生成以下文件：

```text
$RAW/
├── export_manifest.json
└── bundle/
    ├── encoder.onnx
    ├── decoder.onnx
    └── decoder_with_past.onnx
```

`encoder.onnx` 将图像张量转换为编码器特征；`decoder.onnx` 产生首个词元的 logits 和初始缓存；`decoder_with_past.onnx` 使用缓存执行后续解码步骤。`export_manifest.json` 记录图文件哈希和检查结果。后续算子融合与 K/V 复用处理这些 ONNX 图。

执行注意力融合、图化简与交叉注意力 K/V 复用：

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
测速前检查各步内存守护日志、`fuse_manifest.json` 与 `xkv_manifest.json`。融合和 K/V 复用脚本会校验归档输入图的哈希。不同环境的新导出可能生成不同哈希；出现不一致时应先核查来源，不应直接移除断言。

### 4. 推理与测速

先运行 test 的前 3 张图片。预测、词元序列、解析 JSON 与耗时写入 `results_onnx_mha2_xkv.jsonl`：

```bash
"$PY" -I "$A/scripts/donut_cpu_probe.py" \
  --backend onnx_mha2_xkv \
  --model-dir "$A/model" --manifest "$A/cord_v2/test/manifest.json" \
  --start 0 --end 3 --out "$WORK_DIR/smoke_xkv" \
  --bundle-dir "$XKV/bundle" --export-manifest "$XKV/xkv_manifest.json" \
  --runner-dir "$A/scripts" --threads 2 --ort-intra 2 --ort-inter 1
```
完整配对测试应使用新的输出目录。计时脚本会跳过已有结果中的已完成图片，因此复用输出目录不等于从头重跑。

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
## 脚本导航

| 任务 | 入口 |
|---|---|
| 提取 CORD 图片与数据清单 | [cord_prepare.py](scripts/cord_prepare.py) |
| 导出、检查三张 ONNX 图 | [donut_onnx_export.py](scripts/donut_onnx_export.py) |
| 融合、化简带缓存解码器 | [donut_fuse_mha.py](scripts/donut_fuse_mha.py) |
| 移除重复输出的交叉注意力 K/V | [donut_make_xkv_bundle.py](scripts/donut_make_xkv_bundle.py) |
| 执行三图贪心解码循环 | [donut_onnx_runner.py](scripts/donut_onnx_runner.py)（可导入模块） |
| 逐图推理计时并记录预测 | [donut_cpu_probe.py](scripts/donut_cpu_probe.py) |
| 对比输出／汇总耗时 | [donut_compare_runs.py](scripts/donut_compare_runs.py)／[donut_summarize.py](scripts/donut_summarize.py) |
| 单独核对、测试编码器 | [donut_encoder_bench.py](scripts/donut_encoder_bench.py) |
| 采集 ORT 性能轨迹／按时间窗汇总算子 | [donut_ort_profile.py](scripts/donut_ort_profile.py)／[donut_prof_window.py](scripts/donut_prof_window.py) |

## 适用范围与复现状态

**2026 年 10 月 19 日，已在 Mac 的 Docker 环境完成 3 张图片的最小复现。** Docker 配置见[准备环境](#1-准备环境)。

- README 中的 checkpoint 下载命令在 Mac 宿主机执行，下载的 9 个文件哈希均与归档一致。CORD-v2 test Parquet 使用本地缓存并核对 SHA256，重新执行了数据提取。
- Docker 内的固定依赖安装、PyTorch checkpoint 加载、3 张图片的模型封装自检、三张 ONNX 图的导出与加载、注意力融合、图化简和交叉注意力 K/V 复用均通过。导出及优化后的图哈希与原实验一致。
- 当前独立推理器 `donut_onnx_runner.py` 运行了 CORD-v2 test 的前 3 张图片（`cord_test_000.png` 至 `cord_test_002.png`）。**3/3 的词元序列、解码文本和解析 JSON** 均与对应的归档 PyTorch 结果一致。词元概率的最大绝对差为 `1.1205673217773438e-05`，数值并非逐位相同；结果汇总脚本也已通过这 3 张图片的验证。

**全量性能数据仍来自原始 100 张图片的实验。** Mac 验证确认了最小流程可运行，以及 3 张图片的输出一致性，没有重跑全量配对性能基准或算子性能分析。Mac 验证耗时不用于确认或替换归档耗时与加速比。

- 脚本面向指定 CORD-v2 模型，使用批大小 1 的贪心解码。其他模型、任务、批大小与并发服务尚未验证。
- 未包含量化，也未对照 CORD 真值评测准确率。
- 归档记录中的 `script_sha256` 和 `runner_sha256` 对应原实验使用的脚本，不对应当前源码或 Mac 最小复现运行。
- 模型、数据、图的记录哈希用于识别实验资产。仓库不含权重、ONNX 图和 profiler 原始轨迹，包含算子汇总。
- 归档主机名与路径使用 `<pod>`、`<HOME>`、`<WORK_DIR>`、`<VENV>` 等占位符。这些是元数据，不是可执行的本机路径。`results/` 下的 Shell 文件是含有这些占位符的历史命令记录；运行流程应使用 `scripts/` 中的入口与前面的复现步骤。说明性元数据使用英文，实验数值、记录中的哈希与模型输出保持原样。归档汇总省略了一个与本实验无关的参照字段。

## 来源与数据

- [官方 Donut 项目](https://github.com/clovaai/donut)与 [CORD-v2 模型](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2)。
- [CORD 数据集](https://github.com/clovaai/cord)与 [Hugging Face CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)，按 CC BY 4.0 标明来源。归档结果包含模型对 CORD 收据的识别文本与解析 JSON。

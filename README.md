# yolo26.AXERA

Comparing YOLO26 inference at different quantization precision levels — **U8 / U16 / mixed precision on AX620E (AX630C)**.

本仓库完整复现了 YOLO26n 在 AX620E 平台的导出、量化、编译与板端推理对比，用于回答一个具体问题：

> U8 量化和 U16 量化所有算子都在 NPU，为什么混合量化的时候有些算子跑在了 CPU？

**结论：混合量化不会把算子放到 CPU。** 三个模型编译后都只有 1 个 NPU 子图、0 个 CPU 子图；板端 CPU 占用只由「每帧固定的 host 开销 ÷ 单帧耗时」决定，与精度分配无关。

## 结果速览

### 编译产物（Pulsar2 7.0 / AX620E / NPU2）

| 变体 | U16 层 | 输出 cosine (split0/1/2) | 子图 | MACs | 估算 cycles | axmodel |
|------|--------|--------------------------|------|------|-------------|---------|
| U8（全 U8） | 0 | 0.99914 / 0.99077 / 0.97568 | **1 × NPU** | 2.74 G | 6.00 M | 2.75 MB |
| U16（全 U16） | DEFAULT→U16 | 0.99996 / 0.99902 / 0.99743 | **1 × NPU** | 5.60 G | 13.57 M | 3.38 MB |
| 混合（`exp_e3_mix.json`） | 42 个节点 | 0.99937 / 0.99105 / 0.97860 | **1 × NPU** | 3.36 G | 8.34 M | 3.07 MB |

日志证据：`subgraph [0], group: 0, type: GraphType.NPU`、`fuse 1 subgraph(s)`，全日志无 `GraphType.CPU`（见 `logs/`、`results/build_summary.json`）。

### 板端性能与 CPU 占用（AX630C，2 核，每模型 3 轮 × 100 次）

| 变体 | Python avg | P50 | P90 | P99 | 进程 CPU | 系统 CPU | 吞吐 | 原生 `ax_run_model` avg |
|------|-----------|-----|-----|-----|----------|----------|------|------------------------|
| U8 | 11.174 ms | 11.139 | 11.444 | 11.676 | 43.2% | 22.5% | 89.4 fps | 6.349 ms |
| 混合 | 13.839 ms | 13.821 | 14.092 | 14.849 | 34.7% | 20.7% | 72.2 fps | 9.004 ms |
| U16 | 20.027 ms | 19.855 | 20.849 | 21.506 | 23.6% | 13.9% | 49.9 fps | 15.264 ms |

关键分析：host 侧固定开销 = Python 延迟 − 原生延迟 ≈ **4.8 ms**（4.83 / 4.84 / 4.76），
进程 CPU% ≈ 4.8 / 单帧耗时：

- U8：4.83 / 11.17 = 43.2%（实测 43.2%）
- 混合：4.84 / 13.84 = 35.0%（实测 34.7%）
- U16：4.76 / 20.03 = 23.8%（实测 23.6%）

即模型越快、每秒处理的帧数越多，CPU 占用**反而越高**；混合模型的 CPU 占用介于 U8 与 U16 之间是速度差异导致，与「算子跑 CPU」无关。
若客户端观察到明显更高的 CPU 占用（如 54.9% @ 47.6 ms ≈ 26 ms/帧），超出部分应来自 demo 自身的逐帧后处理（decode/NMS/画图/拷贝/日志），建议按同一口径排查。

## 目录结构

```
configs/     Pulsar2 配置：u8 / u16 / mix（客户 exp_e3_mix.json 原样）
scripts/     导出 / 拆图 / 校准集 / 配置生成 / 编译 / 汇总 脚本
demo/        bench_cpu.py：板端延迟 + CPU 占用（/proc 采集，无额外依赖）
onnx/        yolo26n.onnx（one2one 头切成 output_split_0/1/2）
models/      三个编译好的 .axmodel
results/     每轮 benchmark JSON、汇总 comparison.json、精度与编译摘要
logs/        Pulsar2 编译日志与板端运行日志
reports/     summary.md（完整分析报告）
```

## 环境要求

- x86_64 Linux + Docker（Pulsar2 7.0 镜像：`docker-registry.aitsw.axera-tech.com/pulsar2:7.0` 或官方 `pulsar2:7.0`）
- ultralytics 仓库（导出 ONNX 用，`ULTRALYTICS_REPO` 指定路径）
- AX620E 系列开发板（本仓库在 AX630C / ChipType.MC20E 上验证），Python ≥3.8 + `axengine` + `numpy`
- Pulsar2 7.0 driverless 授权：`/root/.hasplm/installed/32434/*.v2c`（`compile.py` 通过 `MAGNETAR_HASP_SRC` 挂载）

## 复现步骤

```bash
git clone git@github.com:ZY-2012/yolo26.AXERA.git && cd yolo26.AXERA
source scripts/env.sh                      # 所有临时文件写入 .work_tmp/，不使用 /tmp

# 1. 导出公开 yolo26n.pt → ONNX
python3 scripts/export_yolo26n.py

# 2. 按 end2end one2one 头切出 3 个尺度输出，对齐 exp_e3_mix.json 的 layer 名
python3 scripts/build_split_onnx.py

# 3. 校准集（coco128 取 100 张 → dataset/images100.tar）
python3 scripts/prepare_calibration.py

# 4. 生成 u8 / u16 / mix 三份配置
python3 scripts/make_configs.py

# 5. 编译（AX620E / NPU2）
export MAGNETAR_HASP_SRC=/path/to/.hasplm   # 内含 installed/32434/*.v2c
python3 scripts/compile.py u8.json u16.json mix.json

# 6. 板端对比
python3 scripts/prepare_input.py            # 生成 640x640 U8 NHWC 输入
export BOARD=root@<board-ip> BOARD_PASSWORD=<password>
bash scripts/board_bench.sh                 # 3 轮 × 100 次 + ax_run_model 交叉验证
python3 scripts/summarize.py
```

> 板端运行 `bench_cpu.py` 需要 `LD_LIBRARY_PATH=/opt/lib`（`axengine` 依赖 `libax_engine.so`），`board_bench.sh` 已处理。

## 实现说明

- **模型**：公开 ultralytics `yolo26n.pt`（end2end，80 类）。客户的 `exp_e3_mix.json` 中 42 个 layer 名全部命中 one2one 头/attention 节点。
- **ONNX 输出**：把 one2one 头按 P3/P4/P5 切成 `output_split_0/1/2`，形状 `1×84×80×80 / 1×84×40×40 / 1×84×20×20`（NCHW），与客户配置的 `output_processors(dst_perm=[0,2,3,1])` 对齐。
- **量化**：
  - `u8.json`：默认全 U8，无 `layer_configs`；
  - `u16.json`：`start/end_tensor_names=["DEFAULT"] → U16`；
  - `mix.json`：客户原始配置 + `target_hardware=AX620E`，42 个节点 U16（`/model.10` C2PSA、`/model.22` attention、`/model.19`、`/model.23/one2one_cv2/cv3` 各尺度）。
- **CPU 测量口径**（`demo/bench_cpu.py`）：系统级 `/proc/stat`、进程级 `/proc/self/stat`，均取基准窗口前后差值；延迟为 `session.run` 前后 `perf_counter`。
- **授权/临时文件**：Pulsar2 授权目录通过 `MAGNETAR_HASP_SRC` 挂载；仓库脚本所有 TMPDIR/缓存指向 `.work_tmp/`，不写 `/tmp`。

## 数据文件

- `results/comparison.json`：三模型板端汇总（含 native / host overhead）
- `results/bench_<model>_r<1..3>.json`：原始每轮数据
- `results/precision_summary.json`：量化精度分析（输出 cosine）
- `results/build_summary.json`：编译产物子图 / MACs / cycles
- `reports/summary.md`：完整中文分析报告
- `logs/{u8,u16,mix}.log`：Pulsar2 完整日志；`logs/board_bench.log`：板端运行记录

## 备注与偏差

- 复现使用公开 `yolo26n.pt` + coco128 校准集，与客户的实际 ONNX/校准集可能不同；量化精度数值仅供参考，CPU/子图结论与校准集无关。
- 本仓库只对量化精度分析（Pulsar2 Reference cosine）与运行耗时负责，未做端到端 mAP 验收。
- 参考实现：[Abandon-ht/YOLO26.axera](https://github.com/Abandon-ht/YOLO26.axera)（one2one 头 NPU 导出思路）；模型与导出框架：[ultralytics](https://github.com/ultralytics/ultralytics)。

## License

Apache-2.0

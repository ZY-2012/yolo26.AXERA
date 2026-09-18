# yolo26.AXERA

Comparing YOLO26 inference at different quantization precision levels — **U8 / U16 / mixed precision on AX620E (AX630C)**.

本仓库完整实现了 YOLO26n 在 AX620E 平台的导出、量化、编译与板端推理对比：

**结论：混合量化不会把算子放到 CPU。** 三个模型编译后都只有 1 个 NPU 子图、0 个 CPU 子图；板端 CPU 占用只由「每帧固定的 host 开销 ÷ 单帧耗时」决定，与精度分配无关。

## 结果速览

### 编译产物（Pulsar2 7.0 / AX620E / NPU2）

| 变体 | U16 层 | 输出 cosine (split0/1/2) | MACs | 估算 cycles | axmodel |
|------|--------|--------------------------|------|-------------|---------|
| U8（全 U8） | 0 | 0.99914 / 0.99077 / 0.97568 | 2.74 G | 6.00 M | 2.75 MB |
| U16（全 U16） | DEFAULT→U16 | 0.99996 / 0.99902 / 0.99743 | 5.60 G | 13.57 M | 3.38 MB |
| 混合（`exp_e3_mix.json`） | 42 个节点 | 0.99937 / 0.99105 / 0.97860 | 3.36 G | 8.34 M | 3.07 MB |

日志证据：`subgraph [0], group: 0, type: GraphType.NPU`、`fuse 1 subgraph(s)`，全日志无 `GraphType.CPU`（见 `logs/`、`results/build_summary.json`）。

### 板端性能与 CPU 占用（AX630C，2 核，每模型 3 轮 × 100 次）

| 变体 | Python avg | P50 | P90 | P99 | 进程 CPU | 系统 CPU | 吞吐 | 原生 `ax_run_model` avg |
|------|-----------|-----|-----|-----|----------|----------|------|------------------------|
| U8 | 11.171 ms | 11.101 | 11.414 | 12.364 | 42.3% | 26.1% | 89.5 fps | 6.360 ms |
| 混合 | 13.895 ms | 13.864 | 14.132 | 14.874 | 34.8% | 26.9% | 71.9 fps | 9.044 ms |
| U16 | 20.051 ms | 19.908 | 20.676 | 22.017 | 23.9% | 19.7% | 49.9 fps | 15.297 ms |

关键分析：host 侧固定开销 = Python 延迟 − 原生延迟 ≈ **4.8 ms**（4.81 / 4.85 / 4.75），
进程 CPU% ≈ 4.8 / 单帧耗时：

- U8：4.81 / 11.17 = 43.1%（实测 42.3%）
- 混合：4.85 / 13.90 = 34.9%（实测 34.8%）
- U16：4.75 / 20.05 = 23.7%（实测 23.9%）

即模型越快、每秒处理的帧数越多，CPU 占用**反而越高**；混合模型的 CPU 占用介于 U8 与 U16 之间是速度差异导致，与「算子跑 CPU」无关。
若客户端观察到明显更高的 CPU 占用（如 54.9% @ 47.6 ms ≈ 26 ms/帧），超出部分应来自 demo 自身的逐帧后处理（decode/NMS/画图/拷贝/日志），建议按同一口径排查。

## 目录结构

```
host/        主机执行的分步脚本（依次跑 01→08）
board/       板端执行的脚本：run_bench.sh + bench_cpu.py（无额外依赖）
scripts/     上述脚本调用的 Python 实现与公共环境（env.sh）
configs/     Pulsar2 配置：u8 / u16 / mix（客户 exp_e3_mix.json 原样）
onnx/        yolo26n.onnx（one2one 头切成 output_split_0/1/2）
models/      三个编译好的 .axmodel
results/     每轮 benchmark JSON、汇总 comparison.json、精度与编译摘要
logs/        Pulsar2 编译日志与板端运行日志
reports/     summary.md（完整分析报告）
```

## 环境要求

- x86_64 Linux + Docker（Pulsar2 7.0 镜像）
- ultralytics 仓库（导出 ONNX 用，`ULTRALYTICS_REPO` 指定路径）
- AX620E 系列开发板（本仓库在 AX630C / ChipType.MC20E 上验证），Python ≥3.8 + `axengine` + `numpy`

## 复现步骤（主机 / 板端分开执行）

### 主机执行（x86 编译机）

```bash
git clone git@github.com:ZY-2012/yolo26.AXERA.git && cd yolo26.AXERA

bash host/01_export_onnx.sh          # 下载 yolo26n.pt → ONNX
bash host/02_split_onnx.sh           # one2one 头切成 output_split_0/1/2
bash host/03_prepare_calibration.sh  # 生成 dataset/images100.tar（coco128 取 100 张）
bash host/04_make_configs.sh         # 生成 u8 / u16 / mix 三份配置
bash host/05_compile.sh              # Pulsar2 7.0 docker 编译（AX620E / NPU2，无需 export 授权）
bash host/06_prepare_input.sh        # 生成 artifacts/input_640.npy

# 板子地址/密码直接改 host/07_push_and_run.sh 顶部两行（BOARD_ADDR / BOARD_PASS）
bash host/07_push_and_run.sh         # 上传 → ssh 触发板端 run_bench.sh → 回拉 results/
bash host/08_summarize.sh            # 汇总 comparison.json + 打印对比表
```

- `01/02` 需要 ultralytics：`pip install ultralytics`，或 `export ULTRALYTICS_REPO=/path/to/ultralytics`。
- `05` 授权自动处理：优先 `<repo>/.p2_home/.hasplm/installed/32434/*.v2c`，没有就用 Pulsar2 7.0 镜像内置的 `/root/*.v2c`，**不需要设置任何环境变量**。
- 所有临时/缓存文件写入 `.work_tmp/`，不使用 `/tmp`。

### 板端执行（`host/07` 会自动触发；也可登录板子单独跑）

`host/07` 把 `board/`、3 个 axmodel、输入 tensor 上传到 `/root/yolo26_bench/`，板端脚本为：

```bash
cd /root/yolo26_bench
bash run_bench.sh                    # 可选：ROUNDS=3 REPEAT=100 WARMUP=10 bash run_bench.sh
```

> `run_bench.sh` 自动设置 `LD_LIBRARY_PATH=/opt/lib`（axengine 依赖 libax_engine.so），依次跑 U8 / 混合 / U16 的 3 轮 × 100 次，存在 `/opt/bin/ax_run_model` 时自动做原生交叉验证，结果写 `results/`。

## 实现说明

- **模型**：公开 ultralytics `yolo26n.pt`（end2end，80 类）。客户的 `exp_e3_mix.json` 中 42 个 layer 名全部命中 one2one 头/attention 节点。
- **ONNX 输出**：把 one2one 头按 P3/P4/P5 切成 `output_split_0/1/2`，形状 `1×84×80×80 / 1×84×40×40 / 1×84×20×20`（NCHW），与客户配置的 `output_processors(dst_perm=[0,2,3,1])` 对齐。
- **量化**：
  - `u8.json`：默认全 U8，无 `layer_configs`；
  - `u16.json`：`start/end_tensor_names=["DEFAULT"] → U16`；
  - `mix.json`：客户原始配置 + `target_hardware=AX620E`，42 个节点 U16（`/model.10` C2PSA、`/model.22` attention、`/model.19`、`/model.23/one2one_cv2/cv3` 各尺度）。
- **CPU 测量口径**（`board/bench_cpu.py`）：系统级 `/proc/stat`、进程级 `/proc/self/stat`，均取基准窗口前后差值；延迟为 `session.run` 前后 `perf_counter`。
- **授权/临时文件**：`scripts/compile.py` 自动查找 `<repo>/.p2_home/.hasplm`（可用 `MAGNETAR_HASP_SRC` 覆盖），找不到就用镜像内置 v2c；所有 TMPDIR/缓存指向 `.work_tmp/`，不写 `/tmp`。

## 数据文件

- `results/comparison.json`：三模型板端汇总（含 native / host overhead）
- `results/bench_<model>_r<1..3>.json`：原始每轮数据
- `results/precision_summary.json`：量化精度分析（输出 cosine）
- `results/build_summary.json`：编译产物子图 / MACs / cycles
- `reports/summary.md`：完整中文分析报告
- `logs/{u8,u16,mix}.log`：Pulsar2 完整日志；`results/ax_run_model.txt`：板端原生工具交叉验证记录

## 备注与偏差

- 复现使用公开 `yolo26n.pt` + coco128 校准集，与客户的实际 ONNX/校准集可能不同；量化精度数值仅供参考，CPU/子图结论与校准集无关。
- 本仓库只对量化精度分析（Pulsar2 Reference cosine）与运行耗时负责，未做端到端 mAP 验收。
- 参考实现：[Abandon-ht/YOLO26.axera](https://github.com/Abandon-ht/YOLO26.axera)（one2one 头 NPU 导出思路）；模型与导出框架：[ultralytics](https://github.com/ultralytics/ultralytics)。

## License

Apache-2.0

# yolo26.AXERA

YOLO26n 在 AX620E（AX630C）上的 **U8 / U16 / 混合量化** 推理对比复现：主机负责导出与量化，板端负责推理与 CPU 占用对比。

**结论：混合量化不会把算子放到 CPU。** 三个模型编译后都只有 1 个 NPU 子图、0 个 CPU 子图；板端 CPU 占用只由「每帧固定的 CPU（host）侧开销 ÷ 单帧耗时」决定，与量化精度分配无关。

## 目录结构

```
host/        主机执行：run_all.sh + 01~05 分步脚本
board/       板端执行：run_bench.sh + bench_cpu.py + summarize.py
scripts/     主机脚本调用的 Python 实现（导出 / 拆图 / 校准集 / 生成配置 / 公共环境）
configs/     Pulsar2 配置：u8.json / u16.json / mix.json（原配置 customer_exp_e3_mix.json）
onnx/        yolo26n.onnx（one2one 头切成 output_split_0/1/2）
models/      三个编译好的 axmodel
results/     每轮 benchmark JSON、comparison.json、精度与编译摘要
logs/        Pulsar2 编译日志
reports/     summary.md（完整分析报告）
```

## 数据文件

- `results/comparison.json`：三模型板端汇总（含 `ax_run_model` 原生延迟与 CPU 侧开销）
- `results/bench_<model>_r<1..3>.json`：原始每轮数据
- `results/ax_run_model.txt`：板端原生工具交叉验证
- `results/precision_summary.json`：量化精度分析（输出 cosine）
- `results/build_summary.json`：编译产物子图 / MACs / cycles
- `reports/summary.md`：完整中文分析报告
- `logs/{u8,u16,mix}.log`：Pulsar2 完整日志

## 分工

| 阶段 | 在哪执行 | 命令 | 产出 |
|------|----------|------|------|
| 导出 ONNX → 拆图 → 校准集 → 生成配置 → **量化编译** | 主机（x86 + Pulsar2） | `bash host/run_all.sh` | `models/yolo26n_{u8,u16,e3_mix}.axmodel` |
| **推理 / 延迟 / CPU 占用对比** | 板端（AX620Q/AX630C） | `bash board/run_bench.sh` | `board/results/*.json` + 终端对比表 |

## 主机执行（导出 + 量化）

**前置条件**

1. Python 环境里安装好 ultralytics（导出 ONNX 用）：`pip install ultralytics`
2. **Pulsar2 环境**：激活后确保 `pulsar2` 命令可用（`host/05_compile.sh` 会检查并提示）

**一键执行**

```bash
bash host/run_all.sh
```

脚本依次执行 5 步（无需任何参数）：

| 步骤 | 脚本 | 做什么 | 产物 |
|------|------|--------|------|
| 1 | `host/01_export_onnx.sh` | 下载公开 `yolo26n.pt`（end2end）并导出固定 640 ONNX | `onnx/yolo26n.onnx` |
| 2 | `host/02_split_onnx.sh` | 把 end2end 的 one2one 头按 P3/P4/P5 切成三个输出，对齐混合配置里的 layer 名与 `output_processors` | 就地覆盖 `onnx/yolo26n.onnx`（三输出） |
| 3 | `host/03_prepare_calibration.sh` | 生成 100 张真实图校准集（首次自动下 coco128） | `dataset/images100.tar` |
| 4 | `host/04_make_configs.sh` | 从 `configs/customer_exp_e3_mix.json` 生成三份配置 | `configs/{u8,u16,mix}.json` |
| 5 | `host/05_compile.sh` | `pulsar2 build` 量化编译（AX620E / NPU2） | `build/*/*.axmodel` 并复制到 `models/` |

三份量化配置的区别：

- `u8.json`：全部算子 U8（对照组）
- `u16.json`：全部算子 U16（`DEFAULT → U16`）
- `mix.json`： `exp_e3_mix.json` 原样，只把 42 个节点设为 U16（`/model.10` C2PSA、`/model.22` attention、`/model.19`、`/model.23/one2one_cv2/cv3` 各尺度），其余 U8

> 想用自己的业务图片做校准：直接替换 `dataset/images100.tar`（100 张图打包，文件名任意）再重新执行第 5 步即可。

## 板端执行（推理对比）

把主机生成的 `models/` 和仓库里的 `board/` 放到板子同一父目录，登入板子一键运行：

```bash
# 1. 主机上建目录并拷贝（板子 IP 换成实际地址；多源 scp 要求目标目录已存在）
ssh root@<板子IP> "mkdir -p /root/yolo26_bench"
scp -r models board root@<板子IP>:/root/yolo26_bench/

# 2. 登入板子
ssh root@<板子IP>
cd /root/yolo26_bench/board
bash run_bench.sh
```

`run_bench.sh` 自动完成（无参数、无环境变量）：

1. 空载 CPU 基线（5 s）
2. U8 / 混合 / U16 三个模型各跑 3 轮 × 100 次，输出 `Benchmark Results`（Repeat / Total / Avg / Min / Max / P50 / P90 / P99 / CPU usage / Throughput）
3. 有 `/opt/bin/ax_run_model` 时自动做纯 NPU 交叉验证
4. 最后打印汇总对比表，原始数据写入 `results/`

板端依赖：`numpy` + `axengine`；脚本自动设置 `LD_LIBRARY_PATH=/opt/lib`。测速默认使用与模型输入同 shape/dtype 的全 0 数据（与 Axera 官方 benchmark 口径一致，耗时与输入内容无关）；如需真实图片，可传 `python3 bench_cpu.py --model xxx.axmodel --input xxx.npy`。

## 结果（本次实测：AX630C，2 核，Pulsar2 7.0）

### 编译产物

| 变体 | U16 层 | 输出 cosine (split0/1/2) | MACs | 估算 cycles | axmodel |
|------|--------|--------------------------|------|-------------|---------|
| U8（全 U8） | 0 | 0.99914 / 0.99077 / 0.97568 | 2.74 G | 6.00 M | 2.75 MB |
| U16（全 U16） | DEFAULT→U16 | 0.99996 / 0.99902 / 0.99743 | 5.60 G | 13.57 M | 3.38 MB |
| 混合（`exp_e3_mix.json`） | 42 个节点 | 0.99937 / 0.99105 / 0.97860 | 3.36 G | 8.34 M | 3.07 MB |

- 输出 cosine = Pulsar2 量化精度分析（EndToEnd Reference）中，量化模型与浮点模型在三个输出张量上的余弦相似度，1.0 为完全一致；它反映量化扰动，不是 mAP。
- 三个模型的编译日志均只有 `subgraph [0] ... type: GraphType.NPU`、`fuse 1 subgraph(s)`，**没有 `GraphType.CPU`**（见 `logs/`）。

### 板端性能与 CPU 占用（每模型 3 轮 × 100 次）

| 变体 | avg | min | P50 | P90 | P99 | 进程 CPU | 系统 CPU | 吞吐 | `ax_run_model` avg |
|------|-----|-----|-----|-----|-----|----------|----------|------|--------------------|
| U8 | 11.157 ms | 10.768 | 11.083 | 11.476 | 12.612 | 42.7% | 24.0% | 89.6 fps | 6.348 ms |
| 混合 | 13.810 ms | 13.480 | 13.803 | 14.022 | 14.414 | 34.7% | 21.3% | 72.4 fps | 8.996 ms |
| U16 | 19.978 ms | 19.376 | 19.832 | 20.783 | 21.432 | 23.7% | 8.6% | 50.0 fps | 15.279 ms |

**关键分析**：CPU（host）侧固定开销 = Python 延迟 − `ax_run_model` 原生延迟 ≈ **4.8 ms**（4.81 / 4.81 / 4.70），进程 CPU% ≈ 4.8 / 单帧耗时。

> 术语说明：这里的 host 指**板子的 CPU**（相对 NPU 而言），不是 x86 主机。axmodel 在 NPU 上执行，每帧的输入准备、内存搬运、Python/axengine 框架调用、输出拷贝都跑在这颗 CPU 上；`ax_run_model` 是官方 C 工具、同样在板端 CPU 上，但只有极少的框架开销，两者差值就是 Python demo 的固定 CPU 开销。

- U8：4.81 / 11.16 = 43.1%（实测 42.7%）
- 混合：4.81 / 13.81 = 34.9%（实测 34.7%）
- U16：4.70 / 19.98 = 23.5%（实测 23.7%）

即模型越快、每秒处理的帧数越多，CPU 占用**反而越高**；混合模型的 CPU 占用介于 U8 与 U16 之间是速度差异导致，与「算子跑 CPU」无关。

## 参考

- [Abandon-ht/YOLO26.axera](https://github.com/Abandon-ht/YOLO26.axera)：one2one 头 NPU 导出
- [ultralytics](https://github.com/ultralytics/ultralytics)：模型与导出框架

## License

Apache-2.0

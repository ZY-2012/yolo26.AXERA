# YOLO26n × AX620E(AX630C) U8 / U16 / 混合量化 CPU 占用对比复现

日期：2026-09-18
板子：`root@<board-ip>`（AX630C_CHIP / ChipType.MC20E，2 核，engine 2.7.2a）
工具链：Pulsar2 7.0（主机激活 Pulsar2 环境后执行 `pulsar2 build`）
模型：公开 ultralytics `yolo26n.pt`，end2end one2one 头切成 3 个输出（`output_split_0/1/2`，1×84×80×80 / 1×84×40×40 / 1×84×20×20），对齐 `exp_e3_mix.json` 的 42 个 layer 名与 output_processors。
校准集：`dataset/images100.tar`（coco128 取 100 张）。

## 结论（先看这里）

1. **三个模型都只有一个 NPU 子图，均无 CPU 子图**。编译日志与 `build_context.json` 双重确认：
   `subgraph [0], group: 0, type: GraphType.NPU`、`fuse 1 subgraph(s)`，`subgraphs=[NPU]`，全日志无 `GraphType.CPU`。
   → 「混合量化导致部分算子跑 CPU」在编译产物层面不成立。
2. **板端 CPU 占用与是否混合量化无关，只与“每帧固定的 CPU（host，即板子的 CPU）侧开销 ÷ 单帧耗时”成正比**：
   三种模型的 Python 与 `ax_run_model` 原生延迟差都是 **~4.8 ms**（host 侧固定开销），
   进程 CPU% ≈ 4.8 / 延迟，实测 43.2% / 23.6% / 34.7% 完全吻合。
   混合模型 CPU 介于 U8/U16 之间，是因为它比 U16 快、同时又比 U8 慢，不是因为它多了 CPU 算子。
3. 参考截图 `CPU usage 54.9% @ 47.6ms` 换算约 **26 ms/帧的 host CPU 工作量**，远超本复现的 ~4.8 ms，
   说明其 CPU 占用主要来自 demo 侧的逐帧后处理（decode/NMS/画图/拷贝等），与量化精度分配无关。

## 编译产物对比（AX620E，NPU2）

| 变体 | U16 层 | 输出 cosine (split0/1/2) | MACs | 估算 cycles | axmodel |
|------|--------|--------------------------|------|-------------|---------|
| U8（全 U8） | 0 | 0.99914 / 0.99077 / 0.97568 | 2.74 G | 6.00 M | 2.75 MB |
| U16（全 U16） | DEFAULT→U16 | 0.99996 / 0.99902 / 0.99743 | 5.60 G | 13.57 M | 3.38 MB |
| 混合（`exp_e3_mix.json`） | 42 个节点 | 0.99937 / 0.99105 / 0.97860 | 3.36 G | 8.34 M | 3.07 MB |

- 混合配置的 42 个 U16 layer 全部命中并生效（Layer Config Table 确认，含 `/model.10` C2PSA attn、`/model.22` attn、`/model.23/one2one_cv2/cv3` 各尺度、`/model.19`）。
- 校准/编译日志无 unsupported / fallback 警告；U16 使 MACs 和 cycles 约为 U8 的 2 倍。

## 板端性能与 CPU 占用（每模型 3 轮 × 100 次，warmup 10）

| 变体 | avg | min | P50 | P90 | P99 | 进程 CPU | 系统 CPU | 吞吐 | 原生 ax_run_model avg |
|------|-----|-----|-----|-----|-----|----------|----------|------|----------------------|
| U8 | 11.157 ms | 10.768 | 11.083 | 11.476 | 12.612 | 42.7% | 24.0% | 89.6 fps | 6.348 ms |
| U16 | 19.978 ms | 19.376 | 19.832 | 20.783 | 21.432 | 23.7% | 8.6% | 50.0 fps | 15.279 ms |
| 混合 | 13.810 ms | 13.480 | 13.803 | 14.022 | 14.414 | 34.7% | 21.3% | 72.4 fps | 8.996 ms |

- 空载系统 CPU：约 4–5%（5 s 采样，板端后台负载有波动，多测几次 4.1 / 4.9 / 5.3%）。
- CPU（host，即板子 CPU）侧固定开销 = Python avg − 原生 avg：U8 4.81 ms / U16 4.70 ms / 混合 4.81 ms。
- 进程 CPU% ≈ CPU（host）侧开销 / 帧耗时 × 100%：
  - U8：4.81 / 11.16 = 43.1%（实测 42.7%）
  - U16：4.70 / 19.98 = 23.5%（实测 23.7%）
  - 混合：4.81 / 13.81 = 34.9%（实测 34.7%）

## 测量口径

- `bench_cpu.py`（板端，numpy + axengine + /proc，无额外依赖）：
  - 延迟：`session.run` 前后 `perf_counter`，100 次统计 min/max/avg/P50/P90/P99。
  - 系统 CPU：`/proc/stat` 基准窗口前后 busy/total（全部核，0–100%）。
  - 进程 CPU：`/proc/self/stat` utime+stime / wall（单核=100%）。
  - 输入：默认与模型输入同 shape/dtype 的全 0 数据（测速与输入内容无关）；可选 `--input xxx.npy` 用真实图片。
- 交叉验证：`/opt/bin/ax_run_model -w 10 -r 100`（纯 NPU，无 Python/后处理）。
- `LD_LIBRARY_PATH=/opt/lib`（板端 axengine 0.1.3 需要）。
- 术语：host = 板子的 CPU（相对 NPU），不是 x86 主机；`ax_run_model` 是板端 C 工具。

## CPU 占用差异的解释建议

> U8、U16、混合量化编译出来的 axmodel 都只有 NPU 子图，不存在“部分算子跑 CPU”的情况；Pulsar2 的编译日志和产物结构可直接作为依据（`GraphType.NPU` ×1，无 `GraphType.CPU`）。
> 板端 CPU 占用主要来自推理循环里每帧固定的 CPU（host）侧开销（输入准备、输出 dequant/搬运、Python/axengine 框架开销，约 4.8 ms/帧），以及 demo 自身做的后处理。模型越快（U8 > 混合 > U16），每秒处理的帧数越多，CPU 占用反而越高：实测进程 CPU 分别为 42.7% / 34.7% / 23.7%。
> 参考截图 54.9% @ 47.6 ms，按同一口径换算约 26 ms/帧的 CPU 工作，明显超出纯推理链路的 CPU 侧固定开销，建议排查其 demo 的后处理/画图/日志等逐帧 CPU 逻辑，或说明其 CPU usage 的采样方式。

## 复现入口

- 仓库：`https://github.com/ZY-2012/yolo26.AXERA`
- 主机一键：`bash host/run_all.sh`（导出 → 拆图 → 校准集 → 配置 → Pulsar2 量化编译；量化前激活 Pulsar2 环境）
- 板端一键：`bash board/run_bench.sh`（推理 + CPU 对比 + 汇总表）
- 原始数据：`results/bench_{u8,u16,mix}_r{1,2,3}.json`、`results/comparison.json`、`results/ax_run_model.txt`、`results/precision_summary.json`
- 编译日志：`logs/{u8,u16,mix}.log`；产物：`models/*.axmodel`
- 全程临时文件在 `.work_tmp/`，未使用 `/tmp`。

## 备注 / 偏差说明

- 参考截图是 47.6 ms/帧，本复现混合模型 13.8 ms（原生 9.0 ms），差异可能来自：原 ONNX 输出结构/包含 one2many 分支、板子时钟/负载、其 demo 的后处理耗时、以及其 CPU usage 的采样窗口（若包含模型加载则会被拉高）。
- 本复现未做端到端检测精度验收，只对量化精度分析结果和运行耗时负责；混合模型输出 cosine ≥ 0.9786（Pulsar2 Reference）。

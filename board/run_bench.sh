#!/usr/bin/env bash
# 板端一键执行：对比 U8 / 混合 / U16 三个 axmodel 的延迟与 CPU 占用。
#
# 用法（无需任何配置）：
#   1) 把 models/ 目录和 board/ 目录一起拷到板子（同一父目录下，例如 /root/yolo26_bench/）
#   2) 登入板子：cd /root/yolo26_bench/board && bash run_bench.sh
#   也可选：ROUNDS=5 REPEAT=200 bash run_bench.sh（默认 3 轮 × 100 次）
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

ROUNDS="${ROUNDS:-3}"
REPEAT="${REPEAT:-100}"
WARMUP="${WARMUP:-10}"

# 板端 axengine 需要 libax_engine.so
export LD_LIBRARY_PATH="/opt/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

model_file() {
    case "$1" in
        u8)  echo "yolo26n_u8.axmodel" ;;
        mix) echo "yolo26n_e3_mix.axmodel" ;;
        u16) echo "yolo26n_u16.axmodel" ;;
    esac
}

find_model() {
    local f
    for f in "models/$(model_file "$1")" "$(model_file "$1")" "../models/$(model_file "$1")"; do
        if [ -f "$f" ]; then echo "$f"; return 0; fi
    done
    echo "找不到 $(model_file "$1")：请把 models/ 和 board/ 放在同一父目录下" >&2
    return 1
}

mkdir -p results
rm -f results/bench_*.json results/idle.json results/ax_run_model.txt

echo "==> 空载 CPU 基线"
python3 bench_cpu.py --model "$(find_model u8)" --repeat 1 --warmup 0 \
    --idle-seconds 5 --json results/idle.json

for round in $(seq 1 "$ROUNDS"); do
    for name in u8 mix u16; do
        echo "==> 第 $round/$ROUNDS 轮：$name"
        python3 bench_cpu.py --model "$(find_model "$name")" \
            --warmup "$WARMUP" --repeat "$REPEAT" \
            --json "results/bench_${name}_r${round}.json"
    done
done

if [ -x /opt/bin/ax_run_model ]; then
    echo "==> ax_run_model 原生交叉验证" | tee results/ax_run_model.txt
    for name in u8 mix u16; do
        {
            echo "--- $name: $(model_file "$name")"
            /opt/bin/ax_run_model -m "$(find_model "$name")" -w "$WARMUP" -r "$REPEAT" 2>&1 | tail -15
        } | tee -a results/ax_run_model.txt
    done
else
    echo "未找到 /opt/bin/ax_run_model，跳过原生交叉验证" | tee results/ax_run_model.txt
fi

echo
echo "================ 汇总 ================"
python3 summarize.py

echo
echo "全部完成，原始数据在 $(pwd)/results/"

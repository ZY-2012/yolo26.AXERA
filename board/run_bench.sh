#!/usr/bin/env bash
# 板端执行：跑 U8 / 混合 / U16 三个模型的延迟与 CPU 占用对比。
# 用法（在板子上，文件已由 host/07_push_and_run.sh 上传到 /root/yolo26_bench）：
#   cd /root/yolo26_bench && bash run_bench.sh
# 可选：ROUNDS=3 REPEAT=100 WARMUP=10 bash run_bench.sh
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

mkdir -p results
rm -f results/idle.json results/ax_run_model.txt

echo "==> idle baseline"
python3 bench_cpu.py --model "$(model_file u8)" --input input_640.npy \
    --repeat 1 --warmup 0 --idle-seconds 5 --json results/idle.json

for round in $(seq 1 "$ROUNDS"); do
    for name in u8 mix u16; do
        echo "==> round $round / $name"
        python3 bench_cpu.py --model "$(model_file "$name")" --input input_640.npy \
            --warmup "$WARMUP" --repeat "$REPEAT" \
            --json "results/bench_${name}_r${round}.json"
    done
done

if [ -x /opt/bin/ax_run_model ]; then
    echo "==> ax_run_model cross-check" | tee results/ax_run_model.txt
    for name in u8 mix u16; do
        {
            echo "--- $name: $(model_file "$name")"
            /opt/bin/ax_run_model -m "$(model_file "$name")" -w "$WARMUP" -r "$REPEAT" 2>&1 | tail -15
        } | tee -a results/ax_run_model.txt
    done
else
    echo "ax_run_model not found, skipped" | tee results/ax_run_model.txt
fi

echo "==> done: $DIR/results"

#!/usr/bin/env bash
# 主机执行 7/8：上传模型/输入/demo 到板子，触发板端 run_bench.sh，回拉 results/。
# 板端脚本是 board/run_bench.sh（板子上单独执行也可以）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ===== 板子信息：按需改这两行（也可用同名环境变量覆盖，无需 export） =====
BOARD_ADDR="${BOARD_ADDR:-root@10.126.29.186}"
BOARD_PASS="${BOARD_PASS:-123456}"
REMOTE_DIR="${REMOTE_DIR:-/root/yolo26_bench}"
ROUNDS="${ROUNDS:-3}"
REPEAT="${REPEAT:-100}"
WARMUP="${WARMUP:-10}"

command -v sshpass >/dev/null || { echo "sshpass not found: apt install sshpass"; exit 1; }

SSH=(sshpass -p "$BOARD_PASS" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "$BOARD_ADDR")
SCP=(sshpass -p "$BOARD_PASS" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

model_path() {
    local name="$1" f
    case "$name" in
        u8)  f="$ROOT/models/yolo26n_u8.axmodel";     [ -f "$f" ] || f="$ROOT/build/u8/yolo26n_u8.axmodel" ;;
        mix) f="$ROOT/models/yolo26n_e3_mix.axmodel"; [ -f "$f" ] || f="$ROOT/build/mix/yolo26n_e3_mix.axmodel" ;;
        u16) f="$ROOT/models/yolo26n_u16.axmodel";    [ -f "$f" ] || f="$ROOT/build/u16/yolo26n_u16.axmodel" ;;
    esac
    [ -f "$f" ] || { echo "missing axmodel for $name: $f" >&2; exit 1; }
    echo "$f"
}

INPUT="${ROOT}/artifacts/input_640.npy"
[ -f "$INPUT" ] || INPUT="${ROOT}/board/input_640.npy"
[ -f "$INPUT" ] || { echo "missing input tensor, run host/06_prepare_input.sh first"; exit 1; }

echo "==> prepare $REMOTE_DIR on $BOARD_ADDR"
"${SSH[@]}" "mkdir -p $REMOTE_DIR/results && rm -f $REMOTE_DIR/results/*"

echo "==> upload"
"${SCP[@]}" "$INPUT" "$BOARD_ADDR:$REMOTE_DIR/input_640.npy"
"${SCP[@]}" "$ROOT/board/bench_cpu.py" "$ROOT/board/run_bench.sh" "$BOARD_ADDR:$REMOTE_DIR/"
for name in u8 mix u16; do
    path="$(model_path "$name")" || exit 1
    "${SCP[@]}" "$path" "$BOARD_ADDR:$REMOTE_DIR/"
done

echo "==> run board/run_bench.sh on board"
"${SSH[@]}" "cd $REMOTE_DIR && ROUNDS=$ROUNDS REPEAT=$REPEAT WARMUP=$WARMUP bash run_bench.sh"

echo "==> pull results"
mkdir -p "$ROOT/results"
"${SCP[@]}" "$BOARD_ADDR:$REMOTE_DIR/results/*" "$ROOT/results/"
echo "==> done: $ROOT/results"

#!/usr/bin/env bash
# Push models + demo to the AX630C board and run the CPU benchmark rounds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOARD="${BOARD:?export BOARD=user@host before running}"
PASS="${BOARD_PASSWORD:?export BOARD_PASSWORD=... before running}"
REMOTE_DIR="${REMOTE_DIR:-/root/yolo26_bench}"
ROUNDS="${ROUNDS:-3}"
REPEAT="${REPEAT:-100}"
WARMUP="${WARMUP:-10}"

SSH=(sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "$BOARD")
SCP=(sshpass -p "$PASS" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

MODELS=("u8:$ROOT/build/u8/yolo26n_u8.axmodel" "u16:$ROOT/build/u16/yolo26n_u16.axmodel" "mix:$ROOT/build/mix/yolo26n_e3_mix.axmodel")

echo "==> preparing $REMOTE_DIR on $BOARD"
"${SSH[@]}" "mkdir -p $REMOTE_DIR/results && rm -f $REMOTE_DIR/results/bench_*.json"

echo "==> copying files"
"${SCP[@]}" "$ROOT/board/input_640.npy" "$BOARD:$REMOTE_DIR/input_640.npy"
"${SCP[@]}" "$ROOT/demo/bench_cpu.py" "$BOARD:$REMOTE_DIR/bench_cpu.py"
for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"; path="${entry#*:}"
  "${SCP[@]}" "$path" "$BOARD:$REMOTE_DIR/yolo26n_${name}.axmodel"
done

echo "==> idle baseline"
"${SSH[@]}" "cd $REMOTE_DIR && LD_LIBRARY_PATH=/opt/lib python3 bench_cpu.py --model yolo26n_u8.axmodel --input input_640.npy --repeat 1 --warmup 0 --idle-seconds 5 --json results/idle.json" || true

for round in $(seq 1 "$ROUNDS"); do
  for entry in "${MODELS[@]}"; do
    name="${entry%%:*}"
    echo "==> round $round model $name"
    "${SSH[@]}" "cd $REMOTE_DIR && LD_LIBRARY_PATH=/opt/lib python3 bench_cpu.py --model yolo26n_${name}.axmodel --input input_640.npy --warmup $WARMUP --repeat $REPEAT --json results/bench_${name}_r${round}.json"
    "${SCP[@]}" "$BOARD:$REMOTE_DIR/results/bench_${name}_r${round}.json" "$ROOT/results/" || true
  done
done

echo "==> ax_run_model cross-check"
for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  echo "--- ax_run_model $name"
  "${SSH[@]}" "LD_LIBRARY_PATH=/opt/lib /opt/bin/ax_run_model -m $REMOTE_DIR/yolo26n_${name}.axmodel -w $WARMUP -r $REPEAT 2>&1 | tail -15" || true
done

"${SCP[@]}" "$BOARD:$REMOTE_DIR/results/idle.json" "$ROOT/results/" || true
echo "==> done, results in $ROOT/results"

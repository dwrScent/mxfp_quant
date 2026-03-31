#!/bin/bash

TASKS=${1:-"wikitext,hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq"}
WBIT=4
WMODE="nvesm2"
ABIT=4
AMODE="nvesm2"
GROUP_SIZE=16

# MODEL=meta-llama/Meta-Llama-3-8B
MODEL=/cephfs/shared/model/llama-3-8b-hf
# MODEL=/cephfs/shared/model/llama-2-7b-hf
# MODEL=/cephfs/shared/model/DeepSeek-R1-Distill-Qwen-7B
# MODEL=/cephfs/shared/model/Qwen-7B

TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-0}
LOG_DIR=${LOG_DIR:-"./logs"}

mkdir -p "$LOG_DIR"

RUN_ID=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$LOG_DIR/summary_${RUN_ID}.txt"

IFS=',' read -r -a TASK_LIST <<< "$TASKS"

cleanup_gpu() {
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()" >/dev/null 2>&1 || true
    sleep 2
}

parse_result() {
    local task="$1"
    local log_file="$2"

    python - "$task" "$log_file" <<'PY'
import re
import sys
from pathlib import Path

task = sys.argv[1]
log_file = Path(sys.argv[2])
text = log_file.read_text(encoding="utf-8", errors="ignore")

if task in {"wikitext", "c4", "ptb"}:
    matches = re.findall(r'(?m)^([0-9]+(?:\.[0-9]+)?)\s*$', text)
    if matches:
        print(f"ppl={matches[-1]}")
    else:
        print("ppl=N/A")
    raise SystemExit(0)

preferred_metric = {
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "boolq": "acc",
}

metric = preferred_metric.get(task)
lines = text.splitlines()

def parse_table_line(line):
    parts = [p.strip() for p in line.split("|")[1:-1]]
    if len(parts) < 7:
        return None
    task_name, _, _, _, metric_name, value, _ = parts[:7]
    return task_name, metric_name, value

if metric is not None:
    for line in lines:
        if f"| {task} " in line and f"| {metric} " in line:
            parsed = parse_table_line(line)
            if parsed:
                _, metric_name, value = parsed
                print(f"{metric_name}={value}")
                raise SystemExit(0)

for line in lines:
    if f"| {task} " in line:
        parsed = parse_table_line(line)
        if parsed:
            _, metric_name, value = parsed
            print(f"{metric_name}={value}")
            raise SystemExit(0)

print("metric=N/A")
PY
}

run_one_task() {
    local task="$1"
    local log_file="$LOG_DIR/${RUN_ID}_${task}.log"
    local status="OK"
    local result="metric=N/A"
    local cmd=(
        python -m mxq.entry
        --model_path "$MODEL"
        --tasks "$task"
        --w_bit "$WBIT"
        --w_mode "$WMODE"
        --a_bit "$ABIT"
        --a_mode "$AMODE"
        --group_size "$GROUP_SIZE"
    )

    echo
    echo "========== Running task: $task =========="
    echo "log: $log_file"

    if [ "$TIMEOUT_SECONDS" -gt 0 ]; then
        timeout "$TIMEOUT_SECONDS" "${cmd[@]}" >"$log_file" 2>&1
    else
        "${cmd[@]}" >"$log_file" 2>&1
    fi
    local exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        result=$(parse_result "$task" "$log_file")
    else
        if grep -qi "out of memory\|cuda error\|killed\|interrupt\|KeyboardInterrupt\|RuntimeError" "$log_file"; then
            status="SKIP"
        else
            status="FAIL"
        fi
        result="see $(basename "$log_file")"
    fi

    cleanup_gpu

    printf "%-16s %-8s %s\n" "$task" "$status" "$result" | tee -a "$SUMMARY_FILE"
}

echo "Model: $MODEL" | tee "$SUMMARY_FILE"
echo "Tasks: $TASKS" | tee -a "$SUMMARY_FILE"
echo "Log dir: $LOG_DIR" | tee -a "$SUMMARY_FILE"
echo | tee -a "$SUMMARY_FILE"
printf "%-16s %-8s %s\n" "Task" "Status" "Result" | tee -a "$SUMMARY_FILE"
printf "%-16s %-8s %s\n" "----" "------" "------" | tee -a "$SUMMARY_FILE"

for task in "${TASK_LIST[@]}"; do
    run_one_task "$task"
done

echo
echo "========== Summary =========="
cat "$SUMMARY_FILE"

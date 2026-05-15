#!/bin/bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

# Edit all experiment configuration here.
METHODS=("nvfp" "nves" "nvint4" "nvesm2_hw" "nvintesm2")
# MODEL_NAMES=("llama2-7b" "qwen-7b" "mistral-7b" "opt-6.7b" "falcon-7b" "llama3-8b")
# MODEL_PATHS=(
#     "/cephfs/shared/model/llama-2-7b-hf"
#     "/cephfs/shared/model/Qwen-7B"
#     "/cephfs/shared/model/mistral-7b"
#     "/cephfs/shared/model/opt-6.7b"
#     "/cephfs/shared/model/falcon-7b"
#     "/cephfs/shared/model/llama-3-8b-hf"
# )
MODEL_NAMES=( "qwen-7b")
MODEL_PATHS=(
    "/cephfs/shared/model/Qwen-7B"
)
TASKS=("wikitext" "c4" "ptb" "hellaswag" "piqa" "winogrande" "arc_easy" "arc_challenge" "boolq")

WBIT=4
ABIT=4
GROUP_SIZE=16
BATCH_SIZE=32
BOOLQ_BATCH_SIZE=8
TIMEOUT_SECONDS=0
LOG_DIR="$SCRIPT_DIR/logs"
RESULT_FILE="$SCRIPT_DIR/result.md"
UPLOAD_URL="https://filebox.expectopatronum.cc/api/file?path="
UPLOAD_TOKEN="lxy666"

RUN_ID=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$LOG_DIR/summary_${RUN_ID}.txt"
RESULTS_TSV="$LOG_DIR/results_${RUN_ID}.tsv"

mkdir -p "$LOG_DIR"

cleanup_gpu() {
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()" >/dev/null 2>&1 || true
    sleep 2
}

sanitize_field() {
    local value="$1"
    value=${value//$'\t'/ }
    value=${value//$'\n'/ }
    printf "%s" "$value"
}

append_record() {
    local model_name="$1"
    local method="$2"
    local task="$3"
    local status="$4"
    local batch_size="$5"
    local log_file="$6"
    local metric="$7"
    local value="$8"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(sanitize_field "$model_name")" \
        "$(sanitize_field "$method")" \
        "$(sanitize_field "$task")" \
        "$(sanitize_field "$status")" \
        "$(sanitize_field "$batch_size")" \
        "$(sanitize_field "$(basename "$log_file")")" \
        "$(sanitize_field "$metric")" \
        "$(sanitize_field "$value")" >>"$RESULTS_TSV"
}

init_result_file() {
    printf "model\tmethod\ttask\tstatus\tbatch\tlog\tmetric\tvalue\n" >"$RESULTS_TSV"
    render_result_file
}

parse_metrics() {
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
    print(f"ppl\t{matches[-1] if matches else 'N/A'}")
    raise SystemExit(0)

lines = text.splitlines()
current_task = None
rows = []

def parse_table_line(line):
    parts = [p.strip() for p in line.split("|")[1:-1]]
    if len(parts) < 9:
        return None
    task_name = parts[0] or None
    metric_name = parts[4]
    value = parts[6]
    return task_name, metric_name, value

for line in lines:
    if "|" not in line or "Metric" in line or "---" in line:
        continue
    parsed = parse_table_line(line)
    if not parsed:
        continue
    task_name, metric_name, value = parsed
    if task_name:
        current_task = task_name
    if current_task == task and metric_name in {"acc", "acc_norm"}:
        rows.append((metric_name, value))

seen = set()
for metric_name, value in rows:
    if metric_name in seen:
        continue
    seen.add(metric_name)
    print(f"{metric_name}\t{value}")
PY
}

classify_failure() {
    local log_file="$1"

    if grep -qi "out of memory\|cuda error\|killed\|interrupt\|KeyboardInterrupt\|RuntimeError\|trust_remote_code=True\|empty range for randrange" "$log_file"; then
        echo "SKIP"
    else
        echo "FAIL"
    fi
}

render_result_file() {
    python - "$RESULTS_TSV" "$RESULT_FILE" "$RUN_ID" "$WBIT" "$ABIT" "$GROUP_SIZE" "$BATCH_SIZE" "$BOOLQ_BATCH_SIZE" "$LOG_DIR" "${METHODS[*]}" "${MODEL_NAMES[*]}" "${TASKS[*]}" <<'PY'
import csv
import sys
from datetime import datetime
from pathlib import Path

results_tsv = Path(sys.argv[1])
result_file = Path(sys.argv[2])
run_id = sys.argv[3]
wbit = sys.argv[4]
abit = sys.argv[5]
group_size = sys.argv[6]
batch_size = sys.argv[7]
boolq_batch_size = sys.argv[8]
log_dir = sys.argv[9]
methods = sys.argv[10].split()
models = sys.argv[11].split()
tasks = sys.argv[12].split()

metrics = {}
statuses = {}
logs = {}

if results_tsv.exists():
    with results_tsv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["model"], row["method"], row["task"])
            statuses[key] = row["status"]
            logs[key] = row["log"]
            metrics[(row["model"], row["method"], row["task"], row["metric"])] = row["value"]

def as_float(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.upper() == "N/A":
            return None
        return float(text)
    except ValueError:
        return None

def format_value(value, percent=False, decimals=3):
    number = as_float(value)
    if number is None:
        return "" if value in (None, "", "N/A") else str(value)
    if percent and abs(number) <= 1.0:
        number *= 100.0
    return f"{number:.{decimals}f}"

def metric_value(model, method, task, metric):
    value = metrics.get((model, method, task, metric))
    if value is not None:
        return value
    status = statuses.get((model, method, task))
    if status and status != "OK":
        return status
    return ""

def average(values, percent=False):
    nums = []
    for value in values:
        number = as_float(value)
        if number is None:
            continue
        if percent and abs(number) <= 1.0:
            number *= 100.0
        nums.append(number)
    if not nums:
        return ""
    return f"{sum(nums) / len(nums):.3f}"

def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)

def ppl_table(model):
    ppl_tasks = ["wikitext", "c4", "ptb"]
    rows = []
    for method in methods:
        raw_values = [metric_value(model, method, task, "ppl") for task in ppl_tasks]
        values = [format_value(value, decimals=3) for value in raw_values]
        rows.append([method, *values, average(raw_values)])
    return md_table(["", *ppl_tasks, "avg"], rows)

def acc_table(model):
    acc_tasks = ["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "boolq"]
    rows = []
    for method in methods:
        raw_values = [metric_value(model, method, task, "acc") for task in acc_tasks]
        values = []
        for task, value in zip(acc_tasks, raw_values):
            cell = format_value(value, percent=True, decimals=2)
            if task == "boolq" and cell and cell not in {"FAIL", "SKIP"}:
                cell = f"{cell} ({boolq_batch_size}bs)"
            values.append(cell)
        rows.append([method, *values, average(raw_values, percent=True)])
    return md_table(["", *acc_tasks, "avg"], rows)

def acc_norm_table(model):
    norm_tasks = ["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "boolq"]
    rows = []
    for method in methods:
        raw_values = [metric_value(model, method, task, "acc_norm") for task in norm_tasks]
        values = [format_value(value, percent=True, decimals=2) for value in raw_values]
        rows.append([f"{method}(norm)", *values, average(raw_values, percent=True)])
    return md_table(["", *norm_tasks, "avg"], rows)

def failed_rows():
    rows = []
    for model in models:
        for method in methods:
            for task in tasks:
                status = statuses.get((model, method, task))
                if status and status != "OK":
                    rows.append([model, method, task, status, logs.get((model, method, task), "")])
    return rows

lines = [
    "# Quantization Evaluation Result",
    "",
    f"- Run id: {run_id}",
    f"- Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"- Weight bit: {wbit}",
    f"- Activation bit: {abit}",
    f"- Group size: {group_size}",
    f"- Default batch size: {batch_size}",
    f"- BoolQ batch size: {boolq_batch_size}",
    f"- Methods: {', '.join(methods)}",
    f"- Models: {', '.join(models)}",
    f"- Tasks: {', '.join(tasks)}",
    f"- Log dir: {log_dir}",
    "",
]

for model in models:
    lines.extend([
        f"## {model}",
        "",
        "### PPL",
        "",
        ppl_table(model),
        "",
        "### Accuracy",
        "",
        acc_table(model),
        "",
        "### Accuracy Norm",
        "",
        acc_norm_table(model),
        "",
    ])

failures = failed_rows()
if failures:
    lines.extend([
        "## Failed or Skipped Tasks",
        "",
        md_table(["Model", "Method", "Task", "Status", "Log"], failures),
        "",
    ])

result_file.write_text("\n".join(lines), encoding="utf-8")
PY
}

run_one_task() {
    local model_name="$1"
    local model_path="$2"
    local method="$3"
    local task="$4"
    local safe_model_name="${model_name//[^A-Za-z0-9_.-]/_}"
    local log_file="$LOG_DIR/${RUN_ID}_${safe_model_name}_${method}_${task}.log"
    local status="OK"
    local result="metric=N/A"
    local current_batch_size="$BATCH_SIZE"
    local exit_code=1

    if [ "$task" = "boolq" ]; then
        current_batch_size="$BOOLQ_BATCH_SIZE"
    fi

    echo
    echo "========== Running: model=$model_name method=$method task=$task =========="
    echo "log: $log_file"

    if [ ! -d "$model_path" ]; then
        status="SKIP"
        result="model_path_not_found"
        printf "%-12s %-12s %-16s %-8s %s\n" "$model_name" "$method" "$task" "$status" "$result" | tee -a "$SUMMARY_FILE"
        append_record "$model_name" "$method" "$task" "$status" "$current_batch_size" "$log_file" "error" "$result"
        render_result_file
        return 0
    fi

    local cmd=(
        python -m mxq.entry
        --model_path "$model_path"
        --tasks "$task"
        --batch_size "$current_batch_size"
        --w_bit "$WBIT"
        --w_mode "$method"
        --a_bit "$ABIT"
        --a_mode "$method"
        --group_size "$GROUP_SIZE"
    )

    if [ "$TIMEOUT_SECONDS" -gt 0 ]; then
        timeout "$TIMEOUT_SECONDS" "${cmd[@]}" >"$log_file" 2>&1
    else
        "${cmd[@]}" >"$log_file" 2>&1
    fi
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        local parsed_any=0
        while IFS=$'\t' read -r metric value; do
            if [ -z "${metric:-}" ]; then
                continue
            fi
            parsed_any=1
            append_record "$model_name" "$method" "$task" "$status" "$current_batch_size" "$log_file" "$metric" "$value"
            if [ "$result" = "metric=N/A" ]; then
                result="$metric=$value"
            else
                result="$result $metric=$value"
            fi
        done < <(parse_metrics "$task" "$log_file")

        if [ "$parsed_any" -eq 0 ]; then
            result="metric=N/A"
            append_record "$model_name" "$method" "$task" "$status" "$current_batch_size" "$log_file" "metric" "N/A"
        fi
    else
        status=$(classify_failure "$log_file")
        result="see $(basename "$log_file")"
        append_record "$model_name" "$method" "$task" "$status" "$current_batch_size" "$log_file" "error" "$result"
    fi

    cleanup_gpu
    render_result_file

    printf "%-12s %-12s %-16s %-8s %s (bs=%s)\n" "$model_name" "$method" "$task" "$status" "$result" "$current_batch_size" | tee -a "$SUMMARY_FILE"
}

upload_result() {
    echo
    echo "========== Uploading result.md =========="
    curl -fS -X PUT "$UPLOAD_URL" \
        -H "Authorization: Bearer $UPLOAD_TOKEN" \
        -H "X-Filename: result.md" \
        --data-binary @"$RESULT_FILE"
}

if [ "${#MODEL_NAMES[@]}" -ne "${#MODEL_PATHS[@]}" ]; then
    echo "MODEL_NAMES and MODEL_PATHS length mismatch" >&2
    exit 1
fi

init_result_file

{
    echo "Run id: $RUN_ID"
    echo "Result file: $RESULT_FILE"
    echo "Log dir: $LOG_DIR"
    echo
    printf "%-12s %-12s %-16s %-8s %s\n" "Model" "Method" "Task" "Status" "Result"
    printf "%-12s %-12s %-16s %-8s %s\n" "-----" "------" "----" "------" "------"
} | tee "$SUMMARY_FILE"

for model_index in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$model_index]}"
    model_path="${MODEL_PATHS[$model_index]}"
    for method in "${METHODS[@]}"; do
        for task in "${TASKS[@]}"; do
            run_one_task "$model_name" "$model_path" "$method" "$task"
        done

        render_result_file
        if ! upload_result; then
            echo "Upload failed after model=$model_name method=$method; result.md is still available at $RESULT_FILE" >&2
        fi
    done
done

echo
echo "========== Summary =========="
cat "$SUMMARY_FILE"
echo
echo "Result file: $RESULT_FILE"

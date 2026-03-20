#!/bin/bash
set -euo pipefail

# Dedicated launcher for openPangu-R-72B-2512 (local transformers path only).

TASKS=${1:-"SuperGPQA"}
QUANT_METHOD=${QUANT_METHOD:-"mxfp"}
MODEL=${MODEL:-"FreedomIntelligence/openPangu-R-72B-2512"}
BACKEND=${BACKEND:-"transformers"}
if [ -z "${MAX_MODEL_LENGTH+x}" ]; then
  MAX_MODEL_LENGTH=4096
fi
if [ -z "${MAX_NEW_TOKENS+x}" ]; then
  MAX_NEW_TOKENS=96
fi
if [ -z "${MAX_SAMPLES+x}" ]; then
  MAX_SAMPLES=""
fi
OVERWRITE=${OVERWRITE:-1}
PYTHON_BIN=${PYTHON_BIN:-python}
HF_HOME=${HF_HOME:-"/cephfs/shared/xyli/hf_cache"}
AUTO_TUNE_MEM=${AUTO_TUNE_MEM:-1}
MODEL_PARALLEL=${MODEL_PARALLEL:-1}
NUM_PROCESSES=${NUM_PROCESSES:-}
TRANSFORMERS_MP_MODE=${TRANSFORMERS_MP_MODE:-"single_process"}
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
fi

if [ "${BACKEND}" != "transformers" ]; then
  echo "[reasoning_pangu] BACKEND=${BACKEND} is not supported in this script anymore." >&2
  echo "[reasoning_pangu] forcing BACKEND=transformers (omni/openai path removed)." >&2
  BACKEND="transformers"
fi

if [ "${AUTO_TUNE_MEM}" = "1" ]; then
  MEM_LIMIT_BYTES=""
  if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    MEM_LIMIT_BYTES="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
  elif [ -f /sys/fs/cgroup/memory.max ]; then
    MEM_LIMIT_BYTES="$(cat /sys/fs/cgroup/memory.max)"
  fi

  # If running under a ~100GiB memcg, clamp defaults to safer values.
  if [ -n "${MEM_LIMIT_BYTES}" ] && [ "${MEM_LIMIT_BYTES}" != "max" ]; then
    MEM_LIMIT_GB=$((MEM_LIMIT_BYTES / 1024 / 1024 / 1024))
    if [ "${MEM_LIMIT_GB}" -le 110 ]; then
      if [ "${MAX_NEW_TOKENS}" -gt 256 ]; then
        MAX_NEW_TOKENS=256
      fi
      if [ "${MAX_MODEL_LENGTH}" -gt 3072 ]; then
        MAX_MODEL_LENGTH=3072
      fi
    fi
  fi
fi

echo "[reasoning_pangu] model=${MODEL}"
echo "[reasoning_pangu] dataset=${TASKS}"
echo "[reasoning_pangu] backend=${BACKEND}"
echo "[reasoning_pangu] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[reasoning_pangu] hf_home=${HF_HOME}"
echo "[reasoning_pangu] max_new_tokens=${MAX_NEW_TOKENS}"
echo "[reasoning_pangu] max_model_length=${MAX_MODEL_LENGTH}"
echo "[reasoning_pangu] max_samples=${MAX_SAMPLES:-all}"
echo "[reasoning_pangu] overwrite=${OVERWRITE}"
echo "[reasoning_pangu] python_bin=${PYTHON_BIN}"
echo "[reasoning_pangu] auto_tune_mem=${AUTO_TUNE_MEM}"
echo "[reasoning_pangu] model_parallel=${MODEL_PARALLEL}"
echo "[reasoning_pangu] num_processes=${NUM_PROCESSES:-auto}"
echo "[reasoning_pangu] transformers_mp_mode=${TRANSFORMERS_MP_MODE}"

export HF_HOME

EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES}" ]; then
  EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${OVERWRITE}" = "1" ]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [ "${MODEL_PARALLEL}" = "1" ]; then
  EXTRA_ARGS+=(--model_parallel)
fi

INFER_ARGS=(
  --model "$MODEL"
  --dataset "$TASKS"
  --backend "$BACKEND"
  --quant_method "$QUANT_METHOD"
  --max_model_length "$MAX_MODEL_LENGTH"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --trust_remote_code
  "${EXTRA_ARGS[@]}"
)

if [ "${MODEL_PARALLEL}" = "1" ] && [ "${TRANSFORMERS_MP_MODE}" = "multi_process" ]; then
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "[reasoning_pangu] MODEL_PARALLEL=1 requires accelerate, but `accelerate` is not found." >&2
    echo "[reasoning_pangu] install with: pip install accelerate" >&2
    exit 1
  fi
  if [ -z "${NUM_PROCESSES}" ]; then
    NUM_PROCESSES="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  fi
  if [ "${NUM_PROCESSES}" -le 1 ]; then
    echo "[reasoning_pangu] MODEL_PARALLEL=1 but only ${NUM_PROCESSES} GPU detected." >&2
    echo "[reasoning_pangu] fallback to single-process transformers run." >&2
    "${PYTHON_BIN}" -m mxq.evaluation.inference "${INFER_ARGS[@]}"
  else
    echo "[reasoning_pangu] launching with accelerate, num_processes=${NUM_PROCESSES}"
    accelerate launch --num_processes "${NUM_PROCESSES}" -m mxq.evaluation.inference "${INFER_ARGS[@]}"
  fi
else
  if [ "${MODEL_PARALLEL}" = "1" ]; then
    echo "[reasoning_pangu] using single-process model_parallel path (recommended for current lighteval)."
  fi
  "${PYTHON_BIN}" -m mxq.evaluation.inference "${INFER_ARGS[@]}"
fi

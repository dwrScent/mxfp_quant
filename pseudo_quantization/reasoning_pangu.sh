#!/bin/bash
set -euo pipefail

# Dedicated launcher for openPangu-R-72B-2512 with omni-infer deployment support.
# Default path: omni-infer OpenAI-compatible API -> lighteval(openai backend).

TASKS=${1:-"SuperGPQA"}
QUANT_METHOD=${QUANT_METHOD:-"mxfp"}
MODEL=${MODEL:-"FreedomIntelligence/openPangu-R-72B-2512"}
BACKEND=${BACKEND:-"openai"}
if [ -z "${MAX_MODEL_LENGTH+x}" ]; then
  MAX_MODEL_LENGTH=4096
fi
if [ -z "${MAX_NEW_TOKENS+x}" ]; then
  MAX_NEW_TOKENS=1024
fi
API_BASE_URL=${API_BASE_URL:-"http://127.0.0.1:8000/v1"}
API_KEY=${API_KEY:-"EMPTY"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"openpangu_r_72b_2512"}
HF_HOME=${HF_HOME:-"/cephfs/shared/xyli/hf_cache"}
AUTO_TUNE_MEM=${AUTO_TUNE_MEM:-1}

OMNI_AUTO_DEPLOY=${OMNI_AUTO_DEPLOY:-0}
OMNI_SERVE_SCRIPT=${OMNI_SERVE_SCRIPT:-"/path/to/omniinfer/tools/scripts/start_serving_openpangu_r_72b_2512.sh"}
OMNI_START_TIMEOUT=${OMNI_START_TIMEOUT:-600}

if [ "${BACKEND}" = "transformers" ] && [ "${AUTO_TUNE_MEM}" = "1" ]; then
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
echo "[reasoning_pangu] api_base_url=${API_BASE_URL}"
echo "[reasoning_pangu] hf_home=${HF_HOME}"
echo "[reasoning_pangu] max_new_tokens=${MAX_NEW_TOKENS}"
echo "[reasoning_pangu] max_model_length=${MAX_MODEL_LENGTH}"
echo "[reasoning_pangu] auto_tune_mem=${AUTO_TUNE_MEM}"

export HF_HOME

if [ "${OMNI_AUTO_DEPLOY}" = "1" ]; then
  echo "[reasoning_pangu] starting omni-infer service via ${OMNI_SERVE_SCRIPT}"
  if [ ! -f "${OMNI_SERVE_SCRIPT}" ]; then
    echo "[reasoning_pangu] omni serve script not found: ${OMNI_SERVE_SCRIPT}" >&2
    exit 1
  fi
  if grep -q "/path/to/model/" "${OMNI_SERVE_SCRIPT}" || grep -q "/path/to/omniinfer/" "${OMNI_SERVE_SCRIPT}"; then
    echo "[reasoning_pangu] serving script still contains placeholders (/path/to/model or /path/to/omniinfer)." >&2
    echo "[reasoning_pangu] please edit ${OMNI_SERVE_SCRIPT} before auto deploy." >&2
    exit 1
  fi
  if ! command -v npu-smi >/dev/null 2>&1; then
    echo "[reasoning_pangu] auto deploy requires Ascend/omniinfer runtime (npu-smi not found)." >&2
    echo "[reasoning_pangu] current environment looks like standard CUDA vllm, which cannot run this serve script." >&2
    exit 1
  fi
  OMNI_SCRIPT_DIR="$(cd "$(dirname "${OMNI_SERVE_SCRIPT}")" && pwd)"
  OMNI_SCRIPT_NAME="$(basename "${OMNI_SERVE_SCRIPT}")"
  (
    cd "${OMNI_SCRIPT_DIR}"
    bash "${OMNI_SCRIPT_NAME}"
  )
fi

if [ "${BACKEND}" = "openai" ]; then
  echo "[reasoning_pangu] waiting for omni-infer endpoint..."
  python - <<'PY'
import os
import time
import urllib.request
import urllib.error

base_url = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
timeout = int(os.environ.get("OMNI_START_TIMEOUT", "600"))
deadline = time.time() + timeout
last_err = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=3) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read(200).decode("utf-8", errors="ignore")
        if status == 200:
            print(f"[reasoning_pangu] endpoint ready: {base_url}/models")
            break
        last_err = f"status={status}, body={body}"
    except Exception as e:
        last_err = repr(e)
    time.sleep(3)
else:
    raise SystemExit(
        "[reasoning_pangu] omni-infer endpoint is not ready within timeout. "
        f"last_error={last_err}"
    )
PY
fi

python -m mxq.evaluation.inference \
  --model "$MODEL" \
  --dataset "$TASKS" \
  --backend "$BACKEND" \
  --quant_method "$QUANT_METHOD" \
  --max_model_length "$MAX_MODEL_LENGTH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --api_base_url "$API_BASE_URL" \
  --api_key "$API_KEY" \
  --served_model_name "$SERVED_MODEL_NAME" \
  --trust_remote_code

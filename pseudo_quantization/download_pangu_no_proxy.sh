#!/usr/bin/env bash
set -euo pipefail

# Resume-safe downloader for FreedomIntelligence/openPangu-R-72B-2512.
# Designed for direct (no-proxy) environments and cephfs-backed cache.
# Supports ModelScope-first strategy, then HF-compatible mirrors and git fallback.

MODEL_ID="${MODEL_ID:-FreedomIntelligence/openPangu-R-72B-2512}"
HF_HOME_DEFAULT="/cephfs/shared/xyli/hf_cache"
HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
RETRIES="${RETRIES:-20}"
SLEEP_SECS="${SLEEP_SECS:-30}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mxq/bin/python}"
MODELSCOPE_MODEL_ID="${MODELSCOPE_MODEL_ID:-FreedomIntelligence/openPangu-R-72B-2512}"
# ModelScope git-lfs repo fallback.
MODELSCOPE_GIT_URL="${MODELSCOPE_GIT_URL:-https://www.modelscope.cn/FreedomIntelligence/openPangu-R-72B-2512.git}"
# Comma-separated fallback list. Order matters.
# You can override, e.g.:
# HF_ENDPOINTS="https://hf-mirror.com,https://huggingface.co"
HF_ENDPOINTS="${HF_ENDPOINTS:-https://hf-mirror.com,https://huggingface.co}"
# Optional git-lfs mirror fallback (used only when HF endpoint flow fails).
# Keep empty if you do not want git fallback.
# Example:
# GIT_MIRROR_URLS="https://gitee.com/hf-models/FreedomIntelligence--openPangu-R-72B-2512.git,https://github.com/FreedomIntelligence/openPangu-R-72B-2512.git"
GIT_MIRROR_URLS="${GIT_MIRROR_URLS:-https://gitee.com/hf-models/FreedomIntelligence--openPangu-R-72B-2512.git,https://github.com/FreedomIntelligence/openPangu-R-72B-2512.git}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  echo "[download_pangu_no_proxy] python not found." >&2
  exit 1
fi

mkdir -p "$HF_HOME"/{hub,transformers}
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Force no-proxy mode.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
export HF_HUB_DISABLE_TELEMETRY=1
# Disable by default to avoid hard failure when hf_transfer package is missing.
# Override with HF_HUB_ENABLE_HF_TRANSFER=1 only if hf_transfer is installed.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MODEL_ID RETRIES SLEEP_SECS HF_ENDPOINTS MODELSCOPE_MODEL_ID

echo "[download_pangu_no_proxy] MODEL_ID=$MODEL_ID"
echo "[download_pangu_no_proxy] MODELSCOPE_MODEL_ID=$MODELSCOPE_MODEL_ID"
echo "[download_pangu_no_proxy] HF_HOME=$HF_HOME"
echo "[download_pangu_no_proxy] PYTHON_BIN=$PYTHON_BIN"
echo "[download_pangu_no_proxy] RETRIES=$RETRIES SLEEP_SECS=$SLEEP_SECS"
echo "[download_pangu_no_proxy] MODELSCOPE_GIT_URL=$MODELSCOPE_GIT_URL"
echo "[download_pangu_no_proxy] HF_ENDPOINTS=$HF_ENDPOINTS"
echo "[download_pangu_no_proxy] GIT_MIRROR_URLS=$GIT_MIRROR_URLS"

echo "[download_pangu_no_proxy] trying ModelScope SDK first..."
if "$PYTHON_BIN" - <<'PY'
import os
import time

model_id = os.environ.get("MODELSCOPE_MODEL_ID", "FreedomIntelligence/openPangu-R-72B-2512")
retries = int(os.environ.get("RETRIES", "20"))
sleep_secs = int(os.environ.get("SLEEP_SECS", "30"))
cache_dir = os.path.join(os.environ.get("HF_HOME"), "modelscope")

try:
    from modelscope import snapshot_download
except Exception as e:
    raise SystemExit(f"[download_pangu_no_proxy] ModelScope SDK unavailable: {repr(e)}")

last_err = None
for i in range(1, retries + 1):
    try:
        local_path = snapshot_download(model_id=model_id, cache_dir=cache_dir)
        print(f"[download_pangu_no_proxy] ModelScope snapshot ready: {local_path}")
        raise SystemExit(0)
    except Exception as e:
        last_err = e
        print(f"[download_pangu_no_proxy] ModelScope attempt {i}/{retries} failed: {repr(e)}")
        if i < retries:
            time.sleep(sleep_secs)

raise SystemExit(f"[download_pangu_no_proxy] ModelScope SDK download failed: {repr(last_err)}")
PY
then
  echo "[download_pangu_no_proxy] done via ModelScope SDK."
  exit 0
fi

echo "[download_pangu_no_proxy] ModelScope SDK failed; trying ModelScope git-lfs..."
if command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1; then
  git lfs install >/dev/null 2>&1 || true
  MS_LOCAL_DIR="$HF_HOME/modelscope_models/openPangu-R-72B-2512"
  mkdir -p "$(dirname "$MS_LOCAL_DIR")"
  rm -rf "$MS_LOCAL_DIR.tmp"
  if git clone "$MODELSCOPE_GIT_URL" "$MS_LOCAL_DIR.tmp"; then
    (
      cd "$MS_LOCAL_DIR.tmp"
      git lfs pull
    )
    rm -rf "$MS_LOCAL_DIR"
    mv "$MS_LOCAL_DIR.tmp" "$MS_LOCAL_DIR"
    echo "[download_pangu_no_proxy] done via ModelScope git-lfs: $MS_LOCAL_DIR"
    echo "[download_pangu_no_proxy] use local model path with inference:"
    echo "  --model $MS_LOCAL_DIR"
    exit 0
  fi
fi

echo "[download_pangu_no_proxy] ModelScope path failed; falling back to HF-compatible endpoints..."
if "$PYTHON_BIN" - <<'PY'
import os
import socket
import sys
import time
import re
import glob

from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "FreedomIntelligence/openPangu-R-72B-2512")
retries = int(os.environ.get("RETRIES", "20"))
sleep_secs = int(os.environ.get("SLEEP_SECS", "30"))
endpoints = [x.strip().rstrip("/") for x in os.environ.get("HF_ENDPOINTS", "").split(",") if x.strip()]
if not endpoints:
    endpoints = ["https://hf-mirror.com", "https://huggingface.co"]

def verify_shards(local_path: str):
    files = glob.glob(os.path.join(local_path, "model-*-of-*.safetensors"))
    if not files:
        # No shard naming convention detected; skip strict check.
        return True, "no_shard_pattern_detected"
    totals = set()
    for f in files:
        m = re.search(r"of-(\d+)\.safetensors$", os.path.basename(f))
        if m:
            totals.add(int(m.group(1)))
    if not totals:
        return True, "no_total_detected"
    if len(totals) != 1:
        return False, f"inconsistent_total_markers={sorted(totals)}"
    expected = next(iter(totals))
    actual = len(files)
    if actual < expected:
        return False, f"shards_incomplete actual={actual} expected={expected}"
    index_file = os.path.join(local_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        return False, "missing model.safetensors.index.json"
    return True, f"shards_complete actual={actual} expected={expected}"

def host_from_endpoint(endpoint: str) -> str:
    # https://hf-mirror.com -> hf-mirror.com
    return endpoint.split("://", 1)[-1].split("/", 1)[0]

all_errors = []
for endpoint in endpoints:
    host = host_from_endpoint(endpoint)
    try:
        socket.gethostbyname(host)
    except Exception as e:
        all_errors.append(f"{endpoint}: dns_fail={repr(e)}")
        print(f"[download_pangu_no_proxy] skip endpoint (dns fail): {endpoint} err={repr(e)}")
        continue

    print(f"[download_pangu_no_proxy] trying endpoint: {endpoint}")
    last_err = None
    for i in range(1, retries + 1):
        try:
            local_path = snapshot_download(
                repo_id=model_id,
                resume_download=True,
                endpoint=endpoint,
            )
            ok, info = verify_shards(local_path)
            if not ok:
                raise RuntimeError(f"snapshot not complete: {info}")
            print(f"[download_pangu_no_proxy] snapshot ready via {endpoint}: {local_path} ({info})")
            raise SystemExit(0)
        except Exception as e:
            last_err = e
            print(
                f"[download_pangu_no_proxy] endpoint={endpoint} attempt {i}/{retries} failed: {repr(e)}"
            )
            if i < retries:
                time.sleep(sleep_secs)
    all_errors.append(f"{endpoint}: {repr(last_err)}")

raise SystemExit(
    "[download_pangu_no_proxy] all endpoints failed.\n"
    "Tried:\n- " + "\n- ".join(all_errors)
)
PY
then
  echo "[download_pangu_no_proxy] done."
  echo "[download_pangu_no_proxy] cache path: $HUGGINGFACE_HUB_CACHE/models--${MODEL_ID/\//--}"
  exit 0
fi

echo "[download_pangu_no_proxy] HF endpoint flow failed; trying git-lfs mirrors..."
if ! command -v git >/dev/null 2>&1; then
  echo "[download_pangu_no_proxy] git not found; cannot use git fallback." >&2
  exit 1
fi
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[download_pangu_no_proxy] git-lfs not found; cannot pull large weight files via git." >&2
  echo "[download_pangu_no_proxy] install git-lfs then rerun." >&2
  exit 1
fi

git lfs install >/dev/null 2>&1 || true

LOCAL_REPO_DIR="$HF_HOME/manual_models/openPangu-R-72B-2512"
mkdir -p "$(dirname "$LOCAL_REPO_DIR")"

IFS=',' read -r -a _git_urls <<< "$GIT_MIRROR_URLS"
for url in "${_git_urls[@]}"; do
  url="$(echo "$url" | xargs)"
  [ -z "$url" ] && continue
  echo "[download_pangu_no_proxy] trying git mirror: $url"
  rm -rf "$LOCAL_REPO_DIR.tmp"
  if git clone "$url" "$LOCAL_REPO_DIR.tmp"; then
    (
      cd "$LOCAL_REPO_DIR.tmp"
      git lfs pull
    )
    rm -rf "$LOCAL_REPO_DIR"
    mv "$LOCAL_REPO_DIR.tmp" "$LOCAL_REPO_DIR"
    echo "[download_pangu_no_proxy] git mirror ready: $LOCAL_REPO_DIR"
    echo "[download_pangu_no_proxy] use local model path with inference:"
    echo "  --model $LOCAL_REPO_DIR"
    echo "[download_pangu_no_proxy] done."
    exit 0
  fi
done

echo "[download_pangu_no_proxy] all git mirrors failed." >&2
exit 1

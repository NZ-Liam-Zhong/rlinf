#!/bin/bash
set -euo pipefail

# One-click launcher for RLinf DreamZero LIBERO SFT on j02.
# It starts Ray locally, then launches training in a detached tmux session.
#
# Usage:
#   bash run_dreamzero_sft_ray_tmux.sh
#
# Key paths (override via env vars):
#   MODEL_PATH   - DreamZero-LIBERO config dir (no safetensors needed)
#   TOKENIZER_PATH - umt5-xxl tokenizer dir
#   DATA_PATH    - LIBERO-10 dataset root (must contain meta/)
#   PROJ_DIR     - path to this rlinf repo
#   VENV_ACTIVATE - path to venv_dreamzero activate script

SESSION_NAME="${SESSION_NAME:-dz_wan_sft}"
RAY_PORT="${RAY_PORT:-6399}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8266}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_znz}"
LOG_FILE="${LOG_FILE:-/tmp/dz_wan_sft.log}"

PROJ_DIR="${PROJ_DIR:-/mnt/project_rlinf/znz/rlinf_fresh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/project_rlinf/znz/venv_dreamzero/bin/activate}"
CONFIG_NAME="${CONFIG_NAME:-libero_sft_dreamzero}"

WANDB_PROJECT="${WANDB_PROJECT:-rlinf}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

MODEL_PATH="/mnt/project_rlinf/znz/checkpoints/DreamZero-LIBERO"
TOKENIZER_PATH="/mnt/project_rlinf_hs/yuanhuining/models/umt5-xxl"
DATA_PATH="/mnt/project_rlinf/znz/datasets/libero_10"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "ERROR: venv activate script not found: ${VENV_ACTIVATE}"
  exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "ERROR: DreamZero config.json not found: ${MODEL_PATH}"
  exit 1
fi

if [ ! -f "${TOKENIZER_PATH}/tokenizer_config.json" ]; then
  echo "ERROR: tokenizer not found: ${TOKENIZER_PATH}"
  exit 1
fi

if [ ! -d "${DATA_PATH}/meta" ]; then
  echo "ERROR: LIBERO dataset not found: ${DATA_PATH}"
  exit 1
fi

if [ ! -d "${PROJ_DIR}" ]; then
  echo "ERROR: project dir not found: ${PROJ_DIR}"
  exit 1
fi

source "${VENV_ACTIVATE}"
PYTHON_BIN="$(which python)"

echo "== GPU status =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo "== Restart Ray =="
"${PYTHON_BIN}" -m ray.scripts.scripts stop --force >/tmp/ray_stop_dz.log 2>&1 || true
"${PYTHON_BIN}" -m ray.scripts.scripts start --head --num-gpus=2 --port="${RAY_PORT}" --dashboard-port="${DASHBOARD_PORT}" --temp-dir="${RAY_TEMP_DIR}" >/tmp/ray_start_dz.log 2>&1

LOCAL_IP="$(hostname -I | awk '{for(i=1;i<=NF;i++) if ($i !~ /^172\.17\./ && $i !~ /^127\./){print $i; exit}}')"
if [ -z "${LOCAL_IP}" ]; then
  LOCAL_IP="$(hostname -I | awk '{print $1}')"
fi

export RAY_ADDRESS="${LOCAL_IP}:${RAY_PORT}"

echo "== Launch tmux session: ${SESSION_NAME} =="
tmux has-session -t "${SESSION_NAME}" 2>/dev/null && tmux kill-session -t "${SESSION_NAME}"

tmux new-session -d -s "${SESSION_NAME}" "bash -lc '
  set -e
  source \"${VENV_ACTIVATE}\"
  export TORCH_COMPILE_DISABLE=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WANDB_PROJECT=\"${WANDB_PROJECT}\"
  export WANDB_API_KEY=\"${WANDB_API_KEY}\"
  export WANDB_MODE=offline
  export RAY_ADDRESS=\"${RAY_ADDRESS}\"
  export PYTHONPATH=\"${PROJ_DIR}:\$PYTHONPATH\"
  cd \"${PROJ_DIR}\"
  bash examples/sft/run_vla_sft.sh \"${CONFIG_NAME}\" 2>&1 | tee \"${LOG_FILE}\"
'"

echo "Launched."
echo "Session: ${SESSION_NAME}"
echo "Ray: ${RAY_ADDRESS}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Monitor:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "  tail -f ${LOG_FILE}"
echo "  grep 'train/loss' ${LOG_FILE}"

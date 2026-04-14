#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MODEL_TYPE="${1:-lstm}"

cd "${SCRIPT_DIR}"

if [[ "${MODEL_TYPE}" != "lstm" && "${MODEL_TYPE}" != "transformer" ]]; then
  echo "Usage: ./run_model.sh [lstm|transformer]"
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}."
  echo "Create it once with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install pandas pyarrow scikit-learn tqdm torch torchvision torchaudio"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is not available in the current environment."
  exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/hw5_ske.py" ]]; then
  echo "Could not find hw5_ske.py in ${SCRIPT_DIR}."
  exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/hw5_data_train.parquet" ]]; then
  echo "Could not find hw5_data_train.parquet in ${SCRIPT_DIR}."
  exit 1
fi

python3 - <<'PY'
import importlib.util
import sys

missing = [name for name in ("pandas", "torch", "pyarrow", "sklearn") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages: {}".format(", ".join(missing)))
    print("Install them in your active environment, for example:")
    print("  pip install pandas pyarrow scikit-learn tqdm torch torchvision torchaudio")
    sys.exit(1)
PY

python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("No CUDA GPU is available in this shell.")
    print("Start a Grace GPU shell first, for example:")
    print("  srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:t4:1 --time=01:00:00 --pty bash")
    sys.exit(1)

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
PY

exec python3 hw5_ske.py "${MODEL_TYPE}"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MODEL_TYPE="${1:-lstm}"
TARGET_DIR="/scratch/user/pranav1014/LSTM"

if [[ -d "${TARGET_DIR}" ]]; then
  cd "${TARGET_DIR}"
else
  cd "${SCRIPT_DIR}"
fi

if [[ "${MODEL_TYPE}" != "lstm" && "${MODEL_TYPE}" != "transformer" ]]; then
  echo "Usage: ./run_model.sh [lstm|transformer]"
  echo "Example:"
  echo "  cd /scratch/user/pranav1014/LSTM"
  echo "  ./run_model.sh lstm"
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
    print("  srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gres=gpu:a100:1 --time=01:00:00 --pty bash")
    sys.exit(1)

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
gpu_name = torch.cuda.get_device_name(0)
print("GPU name:", gpu_name)

try:
    probe = torch.randn(2, 2, device="cuda")
    _ = probe @ probe.t()
    torch.cuda.synchronize()
except Exception as exc:
    print("CUDA runtime probe failed on this GPU:", exc)
    print("If you are targeting an A100, your environment likely needs a newer CUDA-enabled PyTorch build.")
    print("Example A100 shell:")
    print("  srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gres=gpu:a100:1 --time=01:00:00 --pty bash")
    sys.exit(1)
PY

exec python3 hw5_ske.py "${MODEL_TYPE}"

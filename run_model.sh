#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

cd "${SCRIPT_DIR}"

if [[ -d "${VENV_DIR}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

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

missing = [name for name in ("pandas", "torch", "pyarrow") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages: {}".format(", ".join(missing)))
    print("Install them in your active environment, for example:")
    print("  python -m pip install pandas torch pyarrow")
    sys.exit(1)
PY

exec python3 hw5_ske.py

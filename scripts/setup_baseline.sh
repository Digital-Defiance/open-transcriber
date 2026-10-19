#!/usr/bin/env bash
# Install Spotify Basic Pitch for T0 on macOS / Python 3.12+.
# basic-pitch's published dependency wants tensorflow-macos<2.15 which is not
# on PyPI for 3.12; we use the CoreML runtime instead.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python="${PYTHON:-python3}"
"$python" -m pip install --no-deps 'basic-pitch==0.4.0'
"$python" -m pip install 'mir-eval>=0.6' 'pretty-midi>=0.2.9' 'resampy>=0.2.2,<0.4.3' 'setuptools<81'
if [[ "$(uname -s)" == Darwin ]]; then
  "$python" -m pip install coremltools
fi
"$python" -m pip install -e "$root" --no-deps
"$python" -c "from basic_pitch import ICASSP_2022_MODEL_PATH; print('Basic Pitch ready:', ICASSP_2022_MODEL_PATH)"

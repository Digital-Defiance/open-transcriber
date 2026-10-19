#!/usr/bin/env bash
# Omnizart (MIT) for chord transcription. Needs portaudio on macOS (brew install portaudio).
set -euo pipefail
python="${PYTHON:-python3}"
"$python" -m pip install 'omnizart==0.6.3'
"$python" -c "import omnizart; print('Omnizart ready:', omnizart.__version__)"
echo "Next (once): omnizart download-checkpoints"

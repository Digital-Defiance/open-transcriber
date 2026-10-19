#!/usr/bin/env bash
# Omnizart (MIT) for chord transcription. Needs portaudio on macOS (brew install portaudio).
set -euo pipefail
python="${PYTHON:-python3}"
"$python" -m pip install 'omnizart==0.6.3'
"$python" -c "import omnizart; print('Omnizart ready:', omnizart.__version__)"

# On Apple Silicon macOS, compile native arm64 nnls-chroma Vamp plugin if needed.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    vamp_dir="$("$python" -c 'import omnizart, pathlib; print(pathlib.Path(omnizart.__file__).parent / "resource" / "vamp")')"
    target_dylib="$vamp_dir/nnls-chroma.dylib"
    if [[ -f "$target_dylib" ]] && ! file "$target_dylib" | grep -q "arm64"; then
        echo "Building native arm64 nnls-chroma.dylib for Apple Silicon..."
        which brew >/dev/null || { echo "Homebrew required for vamp-plugin-sdk and boost"; exit 1; }
        brew list vamp-plugin-sdk &>/dev/null || brew install vamp-plugin-sdk
        brew list boost &>/dev/null || brew install boost
        tmp_build="$(mktemp -d)/nnls-chroma"
        git clone --depth 1 https://github.com/c4dm/nnls-chroma.git "$tmp_build"
        (
            cd "$tmp_build"
            make -f Makefile.osx \
                ARCHFLAGS="-arch arm64" \
                VAMP_SDK_DIR="/opt/homebrew/include" \
                BOOST_ROOT="/opt/homebrew/include" \
                LDFLAGS="-arch arm64 -dynamiclib -install_name nnls-chroma.dylib /opt/homebrew/lib/libvamp-sdk.a -exported_symbols_list vamp-plugin.list -framework Accelerate"
            mkdir -p "$HOME/Library/Audio/Plug-Ins/Vamp"
            cp nnls-chroma.dylib "$HOME/Library/Audio/Plug-Ins/Vamp/"
            cp nnls-chroma.{cat,n3} "$HOME/Library/Audio/Plug-Ins/Vamp/" 2>/dev/null || true
            cp nnls-chroma.dylib "$target_dylib"
        )
        rm -rf "$(dirname "$tmp_build")"
        echo "Native arm64 nnls-chroma installed successfully."
    fi
fi

echo "Next (once): omnizart download-checkpoints"


#!/usr/bin/env bash
# One-shot setup on a fresh Linux GPU machine (e.g. a RunPod PyTorch pod):
# installs dependencies, builds pagedserve with and without the CUDA kernel,
# and runs the full test suite on the GPU, including the kernel check.
#
# Usage: bash scripts/gpu_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "== system packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq build-essential pkg-config libpcre2-dev curl git >/dev/null

echo "== python packages"
pip install -q --upgrade cmake transformers tokenizers safetensors openai aiohttp numpy "huggingface_hub[cli]"
hash -r

echo "== GPU and PyTorch"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 - <<'EOF'
import sys, torch
print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}, GPU visible: {torch.cuda.is_available()}")
major, minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
if (major, minor) < (2, 5):
    sys.exit("torch >= 2.5 is required: pip install -U torch  (pick the CUDA build that matches nvcc at pytorch.org)")
if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see the GPU")
EOF
ARCH=$(python3 -c "import torch; m, n = torch.cuda.get_device_capability(); print(f'{m}.{n}')")
TORCH_CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])")
JOBS=$(( $(nproc) < 8 ? $(nproc) : 8 ))

echo "== build: LibTorch reference attention (build/)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev >/dev/null
cmake --build build -j"$JOBS"

NVCC=$(command -v nvcc || true)
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
BUILD=build
if [ -z "$NVCC" ]; then
  echo "!! nvcc not found: skipping the CUDA kernel build (pick a pod image with the CUDA toolkit / 'devel')"
else
  NVCC_MAJOR=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
  if [ "$NVCC_MAJOR" != "$TORCH_CUDA_MAJOR" ]; then
    echo "!! nvcc is CUDA $NVCC_MAJOR but torch was built for CUDA $TORCH_CUDA_MAJOR: skipping the kernel build."
    echo "   Install the torch build for CUDA $NVCC_MAJOR (see pytorch.org) and rerun."
  else
    echo "== build: with CUDA paged-attention kernel for sm_${ARCH/./} (build-cuda/)"
    cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DPAGEDSERVE_CUDA=ON -DCMAKE_CUDA_COMPILER="$NVCC" \
      -DTORCH_CUDA_ARCH_LIST="$ARCH" -DCMAKE_CUDA_ARCHITECTURES="${ARCH/./}" -Wno-dev >/dev/null
    cmake --build build-cuda -j"$JOBS"
    BUILD=build-cuda
  fi
fi

echo "== tests ($BUILD)"
tests/run_all.sh "$BUILD"
echo
echo "Setup complete. Use ./$BUILD/pagedserve  (export BUILD=$BUILD for bench/compare.sh)"

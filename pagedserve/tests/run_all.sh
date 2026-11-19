#!/usr/bin/env bash
# Builds test fixtures (tokenizers + tiny random checkpoints) and runs every test.
# Usage: tests/run_all.sh [build_dir]      (needs: pip install torch transformers tokenizers openai)
set -euo pipefail
BUILD=${1:-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$ROOT/tests/data}
cd "$ROOT"

echo "== fixtures -> $DATA"
if [ ! -d "$DATA/tiny-qwen2" ]; then
  python3 tests/make_test_tokenizers.py "$DATA" >/dev/null
  python3 tests/make_tiny_models.py "$DATA" "$DATA" >/dev/null 2>&1
fi

echo "== block manager (C++ unit tests)"
"$BUILD/test_block_manager"

echo "== tokenizer parity vs HF tokenizers"
python3 tests/test_tokenizer.py "$BUILD" "$DATA/tok-qwen2" "$DATA/tok-llama3" "$DATA/tok-gpt2"

echo "== model logits parity vs HF transformers"
python3 tests/test_model_parity.py "$BUILD" "$DATA/tiny-llama3" "$DATA/tiny-qwen2" "$DATA/tiny-qwen3" 2>/dev/null

echo "== engine end-to-end greedy parity"
for m in tiny-qwen2 tiny-llama3 tiny-qwen3; do
  echo "-- $m"
  python3 tests/test_engine_parity.py "$BUILD" "$DATA/$m" 2>/dev/null
done

echo "== HTTP / SSE server"
python3 tests/test_server.py "$BUILD" "$DATA/tiny-qwen2"

if "$BUILD/pagedserve-dev" attn-check 2>&1 | grep -q "built without"; then
  echo "== CUDA kernel check skipped (not built with -DPAGEDSERVE_CUDA=ON)"
else
  echo "== CUDA paged-attention kernel vs reference"
  "$BUILD/pagedserve-dev" attn-check
fi
echo "ALL TESTS PASSED"

# pagedserve

An OpenAI-compatible LLM inference server in C++20 and LibTorch, built around the ideas behind vLLM:

- **Iteration-level continuous batching** with chunked prefill and a per-step token budget
- **Paged KV cache**: a fixed pool of blocks, per-sequence block tables, on-demand allocation
- **Prefix caching**: full blocks keyed by a chained hash, reused across requests and chat turns, with LRU eviction
- **Reference-counted copy-on-write** block sharing (used when a request forks into `n > 1` choices)
- **Async SSE streaming**: HTTP threads never touch the GPU. A single engine thread owns the model, the scheduler and the KV cache
- An optional **custom CUDA paged-attention decode kernel**

It runs Llama 2/3.x, Mistral, Qwen2/2.5 and Qwen3 checkpoints straight from Hugging Face safetensors.

```
 clients (openai SDK, curl)
        |  POST /v1/chat/completions
        v
 +------------------+   Request    +----------------------------------------------+
 | HTTP I/O threads |  ---------->  | engine thread (owns GPU, scheduler, KV cache) |
 |  parse JSON      |  inbox queue  |                                              |
 |  chat template   |               |  1. drain inbox / aborts                     |
 |  tokenize        |               |  2. schedule: decodes + prefill chunks       |
 |                  |               |     under a token budget; admit by free      |
 |  SSE writer  <---|-- per-request |     KV blocks; prefix-cache lookup; preempt  |
 |  (detects        |   channel     |  3. copy-on-write block copies               |
 |   disconnects)   |               |  4. forward pass over a ragged batch         |
 +------------------+               |  5. sample, stop checks, detokenize, emit    |
                                    +----------------------------------------------+
                                          | block ids                 | K/V tensors
                                    +--------------+          +-------------------+
                                    | BlockManager |          | KVCache           |
                                    | free list,   |          | per layer:        |
                                    | refcounts,   |          | [blocks, 16,      |
                                    | hash -> block|          |  kv_heads, dim]   |
                                    | LRU evictable|          +-------------------+
                                    +--------------+
```

## Build

Requirements: GCC 11+ or Clang 14+, CMake 3.22+, `libpcre2-dev`, Python 3 (only for tests and benchmarks), and LibTorch.
The easiest way to get LibTorch is `pip install torch`, which ships the C++ headers and libraries.

```bash
sudo apt install build-essential cmake libpcre2-dev pkg-config
pip install torch                          # provides LibTorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To use a standalone LibTorch zip instead, pass `-DTORCH_ROOT=/path/to/libtorch`.

**With the custom CUDA kernel** (needs the CUDA toolkit and `nvcc`):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPAGEDSERVE_CUDA=ON \
      -DTORCH_CUDA_ARCH_LIST="8.6"         # your GPU's compute capability (A100 8.0, 3090 8.6, 4090 8.9, H100 9.0)
cmake --build build -j
./build/pagedserve-dev attn-check          # kernel vs reference on random inputs; run this before benchmarking
```

The CUDA kernel is optional. Without it the server still runs on GPU through LibTorch ops: the build detects a CUDA-enabled LibTorch and links `libtorch_cuda` even without a toolkit.

Two build gotchas:

- LibTorch's own CMake config refuses to configure without a CUDA toolkit when the LibTorch build has CUDA. `CMakeLists.txt` falls back to linking the libraries directly in that case.
- LibTorch's CMake ignores `CMAKE_CUDA_ARCHITECTURES`. Use `TORCH_CUDA_ARCH_LIST` instead.

## Quickstart on a rented GPU

On a fresh Linux GPU machine (for example a RunPod "PyTorch" pod whose image includes the CUDA toolkit):

```bash
bash scripts/gpu_setup.sh            # deps, both builds, full test suite on the GPU incl. kernel check
export BUILD=build-cuda              # or build/ if the kernel build was skipped

hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b-instruct
python3 tests/test_tokenizer.py $BUILD models/qwen2.5-0.5b-instruct        # real tokenizer parity
python3 tests/check_real_model.py $BUILD models/qwen2.5-0.5b-instruct      # greedy output == transformers

hf download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json \
   --repo-type dataset --local-dir data
THROUGHPUT_ARGS="--num-requests 1000 --dataset-path data/ShareGPT_V3_unfiltered_cleaned_split.json \
   --tokenizer models/qwen2.5-0.5b-instruct" bench/compare.sh models/qwen2.5-0.5b-instruct
```

(`huggingface-cli download` works too on older `huggingface_hub` versions.)

## Run

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b-instruct
./build/pagedserve --model models/qwen2.5-0.5b-instruct --port 8000
```

On startup the server prints its configuration (example for Qwen2.5-0.5B on a GPU):

```
model        Qwen2ForCausalLM, 494M params, 24 layers, 14/2 q/kv heads, head_dim 64
device       cuda:0, bfloat16
kv cache     ... blocks x 16 tokens = ... GB (12288 bytes/token)
scheduler    continuous batching, max_num_seqs 128, token budget 2048, chunked prefill
prefix cache on
attention    CUDA paged-attention kernel (decode)
chat format  chatml
```

Use it with any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
        model="qwen", stream=True, max_tokens=200,
        messages=[{"role": "user", "content": "Explain paged attention in two sentences."}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

```bash
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen", "stream": true, "max_tokens": 64,
  "messages": [{"role": "user", "content": "Hello!"}]}'
curl localhost:8000/stats      # running/waiting, KV usage, prefix hit rate, preemptions, CoW copies
```

Supported architectures: `Qwen2ForCausalLM` (Qwen2/2.5), `Qwen3ForCausalLM`, `LlamaForCausalLM` (Llama-3.x including its RoPE scaling, SmolLM2) and `MistralForCausalLM` (no sliding window, so stay within 4k context on Mistral-7B-v0.1).
Each architecture is verified against Hugging Face on tiny random-weight checkpoints (see Tests). Run a real checkpoint once and compare a few greedy outputs against `transformers` before relying on it.
The tokenizer supports byte-level BPE only (Qwen, Llama-3, SmolLM2, Mistral-Nemo), so SentencePiece-style models such as Llama-2, TinyLlama and Mistral-7B are rejected with a clear error.
On GPUs without bf16 support (T4, V100), pass `--dtype float16`.

### API

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | `messages`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `n`, `stop`, `stream`, `stream_options.include_usage` |
| `POST /v1/completions` | `prompt` as a string or a list of token ids, plus the same sampling fields |
| `GET /v1/models`, `GET /health`, `GET /stats` | |

Extensions: `top_k`, and `ignore_eos` (for benchmarks).
`usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens came from the prefix cache, the same field OpenAI uses.
Not implemented: logprobs, presence/frequency penalties, per-request seeds, tools/function calling.

Main server flags (`pagedserve --help` lists all of them):

| Flag | Default | |
|---|---|---|
| `--block-size` | 16 | tokens per KV block |
| `--max-num-seqs` | 128 | max sequences per batch |
| `--max-num-batched-tokens` | 2048 | per-step token budget (decode + prefill chunks) |
| `--gpu-memory-utilization` / `--kv-cache-gb` | 0.9 / auto | KV cache sizing |
| `--static-batching` | off | baseline: no admissions until the whole batch finishes |
| `--no-prefix-caching`, `--no-chunked-prefill`, `--no-cuda-kernel` | | ablations |

## How it works

### Paged KV cache (`block_manager.*`, `kv_cache.*`)

- **Pool and block tables.** K and V for each layer live in one preallocated tensor `[num_blocks, block_size, kv_heads, head_dim]`. A sequence owns a block table that maps logical block `i` to a physical block id, like a page table.
- **Where a token is stored.** Token `p` lives at slot `table[p / bs] * bs + p % bs`.
- **On-demand allocation.** Blocks come from an O(1) free list as a sequence grows, so waste is at most one partially filled block per sequence. Reserving the maximum length up front would waste most of the memory.
- **Sizing.** The pool size comes from the GPU memory left after loading the weights.

### Prefix caching

- **Chained keys.** Every full block gets the key `H(parent_key, block_tokens)`. A token's K/V depends on every earlier token (through attention) and on its position (through RoPE), so a block is only reusable if its entire prefix matches. The chain encodes exactly that.
- **Verified hits.** A hit compares the stored tokens too, so a hash collision can never return the wrong K/V.
- **Lookup on admission.** When a request is admitted, its prompt is walked block by block, and every matching block gets a reference and is skipped by prefill. The final prompt token is never matched, because its logits are needed to sample the first output token.
- **Eviction.** When a request finishes, its cached blocks drop to refcount 0 but stay in the hash map on an LRU list, so the next turn of a conversation still hits them. They're reclaimed only when the free list runs out. Release order is tail-first, so prefix heads survive longest.
- **Multi-turn caveat.** Re-tokenizing an assistant reply doesn't always reproduce the exact token ids that were generated, so turn N usually reuses the previous prompt plus most of the reply, not all of it.

### Copy-on-write

Hashed (full) blocks are immutable, so normal prefix sharing never needs a copy. CoW happens when sequences fork:

- A request with `n > 1` prefills its prompt once. Then `n` sequences share every block, including the last, partially filled one.
- Before a sequence writes into a block whose refcount is above 1, `make_writable` allocates a new block and records a `(src, dst)` copy. The engine runs all of a step's copies before the forward pass.

### Scheduler (`scheduler.*`)

On every step:

1. **Running sequences first, oldest first.** A decoding sequence costs one token. A sequence still in prefill gets a chunk of its prompt. Everything shares one token budget (chunked prefill), so a long prompt can't stall everyone else's decode.
2. **Preemption.** If a running sequence needs a block and none is free, the newest sequence is preempted by recompute: its blocks are released (full ones stay in the prefix cache) and it is requeued at the front.
3. **Admission.** Waiting requests are admitted while the budget, `max_num_seqs` and free blocks (minus a small watermark) allow.

`--static-batching` is the baseline: it admits a batch only when the previous one has fully drained. It still doesn't pad, so the only difference from continuous batching is the admission policy.

### Model and attention (`model.*`, `attention.*`)

- **Hand-written forward pass.** The transformer is written against LibTorch rather than loaded from a TorchScript export, because an exported Hugging Face model expects contiguous KV and paged attention needs reads and writes through block tables.
- **Fused GEMMs.** Q/K/V and gate/up are fused into single GEMMs.
- **Ragged batch.** All scheduled tokens are flattened into one 1-D batch with no padding. The LM head runs only on rows that will be sampled.
- **Attention paths:**
  - **Decode** (one new token per sequence) is batched. It uses either the CUDA kernel or a reference path that gathers blocks and calls SDPA.
  - **Prefill chunks** attend causally over `[cached prefix | new tokens]` one sequence at a time.

### CUDA decode kernel (`src/cuda/paged_attention.cu`)

- **Layout.** One thread block per (sequence, query head). Each warp walks a strided subset of the sequence's tokens through the block table. Each lane owns `head_dim / 32` dimensions, so K/V row loads are coalesced.
- **Online softmax.** Each warp keeps a running max, sum and accumulator, so scores are never materialized and context length isn't limited by shared memory. The warps are merged at the end.
- **Coverage.** Supports fp32/fp16/bf16, head_dim 64/96/128/256, and grouped-query attention.

### Engine and streaming (`engine.*`, `server_main.cpp`)

- **Thread ownership.** HTTP threads (cpp-httplib pool) parse and tokenize, then push the request into the engine's inbox. Only the engine thread touches sequences, the scheduler and the block manager, so the hot path needs no locks.
- **Token delivery.** Each request has a channel (a mutex and condition variable). The SSE handler waits on it and writes `data: {chunk}` as tokens arrive, so a slow client only backs up its own channel.
- **Disconnects.** A failed write, or a peer that's gone while waiting, aborts the request. Its KV blocks are freed on the next engine step.
- **Incremental detokenization.** Only complete UTF-8 characters are released (a token can be half an emoji). Text that might be the start of a stop string is held back until it can be ruled out.

## Tests

`tests/run_all.sh build` builds fixtures and runs everything (needs `pip install torch transformers tokenizers openai`).
Real checkpoints aren't needed: the tests create tiny random-weight Llama-3, Qwen2 and Qwen3 models with `transformers`, plus look-alike tokenizers.

| Test | What it proves |
|---|---|
| `test_block_manager` (C++) | allocation, chained-hash prefix matching, LRU eviction order, CoW, duplicate registration |
| `test_tokenizer.py` | token-for-token parity with HF `tokenizers` on 318 strings (Unicode, digits, whitespace, special tokens) for Qwen2-, Llama-3- and GPT-2-style tokenizers |
| `test_model_parity.py` | logits for every position match HF `transformers` (max error about 2e-5 in fp32), through chunked prefill over a cached prefix, decode, scrambled block tables and batching |
| `test_engine_parity.py` | a mixed workload (shared prefixes, a second chat turn, a long chunked prompt, staggered arrivals, `n=3`, exact-block prompts) produces exactly HF's greedy tokens under continuous batching, a tiny KV cache that forces preemption, static batching, and with prefix caching and chunking off |
| `test_server.py` | 17 checks with the official `openai` client: chat and completions, streaming equals non-streaming, `n`, stop strings (including held-back text), context-length errors, `cached_tokens`, 48 concurrent streams, and a mid-stream disconnect that frees the KV blocks |
| `pagedserve-dev attn-check` | CUDA kernel vs. reference attention on random inputs (GPU only) |
| `check_real_model.py` | on a real checkpoint, greedy output and chat-template prompts are identical to `transformers` |

The tests force float32, so on a GPU the engine test also exercises the CUDA kernel end to end. The tiny models use head_dim 32 and 64 for this reason.

## Benchmarks

`bench/bench_serving.py` works against any OpenAI-compatible server, including vLLM, which makes it easy to compare.
`bench/compare.sh <model>` runs both A/B comparisons and prints a summary:

1. **Decode throughput, continuous vs. static batching.** Independent requests with `ignore_eos`, so output lengths are exact. Lengths are either uniform random or taken from real ShareGPT conversations (`--dataset-path`, `--tokenizer`): prompt = first user message, output length = the real reply's token count, filtered like vLLM's benchmark. ShareGPT's long-tailed lengths are the realistic case. Metric: generated tokens per second, plus engine steps and average decode batch.
2. **Shared-prefix multi-turn TTFT, prefix cache on vs. off.** Concurrent conversations share a long system prompt and resend their history every turn. Metric: client-measured time to first token, p50/p95, plus the cache hit rate.

```bash
bench/compare.sh models/qwen2.5-0.5b-instruct
THROUGHPUT_ARGS="--num-requests 500 --output-len 16:1024" bench/compare.sh models/qwen2.5-1.5b-instruct --max-num-seqs 256
```

What to expect, and why:

- **Continuous batching.** The gain comes from variation in output length. Static batching keeps finished slots idle until the longest request in the batch is done. With identical output lengths the two modes perform about the same. On a GPU, decode is memory-bandwidth-bound and step time stays nearly flat as the batch grows, so fewer steps means proportionally higher throughput. On a CPU, decode is compute-bound and the gain mostly disappears.
- **Prefix caching.** It only removes the cached part of prefill. Queueing, the uncached suffix and other requests' prefills are all still in TTFT, so the reduction depends on how much of the prompt is shared and on load.

Smoke run (CPU, 2 cores, an 11M-parameter random model). **This is not a performance result.** It only shows the harness works:

| | result |
|---|---|
| continuous vs. static batching | 1.09x tokens/s; 651 vs. 1042 engine steps for the same 16k output tokens |
| prefix cache on vs. off, p95 TTFT | 1294 ms to 143 ms (91.9% of prompt tokens from cache) |

Record real numbers on a GPU, with a real model, and write down the workload alongside them.

## Limitations and next steps

- **Reference decode path cost.** It gathers every sequence's blocks each layer. That's correct but uses transient memory proportional to batch × context. The kernel avoids it.
- **Simple kernel design.** It doesn't split the context across thread blocks (FlashDecoding-style), so small batches with long contexts underuse the GPU.
- **No CUDA graphs.** Kernel-launch overhead is visible at small batch sizes. CUDA graphs for decode are the obvious next optimization.
- **Hardcoded chat templates.** ChatML, Llama-3 and plain are built in, auto-detected from `tokenizer_config.json`. There's no Jinja engine.
- **Tokenizer gaps.** Byte-level BPE only, and no NFC normalization.
- **Single GPU.** No tensor parallelism and no quantization.
- **Future work:** FlashInfer/FlashAttention kernels, speculative decoding, FP8 KV cache, prefix-aware routing across replicas.

## Layout

```
src/block_manager.*   paged KV bookkeeping, prefix cache, CoW          (no Torch dependency)
src/tokenizer.*       byte-level BPE from tokenizer.json, PCRE2 pre-tokenizer
src/kv_cache.*        KV tensors, block copies
src/attention.*       ragged batch layout, paged attention (reference path)
src/cuda/             CUDA decode kernel
src/model.*           Llama/Qwen forward pass, safetensors loading, RoPE scaling
src/scheduler.*       continuous batching, chunked prefill, preemption, static baseline
src/engine.*          step loop, sampling, fork, detokenization, stats
src/server_main.cpp   HTTP routes, SSE streaming, disconnect handling
src/openai_api.*      request validation and response formats
tools/devtool.cpp     logits dump, offline engine runs, kernel check
tests/, bench/
third_party/          cpp-httplib v0.67.0, nlohmann/json v3.12.0 (MIT)
```

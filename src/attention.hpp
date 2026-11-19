// Batch layout for one model step and paged attention over the KV cache.
//
// A step processes a "ragged" batch: the scheduled tokens of all sequences
// are flattened into one 1-D tensor (no padding), so every linear layer runs
// as a single GEMM over all tokens. Only attention needs per-sequence
// structure, which is described by AttentionMetadata:
//   * decode-style chunks (exactly one new token) come first and are handled
//     together in one batched call (or the custom CUDA kernel);
//   * prefill chunks (several new tokens, possibly after a cached prefix)
//     are handled one sequence at a time.
#pragma once

#include <torch/torch.h>

#include <span>
#include <vector>

#include "block_manager.hpp"

namespace ps {

// One sequence's slice of a step.
struct SeqChunk {
  std::span<const TokenId> tokens;        // all tokens of the sequence
  int start = 0;                          // first position computed this step
  int len = 0;                            // number of positions computed this step
  std::span<const BlockId> block_table;   // must already cover [0, start + len)
  bool logits_last = true;                // want logits for the chunk's last token
  bool logits_all = false;                // want logits for every token (tests)
};

struct PrefillMeta {
  int64_t token_start;   // offset into the flattened batch
  int64_t q_len;         // new tokens
  int64_t context_len;   // cached prefix + new tokens
  torch::Tensor block_ids;  // [ceil(context_len / block_size)] int64
};

struct AttentionMetadata {
  int64_t num_decode = 0;             // first num_decode tokens are single-token queries
  torch::Tensor slot_mapping;         // [T] int64: cache slot for each new token's K/V
  torch::Tensor decode_block_tables;  // [num_decode, max_blocks] int32 (padded with 0)
  torch::Tensor decode_context_lens;  // [num_decode] int32
  int64_t decode_max_context = 0;
  std::vector<PrefillMeta> prefills;
  bool use_cuda_kernel = false;
};

struct ModelInput {
  torch::Tensor input_ids;       // [T] int64
  torch::Tensor positions;       // [T] int64
  torch::Tensor logits_indices;  // [N] int64: rows of the batch to project to vocab
  AttentionMetadata attn;
  int64_t num_tokens = 0;
};

// Chunks with len == 1 must come before chunks with len > 1.
ModelInput build_model_input(std::span<const SeqChunk> chunks, int block_size, torch::Device device);

// q: [T, Hq, D]; k, v: [T, Hkv, D] (already rotated). Writes k/v into the
// cache at meta.slot_mapping, then attends over each sequence's context.
// Returns [T, Hq, D].
torch::Tensor paged_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                              torch::Tensor& k_cache, torch::Tensor& v_cache, const AttentionMetadata& meta,
                              double scale);

// Reference decode attention (gathers blocks, then SDPA). Exposed for tests
// that compare it with the CUDA kernel.
torch::Tensor paged_decode_reference(const torch::Tensor& q, const torch::Tensor& k_cache,
                                     const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                                     const torch::Tensor& context_lens, double scale);

// True if built with the CUDA kernel and it supports this head size.
bool cuda_kernel_available(int64_t head_dim);

}  // namespace ps

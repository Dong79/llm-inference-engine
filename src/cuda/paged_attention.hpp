// Custom CUDA paged-attention kernel for decode (one query token per sequence).
#pragma once

#include <torch/torch.h>

namespace ps {

// q:            [num_seqs, num_heads, head_dim]
// k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
// block_tables: [num_seqs, max_blocks] int32
// context_lens: [num_seqs] int32 (tokens to attend to, including the new one)
// Returns [num_seqs, num_heads, head_dim] in q's dtype.
// Supports float32 / float16 / bfloat16 and head_dim 32, 64, 96, 128, 256.
torch::Tensor paged_attention_decode_cuda(const torch::Tensor& q, const torch::Tensor& k_cache,
                                          const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                                          const torch::Tensor& context_lens, int64_t max_context, double scale);

bool paged_attention_decode_cuda_supported(int64_t head_dim);

}  // namespace ps

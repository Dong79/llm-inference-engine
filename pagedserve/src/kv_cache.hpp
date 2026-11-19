// GPU (or CPU) storage for the paged KV cache.
//
// Per layer, K and V are each one tensor of shape
//   [num_blocks, block_size, num_kv_heads, head_dim]
// allocated once at startup. A token at absolute position p of a sequence
// lives in slot  block_table[p / block_size] * block_size + p % block_size.
#pragma once

#include <torch/torch.h>

#include <vector>

#include "block_manager.hpp"

namespace ps {

struct KVCache {
  int num_layers = 0;
  int num_blocks = 0;
  int block_size = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  std::vector<torch::Tensor> k;
  std::vector<torch::Tensor> v;

  KVCache(int layers, int blocks, int block_size, int kv_heads, int head_dim, torch::Device device,
          torch::ScalarType dtype);

  static int64_t bytes_per_block(int layers, int block_size, int kv_heads, int head_dim, torch::ScalarType dtype);

  // Executes copy-on-write block copies (all layers, K and V).
  void copy_blocks(const std::vector<BlockCopy>& copies);
};

}  // namespace ps

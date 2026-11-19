#include "kv_cache.hpp"

namespace ps {

KVCache::KVCache(int layers, int blocks, int bs, int kv_heads, int hd, torch::Device device, torch::ScalarType dtype)
    : num_layers(layers), num_blocks(blocks), block_size(bs), num_kv_heads(kv_heads), head_dim(hd) {
  auto opts = torch::TensorOptions().device(device).dtype(dtype);
  k.reserve(layers);
  v.reserve(layers);
  for (int l = 0; l < layers; ++l) {
    k.push_back(torch::zeros({blocks, bs, kv_heads, hd}, opts));
    v.push_back(torch::zeros({blocks, bs, kv_heads, hd}, opts));
  }
}

int64_t KVCache::bytes_per_block(int layers, int bs, int kv_heads, int hd, torch::ScalarType dtype) {
  return 2LL * layers * bs * kv_heads * hd * static_cast<int64_t>(c10::elementSize(dtype));
}

void KVCache::copy_blocks(const std::vector<BlockCopy>& copies) {
  if (copies.empty()) return;
  std::vector<int64_t> src, dst;
  for (const auto& c : copies) {
    src.push_back(c.src);
    dst.push_back(c.dst);
  }
  auto dev = k[0].device();
  auto s = torch::tensor(src, torch::kInt64).to(dev);
  auto d = torch::tensor(dst, torch::kInt64).to(dev);
  for (int l = 0; l < num_layers; ++l) {
    k[l].index_copy_(0, d, k[l].index_select(0, s));
    v[l].index_copy_(0, d, v[l].index_select(0, s));
  }
}

}  // namespace ps

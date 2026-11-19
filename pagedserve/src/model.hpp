// Llama-family decoder-only transformer written directly against LibTorch.
//
// Why hand-written instead of loading a TorchScript export: an exported
// Hugging Face model expects one contiguous KV tensor per sequence. Paged
// attention needs the forward pass to read and write K/V through block
// tables, so attention (and therefore the model) is implemented here.
//
// Supports Llama 2/3.x, Mistral, Qwen2/2.5 (QKV bias), Qwen3 (QK-norm),
// grouped-query attention, tied embeddings and Llama-3 RoPE scaling.
#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

#include "attention.hpp"
#include "config.hpp"
#include "kv_cache.hpp"

namespace ps {

class Model {
 public:
  Model(const ModelConfig& cfg, const std::string& weights_dir, torch::Device device, torch::ScalarType dtype);

  // Runs one step over a ragged batch and returns float32 logits
  // [logits_indices.size(), vocab] for the requested rows only.
  torch::Tensor forward(const ModelInput& in, KVCache& cache) const;

  const ModelConfig& config() const { return cfg_; }
  torch::Device device() const { return device_; }
  torch::ScalarType dtype() const { return dtype_; }
  int64_t num_parameters() const { return num_params_; }

 private:
  struct Layer {
    torch::Tensor input_norm, post_norm;
    torch::Tensor qkv_w, qkv_b;  // fused [Hq*D + 2*Hkv*D, hidden]; bias optional (Qwen2)
    torch::Tensor q_norm, k_norm;  // per-head RMSNorm weights (Qwen3), optional
    torch::Tensor o_w;
    torch::Tensor gate_up_w;  // fused [2 * intermediate, hidden]
    torch::Tensor down_w;
  };

  torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& w) const;
  torch::Tensor apply_rope(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin) const;

  ModelConfig cfg_;
  torch::Device device_;
  torch::ScalarType dtype_;
  torch::Tensor embed_, final_norm_, lm_head_;
  std::vector<Layer> layers_;
  torch::Tensor rope_cos_, rope_sin_;  // [max_pos, D/2] float32
  int64_t num_params_ = 0;
};

}  // namespace ps

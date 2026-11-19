// Batched token sampling on the model's device.
#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

namespace ps {

struct SamplingParams {
  float temperature = 1.0f;  // 0 = greedy
  float top_p = 1.0f;
  int top_k = -1;            // <= 0 disables
  int max_tokens = 16;
  int n = 1;                 // parallel samples (choices) per request
  std::vector<std::string> stop;
  bool ignore_eos = false;   // benchmark helper: always generate max_tokens
};

// logits: [N, V] float32. Row i is sampled with *params[i]. Returns N token
// ids on the CPU. Greedy rows use argmax; the rest apply temperature, top-k,
// top-p and draw from the resulting distribution.
std::vector<int32_t> sample_tokens(const torch::Tensor& logits, const std::vector<const SamplingParams*>& params);

}  // namespace ps

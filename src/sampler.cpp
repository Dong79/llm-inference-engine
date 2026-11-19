#include "sampler.hpp"

#include <limits>

namespace ps {

std::vector<int32_t> sample_tokens(const torch::Tensor& logits, const std::vector<const SamplingParams*>& params) {
  torch::InferenceMode guard;
  const int64_t n = logits.size(0), vocab = logits.size(1);
  std::vector<int32_t> out(n);
  if (n == 0) return out;

  std::vector<float> temps(n), top_ps(n);
  std::vector<int64_t> top_ks(n);
  bool any_random = false, any_top_k = false, any_top_p = false;
  int64_t max_k = 0;
  for (int64_t i = 0; i < n; ++i) {
    const auto& p = *params[i];
    const bool greedy = p.temperature < 1e-5f;
    any_random |= !greedy;
    temps[i] = greedy ? 1.0f : p.temperature;
    top_ps[i] = p.top_p;
    top_ks[i] = (p.top_k > 0 && p.top_k < vocab) ? p.top_k : vocab;
    if (!greedy && top_ks[i] < vocab) {
      any_top_k = true;
      max_k = std::max(max_k, top_ks[i]);
    }
    any_top_p |= (!greedy && p.top_p < 1.0f);
  }

  auto greedy_ids = logits.argmax(-1);
  if (!any_random) {
    auto cpu = greedy_ids.to(torch::kCPU).contiguous();
    const int64_t* p = cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(p[i]);
    return out;
  }

  const auto dev = logits.device();
  const float neg_inf = -std::numeric_limits<float>::infinity();
  auto x = logits / torch::tensor(temps, torch::kFloat32).to(dev).unsqueeze(1);

  if (any_top_k) {
    // Threshold = k-th largest logit of each row (rows without top-k keep all).
    auto topv = std::get<0>(x.topk(max_k, -1));
    auto kk = torch::tensor(top_ks, torch::kInt64).to(dev).clamp_max(max_k) - 1;
    auto thr = topv.gather(1, kk.unsqueeze(1));
    auto no_k = torch::tensor(top_ks, torch::kInt64).to(dev).eq(vocab).unsqueeze(1);
    thr = thr.masked_fill(no_k, neg_inf);
    x = x.masked_fill(x < thr, neg_inf);
  }
  if (any_top_p) {
    // Keep the smallest set of tokens whose probability mass reaches top_p.
    auto [sorted, idx] = x.sort(-1, /*descending=*/true);
    auto probs = sorted.softmax(-1);
    auto cum = probs.cumsum(-1);
    auto remove = (cum - probs) > torch::tensor(top_ps, torch::kFloat32).to(dev).unsqueeze(1);
    sorted = sorted.masked_fill(remove, neg_inf);
    x = torch::empty_like(x).scatter_(1, idx, sorted);
  }
  auto sampled = torch::multinomial(x.softmax(-1), 1).squeeze(1);
  std::vector<uint8_t> greedy_mask(n);
  for (int64_t i = 0; i < n; ++i) greedy_mask[i] = params[i]->temperature < 1e-5f;
  auto gm = torch::tensor(std::vector<int64_t>(greedy_mask.begin(), greedy_mask.end())).to(dev).to(torch::kBool);
  auto ids = torch::where(gm, greedy_ids, sampled).to(torch::kCPU).contiguous();
  const int64_t* p = ids.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(p[i]);
  return out;
}

}  // namespace ps

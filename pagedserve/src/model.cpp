#include "model.hpp"

#include <cmath>
#include <stdexcept>

#include "safetensors.hpp"

namespace ps {

namespace {

torch::Tensor take(WeightMap& w, const std::string& name) {
  auto it = w.find(name);
  if (it == w.end()) throw std::runtime_error("missing weight: " + name);
  torch::Tensor t = it->second;
  w.erase(it);
  return t;
}

torch::Tensor take_opt(WeightMap& w, const std::string& name) {
  auto it = w.find(name);
  if (it == w.end()) return {};
  torch::Tensor t = it->second;
  w.erase(it);
  return t;
}

// Inverse frequencies, including Llama-3 / linear scaling.
std::vector<double> rope_inv_freq(const ModelConfig& c) {
  const int half = c.head_dim / 2;
  std::vector<double> f(half);
  for (int i = 0; i < half; ++i) f[i] = 1.0 / std::pow(c.rope_theta, (2.0 * i) / c.head_dim);
  const auto& rs = c.rope_scaling;
  if (rs.type == "linear") {
    for (auto& x : f) x /= rs.factor;
  } else if (rs.type == "llama3") {
    const double old_ctx = rs.original_max_position_embeddings;
    const double low_wavelen = old_ctx / rs.low_freq_factor;
    const double high_wavelen = old_ctx / rs.high_freq_factor;
    for (auto& x : f) {
      const double wavelen = 2.0 * M_PI / x;
      if (wavelen < high_wavelen) continue;  // high frequency: unchanged
      if (wavelen > low_wavelen) {
        x /= rs.factor;  // low frequency: fully scaled
      } else {
        const double smooth = (old_ctx / wavelen - rs.low_freq_factor) / (rs.high_freq_factor - rs.low_freq_factor);
        x = (1.0 - smooth) * x / rs.factor + smooth * x;
      }
    }
  }
  return f;
}

}  // namespace

Model::Model(const ModelConfig& cfg, const std::string& dir, torch::Device device, torch::ScalarType dtype)
    : cfg_(cfg), device_(device), dtype_(dtype) {
  torch::NoGradGuard no_grad;
  WeightMap w = load_safetensors_dir(dir, device, dtype);
  for (const auto& [k, t] : w) num_params_ += t.numel();

  embed_ = take(w, "model.embed_tokens.weight");
  final_norm_ = take(w, "model.norm.weight");
  lm_head_ = take_opt(w, "lm_head.weight");
  if (!lm_head_.defined()) {
    if (!cfg.tie_word_embeddings) throw std::runtime_error("missing lm_head.weight");
    lm_head_ = embed_;
  }
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    Layer L;
    L.input_norm = take(w, p + "input_layernorm.weight");
    L.post_norm = take(w, p + "post_attention_layernorm.weight");
    // Fuse Q, K, V into one matrix so the projection is a single GEMM.
    auto q = take(w, p + "self_attn.q_proj.weight");
    auto k = take(w, p + "self_attn.k_proj.weight");
    auto v = take(w, p + "self_attn.v_proj.weight");
    L.qkv_w = torch::cat({q, k, v}, 0).contiguous();
    auto qb = take_opt(w, p + "self_attn.q_proj.bias");
    if (qb.defined())
      L.qkv_b = torch::cat({qb, take(w, p + "self_attn.k_proj.bias"), take(w, p + "self_attn.v_proj.bias")}, 0);
    L.q_norm = take_opt(w, p + "self_attn.q_norm.weight");
    L.k_norm = take_opt(w, p + "self_attn.k_norm.weight");
    L.o_w = take(w, p + "self_attn.o_proj.weight");
    L.gate_up_w = torch::cat({take(w, p + "mlp.gate_proj.weight"), take(w, p + "mlp.up_proj.weight")}, 0).contiguous();
    L.down_w = take(w, p + "mlp.down_proj.weight");
    layers_.push_back(std::move(L));
  }
  for (const auto& [k, t] : w)
    if (k.find("rotary_emb") == std::string::npos) fprintf(stderr, "[model] warning: unused weight %s\n", k.c_str());

  // RoPE tables for every position up to the context limit.
  const auto inv = rope_inv_freq(cfg);
  auto inv_t = torch::tensor(inv, torch::kFloat64);
  auto t = torch::arange(cfg.max_position_embeddings, torch::kFloat64).unsqueeze(1);
  auto freqs = (t * inv_t.unsqueeze(0));
  rope_cos_ = freqs.cos().to(device, torch::kFloat32);
  rope_sin_ = freqs.sin().to(device, torch::kFloat32);
}

torch::Tensor Model::rms_norm(const torch::Tensor& x, const torch::Tensor& w) const {
  // Matches HF: normalise in float32, cast back, then scale.
  auto xf = x.to(torch::kFloat32);
  xf = xf * torch::rsqrt(xf.pow(2).mean(-1, /*keepdim=*/true) + cfg_.rms_norm_eps);
  return w * xf.to(x.scalar_type());
}

torch::Tensor Model::apply_rope(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin) const {
  // Non-interleaved ("rotate_half") layout used by HF Llama/Qwen checkpoints.
  const int64_t half = x.size(-1) / 2;
  auto xf = x.to(torch::kFloat32);
  auto x1 = xf.narrow(-1, 0, half), x2 = xf.narrow(-1, half, half);
  return torch::cat({x1 * cos - x2 * sin, x2 * cos + x1 * sin}, -1).to(x.scalar_type());
}

torch::Tensor Model::forward(const ModelInput& in, KVCache& cache) const {
  torch::InferenceMode guard;
  const int64_t T = in.num_tokens, hq = cfg_.num_heads, hkv = cfg_.num_kv_heads, d = cfg_.head_dim;
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));
  auto cos = rope_cos_.index_select(0, in.positions).unsqueeze(1);  // [T, 1, D/2]
  auto sin = rope_sin_.index_select(0, in.positions).unsqueeze(1);

  auto h = embed_.index_select(0, in.input_ids);  // [T, hidden]
  for (size_t l = 0; l < layers_.size(); ++l) {
    const Layer& L = layers_[l];
    auto x = rms_norm(h, L.input_norm);
    auto qkv = L.qkv_b.defined() ? torch::linear(x, L.qkv_w, L.qkv_b) : torch::linear(x, L.qkv_w);
    auto q = qkv.narrow(1, 0, hq * d).view({T, hq, d});
    auto k = qkv.narrow(1, hq * d, hkv * d).view({T, hkv, d});
    auto v = qkv.narrow(1, (hq + hkv) * d, hkv * d).view({T, hkv, d});
    if (L.q_norm.defined()) {
      q = rms_norm(q, L.q_norm);
      k = rms_norm(k, L.k_norm);
    }
    q = apply_rope(q, cos, sin);
    k = apply_rope(k, cos, sin);
    auto attn = paged_attention(q, k, v.contiguous(), cache.k[l], cache.v[l], in.attn, scale);
    h = h + torch::linear(attn.reshape({T, hq * d}), L.o_w);

    x = rms_norm(h, L.post_norm);
    auto gu = torch::linear(x, L.gate_up_w);
    const int64_t inter = gu.size(1) / 2;
    auto act = torch::silu(gu.narrow(1, 0, inter)) * gu.narrow(1, inter, inter);
    h = h + torch::linear(act, L.down_w);
  }
  // Only rows that will be sampled go through the norm + LM head.
  auto last = h.index_select(0, in.logits_indices);
  last = rms_norm(last, final_norm_);
  return torch::linear(last, lm_head_).to(torch::kFloat32);
}

}  // namespace ps

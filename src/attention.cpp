#include "attention.hpp"

#include <algorithm>
#include <stdexcept>

#ifdef PAGEDSERVE_WITH_CUDA
#include "cuda/paged_attention.hpp"
#endif

namespace ps {

bool cuda_kernel_available(int64_t head_dim) {
#ifdef PAGEDSERVE_WITH_CUDA
  return paged_attention_decode_cuda_supported(head_dim);
#else
  return false;
#endif
}

ModelInput build_model_input(std::span<const SeqChunk> chunks, int block_size, torch::Device device) {
  std::vector<int64_t> ids, pos, slots, logits_idx;
  std::vector<int32_t> bt_flat, ctx_lens;
  int64_t num_decode = 0;
  int max_blocks = 0;  // widest decode block table, for padding
  for (const auto& c : chunks)
    if (c.len == 1) max_blocks = std::max(max_blocks, (c.start + 1 + block_size - 1) / block_size);

  ModelInput in;
  std::vector<PrefillMeta> prefills;
  for (const auto& c : chunks) {
    const int64_t token_start = static_cast<int64_t>(ids.size());
    const int ctx = c.start + c.len;
    if (static_cast<int>(c.block_table.size()) * block_size < ctx)
      throw std::logic_error("block table does not cover the chunk");
    for (int i = 0; i < c.len; ++i) {
      const int p = c.start + i;
      ids.push_back(c.tokens[p]);
      pos.push_back(p);
      slots.push_back(static_cast<int64_t>(c.block_table[p / block_size]) * block_size + p % block_size);
      if (c.logits_all) logits_idx.push_back(token_start + i);
    }
    if (c.logits_last && !c.logits_all) logits_idx.push_back(token_start + c.len - 1);
    if (c.len == 1) {
      if (num_decode != token_start) throw std::logic_error("decode chunks must precede prefill chunks");
      ++num_decode;
      for (int b = 0; b < max_blocks; ++b)
        bt_flat.push_back(b < static_cast<int>(c.block_table.size()) ? c.block_table[b] : 0);
      ctx_lens.push_back(ctx);
      in.attn.decode_max_context = std::max<int64_t>(in.attn.decode_max_context, ctx);
    } else {
      const int nb = (ctx + block_size - 1) / block_size;
      std::vector<int64_t> blocks(c.block_table.begin(), c.block_table.begin() + nb);
      prefills.push_back({token_start, c.len, ctx, torch::tensor(blocks, torch::kInt64).to(device)});
    }
  }
  auto to_dev = [&](const std::vector<int64_t>& v) { return torch::tensor(v, torch::kInt64).to(device); };
  in.num_tokens = static_cast<int64_t>(ids.size());
  in.input_ids = to_dev(ids);
  in.positions = to_dev(pos);
  in.logits_indices = to_dev(logits_idx);
  in.attn.slot_mapping = to_dev(slots);
  in.attn.num_decode = num_decode;
  if (num_decode > 0) {
    in.attn.decode_block_tables = torch::tensor(bt_flat, torch::kInt32).view({num_decode, max_blocks}).to(device);
    in.attn.decode_context_lens = torch::tensor(ctx_lens, torch::kInt32).to(device);
  }
  in.attn.prefills = std::move(prefills);
  return in;
}

namespace {

// SDPA with grouped-query attention: q [B, Hq, Lq, D], k/v [B, Hkv, Lk, D].
torch::Tensor sdpa_gqa(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                       const std::optional<torch::Tensor>& mask, bool causal, double scale) {
  return at::scaled_dot_product_attention(q, k, v, mask, 0.0, causal, scale, /*enable_gqa=*/q.size(1) != k.size(1));
}

}  // namespace

torch::Tensor paged_decode_reference(const torch::Tensor& q, const torch::Tensor& k_cache,
                                     const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                                     const torch::Tensor& context_lens, double scale) {
  const int64_t nd = q.size(0), hkv = k_cache.size(2), d = k_cache.size(3), bs = k_cache.size(1);
  const int64_t mb = block_tables.size(1);
  auto bt = block_tables.to(torch::kInt64).flatten();
  // [nd, Hkv, mb * bs, D]: every block of every sequence gathered contiguously.
  auto kd = k_cache.index_select(0, bt).view({nd, mb * bs, hkv, d}).transpose(1, 2);
  auto vd = v_cache.index_select(0, bt).view({nd, mb * bs, hkv, d}).transpose(1, 2);
  auto ar = torch::arange(mb * bs, torch::TensorOptions().device(q.device()).dtype(torch::kInt64));
  auto mask = (ar.unsqueeze(0) < context_lens.to(torch::kInt64).unsqueeze(1)).view({nd, 1, 1, mb * bs});
  auto o = sdpa_gqa(q.unsqueeze(2), kd, vd, mask, false, scale);  // [nd, Hq, 1, D]
  return o.squeeze(2);
}

torch::Tensor paged_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                              torch::Tensor& k_cache, torch::Tensor& v_cache, const AttentionMetadata& meta,
                              double scale) {
  const int64_t hkv = k.size(1), d = k.size(2);
  // 1. Write this step's K/V into their slots.
  k_cache.view({-1, hkv, d}).index_copy_(0, meta.slot_mapping, k);
  v_cache.view({-1, hkv, d}).index_copy_(0, meta.slot_mapping, v);

  auto out = torch::empty_like(q);
  // 2. Single-token queries, batched.
  if (meta.num_decode > 0) {
    auto qd = q.narrow(0, 0, meta.num_decode);
    torch::Tensor od;
#ifdef PAGEDSERVE_WITH_CUDA
    if (meta.use_cuda_kernel && q.is_cuda() && paged_attention_decode_cuda_supported(q.size(2)))
      od = paged_attention_decode_cuda(qd, k_cache, v_cache, meta.decode_block_tables, meta.decode_context_lens,
                                       meta.decode_max_context, scale);
    else
#endif
      od = paged_decode_reference(qd, k_cache, v_cache, meta.decode_block_tables, meta.decode_context_lens, scale);
    out.narrow(0, 0, meta.num_decode).copy_(od);
  }
  // 3. Prefill chunks: queries attend to [cached prefix | new tokens] causally.
  for (const auto& p : meta.prefills) {
    auto qp = q.narrow(0, p.token_start, p.q_len).transpose(0, 1).unsqueeze(0);  // [1, Hq, q, D]
    auto kp = k_cache.index_select(0, p.block_ids).view({-1, hkv, d}).narrow(0, 0, p.context_len);
    auto vp = v_cache.index_select(0, p.block_ids).view({-1, hkv, d}).narrow(0, 0, p.context_len);
    kp = kp.transpose(0, 1).unsqueeze(0);  // [1, Hkv, ctx, D]
    vp = vp.transpose(0, 1).unsqueeze(0);
    torch::Tensor o;
    if (p.context_len == p.q_len) {
      o = sdpa_gqa(qp, kp, vp, std::nullopt, /*causal=*/true, scale);
    } else {
      // Query i sits at absolute position (ctx - q + i) and may see keys <= that.
      auto opts = torch::TensorOptions().device(q.device()).dtype(torch::kInt64);
      auto qi = torch::arange(p.q_len, opts).unsqueeze(1) + (p.context_len - p.q_len);
      auto kj = torch::arange(p.context_len, opts).unsqueeze(0);
      o = sdpa_gqa(qp, kp, vp, kj <= qi, false, scale);
    }
    out.narrow(0, p.token_start, p.q_len).copy_(o.squeeze(0).transpose(0, 1));
  }
  return out;
}

}  // namespace ps

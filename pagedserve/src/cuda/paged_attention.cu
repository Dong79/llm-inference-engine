// Paged-attention decode kernel.
//
// Grid:  (num_heads, num_seqs). One thread block computes one query head of
//        one sequence.
// Block: NUM_WARPS warps. Warp w handles tokens w, w + NUM_WARPS, ... of the
//        sequence. Inside a warp, lane l owns dims l, l + 32, ... of the head,
//        so loads of a K/V row are coalesced across the warp.
//
// Each warp keeps a running max m, running sum l and an output accumulator
// (online softmax), so scores never need to be materialised and context
// length is unbounded by shared memory. At the end the warps' partial results
// are merged:  out = sum_w exp(m_w - M) * acc_w / sum_w exp(m_w - M) * l_w.
//
// Physical location of token t:  block_tables[seq][t / block_size],
// offset t % block_size, so K/V never have to be contiguous.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "paged_attention.hpp"

namespace ps {

namespace {

constexpr int kWarpSize = 32;
constexpr int kNumWarps = 4;

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(c10::Half x) { return __half2float(static_cast<__half>(x)); }
__device__ __forceinline__ float to_float(c10::BFloat16 x) { return static_cast<float>(x); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_decode_kernel(scalar_t* __restrict__ out,           // [S, H, D]
                                              const scalar_t* __restrict__ q,       // [S, H, D]
                                              const scalar_t* __restrict__ k_cache, // [B, bs, Hkv, D]
                                              const scalar_t* __restrict__ v_cache,
                                              const int* __restrict__ block_tables,  // [S, max_blocks]
                                              const int* __restrict__ context_lens,  // [S]
                                              int num_heads, int num_kv_heads, int block_size, int max_blocks,
                                              float scale) {
  constexpr int DIMS_PER_LANE = HEAD_DIM / kWarpSize;
  const int head = blockIdx.x;
  const int seq = blockIdx.y;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int kv_head = head / (num_heads / num_kv_heads);  // grouped-query attention
  const int ctx = context_lens[seq];
  const int* table = block_tables + static_cast<int64_t>(seq) * max_blocks;

  // This lane's slice of the (pre-scaled) query.
  float qv[DIMS_PER_LANE];
  const scalar_t* qp = q + (static_cast<int64_t>(seq) * num_heads + head) * HEAD_DIM;
#pragma unroll
  for (int i = 0; i < DIMS_PER_LANE; ++i) qv[i] = to_float(qp[lane + i * kWarpSize]) * scale;

  float m = -INFINITY, l = 0.f;
  float acc[DIMS_PER_LANE];
#pragma unroll
  for (int i = 0; i < DIMS_PER_LANE; ++i) acc[i] = 0.f;

  const int64_t token_stride = static_cast<int64_t>(num_kv_heads) * HEAD_DIM;
  for (int t = warp; t < ctx; t += kNumWarps) {
    const int64_t slot = static_cast<int64_t>(table[t / block_size]) * block_size + (t % block_size);
    const int64_t base = slot * token_stride + static_cast<int64_t>(kv_head) * HEAD_DIM;
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) dot += qv[i] * to_float(k_cache[base + lane + i * kWarpSize]);
    const float s = warp_sum(dot);  // every lane now holds the score
    const float m_new = fmaxf(m, s);
    const float correction = __expf(m - m_new);
    const float p = __expf(s - m_new);
    l = l * correction + p;
#pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i)
      acc[i] = acc[i] * correction + p * to_float(v_cache[base + lane + i * kWarpSize]);
    m = m_new;
  }

  // Merge the warps' partial softmax states.
  __shared__ float sm_m[kNumWarps], sm_l[kNumWarps];
  __shared__ float sm_acc[kNumWarps][HEAD_DIM];
  if (lane == 0) {
    sm_m[warp] = m;
    sm_l[warp] = l;
  }
#pragma unroll
  for (int i = 0; i < DIMS_PER_LANE; ++i) sm_acc[warp][lane + i * kWarpSize] = acc[i];
  __syncthreads();

  float M = -INFINITY;
#pragma unroll
  for (int w = 0; w < kNumWarps; ++w) M = fmaxf(M, sm_m[w]);
  float L = 0.f;
  float wscale[kNumWarps];
#pragma unroll
  for (int w = 0; w < kNumWarps; ++w) {
    // A warp that saw no tokens (ctx < kNumWarps) has m = -inf: weight 0.
    wscale[w] = sm_m[w] == -INFINITY ? 0.f : __expf(sm_m[w] - M);
    L += wscale[w] * sm_l[w];
  }
  const float inv_L = 1.f / L;
  scalar_t* op = out + (static_cast<int64_t>(seq) * num_heads + head) * HEAD_DIM;
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
    float o = 0.f;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) o += wscale[w] * sm_acc[w][d];
    op[d] = static_cast<scalar_t>(o * inv_L);
  }
}

template <typename scalar_t>
void launch(const torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& bt, const torch::Tensor& cl, float scale) {
  const int num_seqs = q.size(0), num_heads = q.size(1), head_dim = q.size(2);
  const int block_size = k.size(1), num_kv_heads = k.size(2), max_blocks = bt.size(1);
  dim3 grid(num_heads, num_seqs);
  dim3 block(kNumWarps * kWarpSize);
  auto stream = at::cuda::getCurrentCUDAStream();
#define PS_LAUNCH(D)                                                                                       \
  paged_attention_decode_kernel<scalar_t, D><<<grid, block, 0, stream>>>(                                  \
      out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),    \
      bt.data_ptr<int>(), cl.data_ptr<int>(), num_heads, num_kv_heads, block_size, max_blocks, scale)
  switch (head_dim) {
    case 32: PS_LAUNCH(32); break;
    case 64: PS_LAUNCH(64); break;
    case 96: PS_LAUNCH(96); break;
    case 128: PS_LAUNCH(128); break;
    case 256: PS_LAUNCH(256); break;
    default: TORCH_CHECK(false, "paged_attention_decode_cuda: unsupported head_dim ", head_dim);
  }
#undef PS_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool paged_attention_decode_cuda_supported(int64_t d) { return d == 32 || d == 64 || d == 96 || d == 128 || d == 256; }

torch::Tensor paged_attention_decode_cuda(const torch::Tensor& q_in, const torch::Tensor& k_cache,
                                          const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                                          const torch::Tensor& context_lens, int64_t /*max_context*/, double scale) {
  TORCH_CHECK(q_in.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "tensors must be on CUDA");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32 && context_lens.scalar_type() == torch::kInt32,
              "block_tables/context_lens must be int32");
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(), "KV cache must be contiguous");
  TORCH_CHECK(q_in.scalar_type() == k_cache.scalar_type(), "q and cache dtypes differ");
  const at::cuda::OptionalCUDAGuard guard(q_in.device());
  auto q = q_in.contiguous();
  auto bt = block_tables.contiguous();
  auto cl = context_lens.contiguous();
  auto out = torch::empty_like(q);
  const float s = static_cast<float>(scale);
  switch (q.scalar_type()) {
    case torch::kFloat32: launch<float>(out, q, k_cache, v_cache, bt, cl, s); break;
    case torch::kFloat16: launch<c10::Half>(out, q, k_cache, v_cache, bt, cl, s); break;
    case torch::kBFloat16: launch<c10::BFloat16>(out, q, k_cache, v_cache, bt, cl, s); break;
    default: TORCH_CHECK(false, "unsupported dtype");
  }
  return out;
}

}  // namespace ps

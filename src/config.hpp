// Model and engine configuration.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace ps {

struct RopeScaling {
  std::string type = "default";  // "default" | "linear" | "llama3"
  double factor = 1.0;
  double low_freq_factor = 1.0;
  double high_freq_factor = 4.0;
  int original_max_position_embeddings = 8192;
};

// Decoder-only Llama-family transformer (Llama 2/3, Mistral, Qwen2/2.5/3, SmolLM2).
struct ModelConfig {
  std::string architecture;
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  double rms_norm_eps = 1e-6;
  double rope_theta = 10000.0;
  RopeScaling rope_scaling;
  int max_position_embeddings = 4096;
  bool tie_word_embeddings = false;
  std::vector<int> eos_token_ids;  // config.json + generation_config.json
  std::string dtype;               // checkpoint dtype, e.g. "bfloat16"

  // Reads <dir>/config.json and <dir>/generation_config.json (optional).
  static ModelConfig from_dir(const std::string& dir);
};

struct SchedulerConfig {
  int max_num_seqs = 128;             // max sequences in a batch
  int max_num_batched_tokens = 2048;  // per-step token budget (prefill + decode)
  bool static_batching = false;       // baseline: admit a batch, run it to completion
  bool chunked_prefill = true;        // split long prompts across steps
  double watermark = 0.01;            // fraction of KV blocks kept free at admission
};

struct EngineConfig {
  std::string model_dir;
  std::string device = "auto";  // "auto" | "cpu" | "cuda" | "cuda:N"
  std::string dtype = "auto";   // "auto" | "float32" | "float16" | "bfloat16"
  int block_size = 16;
  int num_kv_blocks = 0;        // 0 = derive from kv_cache_memory_gb / GPU memory
  double kv_cache_memory_gb = 0;  // 0 = auto
  double gpu_memory_utilization = 0.90;
  int max_model_len = 0;          // 0 = model's max_position_embeddings (capped at 32768)
  bool enable_prefix_caching = true;
  bool use_cuda_kernel = true;    // custom paged-attention decode kernel, if built
  std::string chat_template = "auto";  // "auto" | "chatml" | "llama3" | "plain"
  std::string served_model_name;
  SchedulerConfig sched;
};

}  // namespace ps

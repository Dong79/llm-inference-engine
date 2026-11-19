// Tiny command-line parser shared by the server and the dev tool.
#pragma once

#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"

namespace ps {

class Args {
 public:
  Args(int argc, char** argv, int first = 1) {
    for (int i = first; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) {
        positional_.push_back(a);
        continue;
      }
      a = a.substr(2);
      if (auto eq = a.find('='); eq != std::string::npos) {
        kv_[a.substr(0, eq)] = a.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv_[a] = argv[++i];
      } else {
        kv_[a] = "true";
      }
    }
  }
  bool has(const std::string& k) const { return kv_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def = "") {
    used_.insert(k);
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  int i(const std::string& k, int def) { return has(k) ? std::stoi(str(k)) : (used_.insert(k), def); }
  double d(const std::string& k, double def) { return has(k) ? std::stod(str(k)) : (used_.insert(k), def); }
  bool flag(const std::string& k) {
    used_.insert(k);
    auto it = kv_.find(k);
    return it != kv_.end() && it->second != "false" && it->second != "0";
  }
  const std::vector<std::string>& positional() const { return positional_; }
  // Throws on options nobody asked for (catches typos like --max-num-seq).
  void check_unused() const {
    for (const auto& [k, v] : kv_)
      if (!used_.count(k)) throw std::invalid_argument("unknown option --" + k);
  }

 private:
  std::map<std::string, std::string> kv_;
  std::set<std::string> used_;
  std::vector<std::string> positional_;
};

inline const char* kEngineHelp = R"(engine options:
  --model DIR                 Hugging Face model directory (config.json, *.safetensors, tokenizer.json)
  --device auto|cpu|cuda[:N]  default auto
  --dtype auto|float32|float16|bfloat16
  --block-size N              tokens per KV block (default 16)
  --num-kv-blocks N           KV blocks (default: derived from memory)
  --kv-cache-gb X             KV cache size in GB (default: GPU free memory, or 1 GB on CPU)
  --gpu-memory-utilization X  fraction of GPU memory to use (default 0.90)
  --max-model-len N           max prompt + output tokens per sequence
  --max-num-seqs N            max sequences per batch (default 128)
  --max-num-batched-tokens N  per-step token budget (default 2048)
  --static-batching           baseline: no admissions until the whole batch finishes
  --no-chunked-prefill        a prompt must fit in one step's token budget
  --no-prefix-caching         disable hash-based prefix sharing
  --no-cuda-kernel            use the LibTorch reference attention on GPU too
  --chat-template auto|chatml|llama3|plain
  --served-model-name NAME
)";

inline EngineConfig parse_engine_args(Args& a) {
  EngineConfig c;
  c.model_dir = a.str("model");
  if (c.model_dir.empty()) throw std::invalid_argument("--model is required");
  c.device = a.str("device", "auto");
  c.dtype = a.str("dtype", "auto");
  c.block_size = a.i("block-size", 16);
  c.num_kv_blocks = a.i("num-kv-blocks", 0);
  c.kv_cache_memory_gb = a.d("kv-cache-gb", 0);
  c.gpu_memory_utilization = a.d("gpu-memory-utilization", 0.90);
  c.max_model_len = a.i("max-model-len", 0);
  c.sched.max_num_seqs = a.i("max-num-seqs", 128);
  c.sched.max_num_batched_tokens = a.i("max-num-batched-tokens", 2048);
  c.sched.static_batching = a.flag("static-batching");
  c.sched.chunked_prefill = !a.flag("no-chunked-prefill");
  c.enable_prefix_caching = !a.flag("no-prefix-caching");
  c.use_cuda_kernel = !a.flag("no-cuda-kernel");
  c.chat_template = a.str("chat-template", "auto");
  c.served_model_name = a.str("served-model-name", "");
  return c;
}

}  // namespace ps

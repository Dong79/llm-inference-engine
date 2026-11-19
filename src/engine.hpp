// The inference engine: owns the model, KV cache, block manager and
// scheduler, and runs the step loop on a single dedicated thread.
//
// HTTP threads call add_request()/abort_request(), which only touch a small
// mutex-protected inbox. Everything else (scheduler, block manager, model,
// sequences) is owned by the engine thread, so the hot path needs no locks
// and a slow client can never stall the GPU loop.
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "block_manager.hpp"
#include "chat_template.hpp"
#include "config.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "scheduler.hpp"
#include "sequence.hpp"
#include "tokenizer.hpp"

namespace ps {

struct EngineStats {
  int num_running = 0;
  int num_waiting = 0;
  int kv_blocks_total = 0;
  int kv_blocks_free = 0;     // never-used or fully released
  int kv_blocks_cached = 0;   // cached prefixes, reclaimable
  uint64_t prefix_query_tokens = 0;
  uint64_t prefix_hit_tokens = 0;
  uint64_t num_preemptions = 0;
  uint64_t num_cow_copies = 0;
  uint64_t num_evictions = 0;
  uint64_t steps = 0;
  uint64_t prefill_tokens = 0;    // prompt tokens actually computed
  uint64_t generated_tokens = 0;
  uint64_t requests_finished = 0;
  double busy_seconds = 0;        // time spent inside step()
};

class Engine {
 public:
  explicit Engine(const EngineConfig& cfg);
  ~Engine();

  // Starts the background step loop. Without it, call step() yourself.
  void start();
  void stop();

  // Thread-safe.
  void add_request(std::shared_ptr<Request> req);
  void abort_request(const std::string& request_id);
  EngineStats stats() const;

  // Runs one scheduling + forward + sampling iteration. Returns the number of
  // tokens processed (0 when idle). Engine-thread only.
  int step();
  bool has_work();

  const Tokenizer& tokenizer() const { return *tok_; }
  const ChatTemplate& chat_template() const { return chat_; }
  const ModelConfig& model_config() const { return mcfg_; }
  int max_model_len() const { return max_model_len_; }
  const std::string& model_name() const { return model_name_; }
  std::string describe() const;

 private:
  void loop(std::stop_token st);
  void drain_inbox();
  void process_token(Sequence* s, TokenId t);
  void finish_seq(Sequence* s);
  void register_full_blocks(Sequence* s);
  void fail_all(const std::string& why);
  void publish_stats(double step_seconds, int prefill_tokens, int generated, bool counted);

  EngineConfig cfg_;
  torch::Device device_{torch::kCPU};
  torch::ScalarType dtype_ = torch::kFloat32;
  ModelConfig mcfg_;
  std::unique_ptr<Tokenizer> tok_;
  ChatTemplate chat_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<KVCache> kv_;
  std::unique_ptr<BlockManager> bm_;
  std::unique_ptr<Scheduler> sched_;
  std::unordered_set<TokenId> eos_ids_;
  int max_model_len_ = 0;
  std::string model_name_;
  bool cuda_kernel_on_ = false;

  // Engine-thread state.
  std::unordered_map<uint64_t, std::unique_ptr<Sequence>> seqs_;
  std::unordered_map<std::string, std::vector<Sequence*>> req_seqs_;
  uint64_t next_seq_id_ = 1;

  // Inbox shared with HTTP threads.
  std::mutex mu_;
  std::condition_variable_any cv_;
  std::vector<std::shared_ptr<Request>> inbox_;
  std::vector<std::string> aborts_;

  mutable std::mutex stats_mu_;
  EngineStats stats_;
  Clock::time_point last_log_ = Clock::now();
  uint64_t gen_at_last_log_ = 0;

  std::jthread thread_;
};

}  // namespace ps

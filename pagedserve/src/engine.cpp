#include "engine.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "attention.hpp"
#include "sampler.hpp"
#include "utf8.hpp"

#ifdef PAGEDSERVE_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace ps {

namespace {

torch::Device pick_device(const std::string& name) {
  if (name == "auto") return torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
  return torch::Device(name);
}

torch::ScalarType pick_dtype(const std::string& name, const ModelConfig& mc, torch::Device dev) {
  if (name == "float32") return torch::kFloat32;
  if (name == "float16") return torch::kFloat16;
  if (name == "bfloat16") return torch::kBFloat16;
  if (name != "auto") throw std::runtime_error("unknown dtype " + name);
  if (dev.is_cpu()) return torch::kFloat32;
  return mc.dtype == "float16" ? torch::kFloat16 : torch::kBFloat16;
}

const char* dtype_name(torch::ScalarType t) {
  switch (t) {
    case torch::kFloat32: return "float32";
    case torch::kFloat16: return "float16";
    case torch::kBFloat16: return "bfloat16";
    default: return "other";
  }
}

}  // namespace

Engine::Engine(const EngineConfig& cfg) : cfg_(cfg) {
  mcfg_ = ModelConfig::from_dir(cfg.model_dir);
  device_ = pick_device(cfg.device);
  dtype_ = pick_dtype(cfg.dtype, mcfg_, device_);
  tok_ = Tokenizer::from_dir(cfg.model_dir);
  chat_ = ChatTemplate::create(cfg.chat_template, *tok_);
  model_name_ = cfg.served_model_name.empty() ? cfg.model_dir : cfg.served_model_name;

  for (int id : mcfg_.eos_token_ids) eos_ids_.insert(id);
  if (auto eot = chat_.end_of_turn())
    if (auto id = tok_->token_to_id(*eot)) eos_ids_.insert(*id);

  model_ = std::make_unique<Model>(mcfg_, cfg.model_dir, device_, dtype_);

  // Size the KV cache.
  const int64_t per_block =
      KVCache::bytes_per_block(mcfg_.num_layers, cfg.block_size, mcfg_.num_kv_heads, mcfg_.head_dim, dtype_);
  int64_t num_blocks = cfg.num_kv_blocks;
  if (num_blocks <= 0) {
    double gb = cfg.kv_cache_memory_gb;
    if (gb <= 0) {
      if (device_.is_cpu()) {
        gb = 1.0;
      } else {
#ifdef PAGEDSERVE_WITH_CUDA
        // Weights are loaded; give the KV cache what is left of the budget,
        // keeping headroom for activations of a full token budget.
        size_t free_b = 0, total_b = 0;
        cudaSetDevice(device_.index());
        cudaMemGetInfo(&free_b, &total_b);
        const double used = static_cast<double>(total_b - free_b);
        const double headroom = 1.5e9;
        gb = (cfg.gpu_memory_utilization * static_cast<double>(total_b) - used - headroom) / 1e9;
        if (gb < 0.25) throw std::runtime_error("not enough GPU memory left for the KV cache");
#else
        gb = 4.0;
        fprintf(stderr, "[engine] CUDA kernels not built; defaulting to a 4 GB KV cache (--kv-cache-gb to change)\n");
#endif
      }
    }
    num_blocks = static_cast<int64_t>(gb * 1e9 / static_cast<double>(per_block));
  }
  if (num_blocks < 4) throw std::runtime_error("KV cache too small");
  kv_ = std::make_unique<KVCache>(mcfg_.num_layers, static_cast<int>(num_blocks), cfg.block_size, mcfg_.num_kv_heads,
                                  mcfg_.head_dim, device_, dtype_);
  bm_ = std::make_unique<BlockManager>(static_cast<int>(num_blocks), cfg.block_size, cfg.enable_prefix_caching);
  sched_ = std::make_unique<Scheduler>(cfg.sched, *bm_);

  // A single sequence must always fit in the cache, or it could never run.
  max_model_len_ = cfg.max_model_len > 0 ? cfg.max_model_len : std::min(mcfg_.max_position_embeddings, 32768);
  max_model_len_ = std::min(max_model_len_, mcfg_.max_position_embeddings);
  const int watermark = std::max(1, static_cast<int>(cfg.sched.watermark * num_blocks));
  max_model_len_ = std::min<int64_t>(max_model_len_, (num_blocks - watermark - 1) * cfg.block_size);
  cuda_kernel_on_ = cfg.use_cuda_kernel && cuda_kernel_available(mcfg_.head_dim) && device_.is_cuda();
  stats_.kv_blocks_total = static_cast<int>(num_blocks);
}

Engine::~Engine() { stop(); }

std::string Engine::describe() const {
  std::ostringstream os;
  const double kv_gb = static_cast<double>(KVCache::bytes_per_block(mcfg_.num_layers, cfg_.block_size,
                                                                    mcfg_.num_kv_heads, mcfg_.head_dim, dtype_)) *
                       bm_->num_blocks() / 1e9;
  const int64_t kv_per_token = KVCache::bytes_per_block(mcfg_.num_layers, 1, mcfg_.num_kv_heads, mcfg_.head_dim, dtype_);
  os << "model        " << mcfg_.architecture << ", " << model_->num_parameters() / 1e6 << "M params, "
     << mcfg_.num_layers << " layers, " << mcfg_.num_heads << "/" << mcfg_.num_kv_heads << " q/kv heads, head_dim "
     << mcfg_.head_dim << "\n"
     << "device       " << device_.str() << ", " << dtype_name(dtype_) << "\n"
     << "kv cache     " << bm_->num_blocks() << " blocks x " << cfg_.block_size << " tokens = " << kv_gb << " GB ("
     << kv_per_token << " bytes/token)\n"
     << "max len      " << max_model_len_ << " tokens\n"
     << "scheduler    " << (cfg_.sched.static_batching ? "static batching (baseline)" : "continuous batching")
     << ", max_num_seqs " << cfg_.sched.max_num_seqs << ", token budget " << cfg_.sched.max_num_batched_tokens
     << (cfg_.sched.chunked_prefill ? ", chunked prefill" : "") << "\n"
     << "prefix cache " << (cfg_.enable_prefix_caching ? "on" : "off") << "\n"
     << "attention    " << (cuda_kernel_on_ ? "CUDA paged-attention kernel (decode)" : "LibTorch gather + SDPA") << "\n"
     << "chat format  " << chat_.name() << "\n";
  return os.str();
}

void Engine::start() {
  thread_ = std::jthread([this](std::stop_token st) { loop(st); });
}

void Engine::stop() {
  if (thread_.joinable()) {
    thread_.request_stop();
    cv_.notify_all();
    thread_.join();
  }
}

void Engine::add_request(std::shared_ptr<Request> req) {
  if (req->prompt.empty()) throw std::invalid_argument("empty prompt");
  if (static_cast<int>(req->prompt.size()) >= max_model_len_) throw std::invalid_argument("prompt too long");
  {
    std::lock_guard<std::mutex> lk(mu_);
    inbox_.push_back(std::move(req));
  }
  cv_.notify_one();
}

void Engine::abort_request(const std::string& id) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    aborts_.push_back(id);
  }
  cv_.notify_one();
}

EngineStats Engine::stats() const {
  std::lock_guard<std::mutex> lk(stats_mu_);
  return stats_;
}

bool Engine::has_work() {
  if (sched_->has_work()) return true;
  std::lock_guard<std::mutex> lk(mu_);
  return !inbox_.empty() || !aborts_.empty();
}

void Engine::loop(std::stop_token st) {
  while (!st.stop_requested()) {
    if (!sched_->has_work()) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, st, [&] { return !inbox_.empty() || !aborts_.empty(); });
      if (st.stop_requested()) break;
    }
    try {
      step();
    } catch (const std::exception& e) {
      fprintf(stderr, "[engine] step failed: %s\n", e.what());
      fail_all(e.what());
    }
  }
}

void Engine::drain_inbox() {
  std::vector<std::shared_ptr<Request>> reqs;
  std::vector<std::string> aborts;
  {
    std::lock_guard<std::mutex> lk(mu_);
    reqs.swap(inbox_);
    aborts.swap(aborts_);
  }
  for (auto& r : reqs) {
    auto s = std::make_unique<Sequence>();
    s->id = next_seq_id_++;
    s->req = r;
    s->tokens = r->prompt;
    s->prompt_len = static_cast<int>(r->prompt.size());
    r->num_live_seqs = 1;
    r->usage.prompt_tokens = s->prompt_len;
    req_seqs_[r->id].push_back(s.get());
    sched_->add_waiting(s.get());
    seqs_.emplace(s->id, std::move(s));
  }
  for (const auto& id : aborts) {
    auto it = req_seqs_.find(id);
    if (it == req_seqs_.end()) continue;
    auto seqs = it->second;  // copy: finish_seq edits the map
    for (Sequence* s : seqs) finish_seq(s);
  }
}

void Engine::register_full_blocks(Sequence* s) {
  if (!bm_->prefix_caching()) return;
  const int bs = bm_->block_size();
  const size_t full = static_cast<size_t>(s->num_computed / bs);
  while (s->block_hashes.size() < full) {
    const size_t i = s->block_hashes.size();
    const uint64_t parent = i == 0 ? BlockManager::kRootKey : s->block_hashes[i - 1];
    std::span<const TokenId> chunk(s->tokens.data() + i * bs, bs);
    const uint64_t key = BlockManager::hash_block(parent, chunk);
    bm_->register_full_block(s->block_table[i], key, parent, chunk);
    s->block_hashes.push_back(key);
  }
}

int Engine::step() {
  const auto t0 = Clock::now();
  drain_inbox();
  SchedulerOutput out = sched_->schedule();
  if (out.empty()) {
    publish_stats(0, 0, 0, /*counted=*/false);  // aborts may have changed state
    return 0;
  }

  kv_->copy_blocks(out.copies);

  // Build the ragged batch. Rows that finish their pending tokens get logits.
  std::vector<SeqChunk> chunks;
  std::vector<Sequence*> to_sample;
  int prefill_tokens = 0;
  for (const auto& c : out.chunks) {
    Sequence* s = c.seq;
    const bool last = s->num_computed + c.num_tokens == static_cast<int>(s->tokens.size());
    chunks.push_back({s->tokens, s->num_computed, c.num_tokens, s->block_table, last, false});
    if (last) to_sample.push_back(s);
    prefill_tokens += std::max(0, std::min(s->num_computed + c.num_tokens, s->prompt_len) - s->num_computed);
  }
  ModelInput input = build_model_input(chunks, bm_->block_size(), device_);
  input.attn.use_cuda_kernel = cuda_kernel_on_;
  torch::Tensor logits = model_->forward(input, *kv_);

  for (const auto& c : out.chunks) {
    c.seq->num_computed += c.num_tokens;
    register_full_blocks(c.seq);
  }

  // Sampling. A request with n > 1 forks after its prompt: the parent's row
  // is sampled n times and the children share the parent's blocks.
  std::vector<int64_t> rows;
  std::vector<const SamplingParams*> params;
  std::vector<int> fan_out;
  for (size_t i = 0; i < to_sample.size(); ++i) {
    Sequence* s = to_sample[i];
    const auto& req = *s->req;
    const int copies = (!req.forked && s->index == 0 && s->num_generated() == 0) ? req.params.n : 1;
    fan_out.push_back(copies);
    for (int k = 0; k < copies; ++k) {
      rows.push_back(static_cast<int64_t>(i));
      params.push_back(&req.params);
    }
  }
  if (rows.size() != to_sample.size())
    logits = logits.index_select(0, torch::tensor(rows, torch::kInt64).to(logits.device()));
  const std::vector<TokenId> sampled = sample_tokens(logits, params);

  int generated = 0;
  size_t r = 0;
  for (size_t i = 0; i < to_sample.size(); ++i) {
    Sequence* s = to_sample[i];
    if (fan_out[i] > 1) {
      auto& req = *s->req;
      req.forked = true;
      for (int k = 1; k < fan_out[i]; ++k) {
        auto child = std::make_unique<Sequence>();
        child->id = next_seq_id_++;
        child->req = s->req;
        child->index = k;
        child->tokens = s->tokens;
        child->prompt_len = s->prompt_len;
        child->num_computed = s->num_computed;
        child->block_table = s->block_table;
        child->block_hashes = s->block_hashes;
        for (BlockId b : child->block_table) bm_->incref(b);
        ++req.num_live_seqs;
        req_seqs_[req.id].push_back(child.get());
        sched_->add_running(child.get());
        Sequence* cp = child.get();
        seqs_.emplace(cp->id, std::move(child));
        process_token(cp, sampled[r + k]);
        ++generated;
      }
    }
    process_token(s, sampled[r]);
    ++generated;
    r += fan_out[i];
  }
  publish_stats(std::chrono::duration<double>(Clock::now() - t0).count(), prefill_tokens, generated, true);
  return out.num_tokens;
}

void Engine::process_token(Sequence* s, TokenId t) {
  Request& req = *s->req;
  const SamplingParams& p = req.params;
  s->tokens.push_back(t);
  ++req.usage.completion_tokens;
  if (!req.first_token_time) req.first_token_time = Clock::now();

  std::optional<std::string> finish;
  const bool is_eos = !p.ignore_eos && eos_ids_.count(t);
  if (is_eos) finish = "stop";

  // Incremental detokenization: only complete UTF-8 characters are released.
  if (!is_eos && !tok_->is_special(t)) s->pending_bytes += tok_->token_bytes(t);
  const size_t complete = utf8::complete_prefix_len(s->pending_bytes);
  const size_t prev_len = s->text.size();
  s->text += utf8::sanitize(std::string_view(s->pending_bytes).substr(0, complete));
  s->pending_bytes.erase(0, complete);

  size_t max_stop = 0;
  for (const auto& st : p.stop) max_stop = std::max(max_stop, st.size());
  if (!finish && max_stop > 0) {
    const size_t from = prev_len >= max_stop ? prev_len - max_stop + 1 : 0;
    size_t best = std::string::npos;
    for (const auto& st : p.stop)
      if (!st.empty()) best = std::min(best, s->text.find(st, from));
    if (best != std::string::npos) {
      s->text.resize(std::max(best, s->emitted));  // stop string itself is not returned
      finish = "stop";
    }
  }
  if (!finish && s->num_generated() >= p.max_tokens) finish = "length";
  if (!finish && static_cast<int>(s->tokens.size()) >= max_model_len_) finish = "length";
  if (finish && *finish == "length") s->text += utf8::sanitize(s->pending_bytes);

  // Hold back text that could be the beginning of a stop string.
  size_t upto = s->text.size();
  if (!finish && max_stop > 0) {
    size_t hold = 0;
    for (const auto& st : p.stop) {
      const size_t lim = std::min(st.size() - 1, s->text.size() - s->emitted);
      for (size_t k = lim; k > hold; --k)
        if (s->text.compare(s->text.size() - k, k, st, 0, k) == 0) {
          hold = k;
          break;
        }
    }
    upto -= hold;
  }
  StreamEvent ev;
  ev.index = s->index;
  ev.text = s->text.substr(s->emitted, upto - s->emitted);
  ev.num_tokens = 1;
  ev.token_ids = {t};
  ev.finish_reason = finish;
  s->emitted = upto;
  req.channel->push(std::move(ev));
  if (finish) finish_seq(s);
}

void Engine::finish_seq(Sequence* s) {
  sched_->remove(s);
  auto req = s->req;
  if (--req->num_live_seqs == 0) {
    req->channel->close(req->usage);
    req_seqs_.erase(req->id);
    std::lock_guard<std::mutex> lk(stats_mu_);
    ++stats_.requests_finished;
  } else {
    auto& v = req_seqs_[req->id];
    v.erase(std::remove(v.begin(), v.end(), s), v.end());
  }
  seqs_.erase(s->id);
}

void Engine::fail_all(const std::string& why) {
  std::vector<Sequence*> all;
  for (auto& [id, s] : seqs_) all.push_back(s.get());
  for (Sequence* s : all) {
    StreamEvent ev;
    ev.index = s->index;
    ev.finish_reason = "abort";
    s->req->channel->push(ev);
    finish_seq(s);
  }
  fprintf(stderr, "[engine] aborted %zu sequences: %s\n", all.size(), why.c_str());
}

void Engine::publish_stats(double step_seconds, int prefill_tokens, int generated, bool counted) {
  std::lock_guard<std::mutex> lk(stats_mu_);
  stats_.num_running = static_cast<int>(sched_->num_running());
  stats_.num_waiting = static_cast<int>(sched_->num_waiting());
  stats_.kv_blocks_free = bm_->num_free_uncached();
  stats_.kv_blocks_cached = bm_->num_cached_unused();
  stats_.prefix_query_tokens = bm_->prefix_query_tokens;
  stats_.prefix_hit_tokens = bm_->prefix_hit_tokens;
  stats_.num_preemptions = sched_->num_preemptions();
  stats_.num_cow_copies = bm_->num_cow_copies;
  stats_.num_evictions = bm_->num_evictions;
  stats_.steps += counted ? 1 : 0;
  stats_.prefill_tokens += prefill_tokens;
  stats_.generated_tokens += generated;
  stats_.busy_seconds += step_seconds;

  const auto now = Clock::now();
  const double dt = std::chrono::duration<double>(now - last_log_).count();
  if (dt >= 5.0 && thread_.joinable()) {
    const double hit = stats_.prefix_query_tokens
                           ? 100.0 * stats_.prefix_hit_tokens / static_cast<double>(stats_.prefix_query_tokens)
                           : 0.0;
    const int used = stats_.kv_blocks_total - stats_.kv_blocks_free - stats_.kv_blocks_cached;
    fprintf(stderr, "[engine] running %d | waiting %d | gen %.1f tok/s | kv used %.1f%% | prefix hit %.1f%% | "
                    "preempted %llu\n",
            stats_.num_running, stats_.num_waiting, (stats_.generated_tokens - gen_at_last_log_) / dt,
            100.0 * used / stats_.kv_blocks_total, hit, static_cast<unsigned long long>(stats_.num_preemptions));
    last_log_ = now;
    gen_at_last_log_ = stats_.generated_tokens;
  }
}

}  // namespace ps

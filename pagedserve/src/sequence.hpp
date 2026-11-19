// Requests, sequences and the channel that carries results back to the
// HTTP thread.
//
// Threading model: a Request is created by an HTTP thread and handed to the
// engine. After that, only the engine thread touches Sequence objects and the
// engine-side fields of Request. The two threads communicate only through
// (a) the engine's inbound queue, (b) RequestChannel, and (c) the atomic
// `aborted` flag.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "block_manager.hpp"
#include "sampler.hpp"

namespace ps {

using Clock = std::chrono::steady_clock;

struct Usage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int cached_tokens = 0;  // prompt tokens served from the prefix cache
};

struct StreamEvent {
  int index = 0;                       // choice index (n > 1)
  std::string text;                    // new text since the previous event
  int num_tokens = 0;                  // tokens covered by this event
  std::vector<TokenId> token_ids;
  std::optional<std::string> finish_reason;  // "stop" | "length" | "abort"
};

// Single-producer (engine) / single-consumer (HTTP thread) queue.
class RequestChannel {
 public:
  void push(StreamEvent ev) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      events_.push_back(std::move(ev));
    }
    cv_.notify_one();
  }
  void close(const Usage& usage) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      usage_ = usage;
      closed_ = true;
    }
    cv_.notify_one();
  }
  // Waits up to `timeout` for events. Returns all pending events; `done` is
  // set once the request has finished and every event has been drained.
  std::vector<StreamEvent> pop(std::chrono::milliseconds timeout, bool& done) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, timeout, [&] { return !events_.empty() || closed_; });
    std::vector<StreamEvent> out(std::make_move_iterator(events_.begin()), std::make_move_iterator(events_.end()));
    events_.clear();
    done = closed_;
    return out;
  }
  Usage usage() {
    std::lock_guard<std::mutex> lk(mu_);
    return usage_;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<StreamEvent> events_;
  bool closed_ = false;
  Usage usage_;
};

struct Request {
  std::string id;
  std::vector<TokenId> prompt;
  SamplingParams params;
  std::shared_ptr<RequestChannel> channel = std::make_shared<RequestChannel>();
  std::atomic<bool> aborted{false};
  Clock::time_point arrival = Clock::now();

  // Engine-thread state.
  int num_live_seqs = 0;
  bool forked = false;
  Usage usage;
  std::optional<Clock::time_point> first_token_time;
};

enum class SeqStatus { Waiting, Running, Finished };

struct Sequence {
  uint64_t id = 0;
  std::shared_ptr<Request> req;
  int index = 0;  // choice index within the request
  std::vector<TokenId> tokens;  // prompt + generated
  int prompt_len = 0;
  int num_computed = 0;         // tokens whose K/V are in the cache
  std::vector<BlockId> block_table;
  std::vector<uint64_t> block_hashes;  // chained keys of the full blocks registered so far
  SeqStatus status = SeqStatus::Waiting;
  int num_preemptions = 0;

  // Incremental detokenization state.
  std::string pending_bytes;  // trailing bytes of an incomplete UTF-8 character
  std::string text;           // decoded output so far
  size_t emitted = 0;         // bytes of `text` already sent to the client

  int num_generated() const { return static_cast<int>(tokens.size()) - prompt_len; }
  int num_uncomputed() const { return static_cast<int>(tokens.size()) - num_computed; }
};

}  // namespace ps

// Iteration-level scheduler (continuous batching).
//
// Every engine step calls schedule(), which decides which sequences run and
// how many tokens each one computes:
//   1. Running sequences first, in arrival order. A decoding sequence needs
//      one token; a sequence still prefilling gets a chunk of its prompt.
//      All of them share one per-step token budget (chunked prefill), so a
//      long prompt cannot stall everyone else's decode for a whole step.
//   2. If a running sequence needs a KV block and none is free, the most
//      recently admitted sequence is preempted: its blocks are released and
//      it goes back to the front of the queue to be recomputed later.
//   3. Waiting sequences are admitted while the budget, max_num_seqs and free
//      KV blocks allow. On admission the prompt is matched against the prefix
//      cache, and only the uncached suffix needs computing.
//
// static_batching=true gives the baseline the resume compares against: a
// batch is admitted only when the previous batch has completely finished.
#pragma once

#include <deque>
#include <vector>

#include "block_manager.hpp"
#include "config.hpp"
#include "sequence.hpp"

namespace ps {

struct ScheduledChunk {
  Sequence* seq;
  int num_tokens;
};

struct SchedulerOutput {
  std::vector<ScheduledChunk> chunks;  // single-token chunks first, then prefill chunks
  std::vector<BlockCopy> copies;       // copy-on-write copies to run before the forward pass
  std::vector<Sequence*> preempted;
  int num_tokens = 0;
  bool empty() const { return chunks.empty(); }
};

class Scheduler {
 public:
  Scheduler(const SchedulerConfig& cfg, BlockManager& bm);

  void add_waiting(Sequence* s);
  // Forked children already hold blocks and go straight to the running set.
  void add_running(Sequence* s);
  // Removes a finished or aborted sequence and releases its blocks.
  void remove(Sequence* s);

  SchedulerOutput schedule();

  size_t num_waiting() const { return waiting_.size(); }
  size_t num_running() const { return running_.size(); }
  bool has_work() const { return !waiting_.empty() || !running_.empty(); }
  uint64_t num_preemptions() const { return num_preemptions_; }

 private:
  // Allocates blocks (and copy-on-write copies) so that positions
  // [num_computed, num_computed + n) can be written. False if out of blocks.
  bool ensure_blocks(Sequence* s, int n, std::vector<BlockCopy>& copies);
  void preempt(Sequence* s);
  bool try_admit(Sequence* s, int budget, int& num_new, std::vector<BlockCopy>& copies, bool whole_prompt);

  SchedulerConfig cfg_;
  BlockManager& bm_;
  std::deque<Sequence*> waiting_;
  std::vector<Sequence*> running_;  // ordered by priority (oldest first)
  int watermark_blocks_;
  uint64_t num_preemptions_ = 0;
};

}  // namespace ps

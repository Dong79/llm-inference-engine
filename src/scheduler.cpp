#include "scheduler.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace ps {

Scheduler::Scheduler(const SchedulerConfig& cfg, BlockManager& bm)
    : cfg_(cfg), bm_(bm), watermark_blocks_(std::max(1, static_cast<int>(cfg.watermark * bm.num_blocks()))) {}

void Scheduler::add_waiting(Sequence* s) {
  s->status = SeqStatus::Waiting;
  waiting_.push_back(s);
}

void Scheduler::add_running(Sequence* s) {
  s->status = SeqStatus::Running;
  running_.push_back(s);
}

void Scheduler::remove(Sequence* s) {
  if (auto it = std::find(running_.begin(), running_.end(), s); it != running_.end()) running_.erase(it);
  if (auto it = std::find(waiting_.begin(), waiting_.end(), s); it != waiting_.end()) waiting_.erase(it);
  bm_.free_table(s->block_table);
  s->block_hashes.clear();
  s->status = SeqStatus::Finished;
}

bool Scheduler::ensure_blocks(Sequence* s, int n, std::vector<BlockCopy>& copies) {
  const int bs = bm_.block_size();
  const int end = s->num_computed + n;
  const size_t needed = static_cast<size_t>((end + bs - 1) / bs);
  // The first block we write into may be a partially filled block shared with
  // a sibling sequence (after a fork): copy it before writing.
  const size_t first = static_cast<size_t>(s->num_computed / bs);
  if (first < s->block_table.size() && !bm_.make_writable(s->block_table, first, copies)) return false;
  while (s->block_table.size() < needed) {
    BlockId b = bm_.allocate();
    if (b < 0) return false;
    s->block_table.push_back(b);
  }
  return true;
}

void Scheduler::preempt(Sequence* s) {
  // Recompute-style preemption: drop the KV and requeue at the front. Its full
  // blocks stay in the prefix cache, so re-admission is often cheap.
  bm_.free_table(s->block_table);
  s->block_hashes.clear();
  s->num_computed = 0;
  s->status = SeqStatus::Waiting;
  ++s->num_preemptions;
  ++num_preemptions_;
  waiting_.push_front(s);
}

bool Scheduler::try_admit(Sequence* s, int budget, int& num_new, std::vector<BlockCopy>& copies,
                          bool whole_prompt) {
  const int bs = bm_.block_size();
  const int total = static_cast<int>(s->tokens.size());
  if (s->block_table.empty() && s->num_computed == 0) {
    // Prefix cache lookup. Never match the final token: we need its logits.
    const int matched = bm_.match_prefix(s->tokens, total - 1, s->block_table, s->block_hashes);
    s->num_computed = matched;
  }
  const int remaining = total - s->num_computed;
  num_new = whole_prompt ? remaining : std::min(remaining, budget);
  if (!cfg_.chunked_prefill && num_new < remaining) num_new = 0;
  // Blocks needed for everything up to the end of the prompt must fit, plus a
  // watermark so that admitted sequences can still grow by a few tokens.
  const int needed = (total + bs - 1) / bs - static_cast<int>(s->block_table.size());
  const bool fits = num_new > 0 && bm_.num_available() - needed >= watermark_blocks_;
  if (!fits || !ensure_blocks(s, whole_prompt ? remaining : num_new, copies)) {
    // Roll back the prefix match; the blocks return to the cache.
    bm_.free_table(s->block_table);
    s->block_hashes.clear();
    s->num_computed = 0;
    return false;
  }
  bm_.prefix_query_tokens += total;
  bm_.prefix_hit_tokens += s->num_computed;
  if (s->req && s->index == 0 && s->num_generated() == 0) s->req->usage.cached_tokens = s->num_computed;
  return true;
}

SchedulerOutput Scheduler::schedule() {
  SchedulerOutput out;
  int budget = cfg_.max_num_batched_tokens;
  std::vector<ScheduledChunk> singles, multis;

  // Static batching: admit a whole batch only once the previous one is done.
  if (cfg_.static_batching && running_.empty()) {
    while (!waiting_.empty() && static_cast<int>(running_.size()) < cfg_.max_num_seqs) {
      Sequence* s = waiting_.front();
      int n = 0;
      if (!try_admit(s, INT_MAX, n, out.copies, /*whole_prompt=*/true)) break;
      waiting_.pop_front();
      s->status = SeqStatus::Running;
      running_.push_back(s);
    }
  }

  // 1. Running sequences, oldest first.
  for (size_t i = 0; i < running_.size() && budget > 0;) {
    Sequence* s = running_[i];
    int n = std::min(s->num_uncomputed(), budget);
    if (n <= 0) {
      ++i;
      continue;
    }
    if (!cfg_.chunked_prefill && n < s->num_uncomputed()) break;
    bool self_preempted = false;
    while (!ensure_blocks(s, n, out.copies)) {
      // Out of KV blocks: preempt the lowest-priority (newest) sequence.
      Sequence* victim = running_.back();
      running_.pop_back();
      preempt(victim);
      out.preempted.push_back(victim);
      if (victim == s) {
        self_preempted = true;
        break;
      }
    }
    if (self_preempted) break;
    (n == 1 ? singles : multis).push_back({s, n});
    budget -= n;
    ++i;
  }

  // 2. Admit waiting sequences (continuous batching). Skip if we just had to
  //    preempt: memory is tight and admitting would only cause thrashing.
  if (!cfg_.static_batching && out.preempted.empty()) {
    while (!waiting_.empty() && budget > 0 && static_cast<int>(running_.size()) < cfg_.max_num_seqs) {
      Sequence* s = waiting_.front();
      int n = 0;
      if (!try_admit(s, budget, n, out.copies, /*whole_prompt=*/false)) break;
      waiting_.pop_front();
      s->status = SeqStatus::Running;
      running_.push_back(s);
      (n == 1 ? singles : multis).push_back({s, n});
      budget -= n;
    }
  }

  out.chunks = std::move(singles);
  out.chunks.insert(out.chunks.end(), multis.begin(), multis.end());
  for (const auto& c : out.chunks) out.num_tokens += c.num_tokens;
  return out;
}

}  // namespace ps

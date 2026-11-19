// Paged KV-cache block manager.
//
// The KV cache is one big pool of fixed-size blocks (block_size tokens each).
// Every sequence owns a block table: logical block i -> physical block id,
// exactly like a page table. This class only does bookkeeping on block ids;
// the actual K/V tensors live in KVCache (kv_cache.hpp).
//
// Features:
//   * O(1) allocation from a free list.
//   * Reference counts, so several sequences can point at the same block.
//   * Prefix caching: every FULL block is keyed by a chained hash
//       key(i) = H(key(i-1), tokens in block i)
//     so the key encodes the entire prefix, not just the block's own tokens.
//     (A token's K/V depends on every token before it and on its position.)
//   * Blocks whose refcount drops to 0 but that are in the prefix cache are not
//     freed. They go to an LRU "evictable" list and can be revived by a later
//     request with the same prefix. They are only reclaimed when the free list
//     is empty.
//   * Copy-on-write: a sequence about to write into a block it shares gets a
//     private copy first (the caller performs the actual tensor copy).
//
// Not thread-safe by design: only the engine thread touches it.
#pragma once

#include <cstdint>
#include <list>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ps {

using BlockId = int32_t;
using TokenId = int32_t;

struct BlockCopy {
  BlockId src;
  BlockId dst;
};

class BlockManager {
 public:
  BlockManager(int num_blocks, int block_size, bool enable_prefix_caching);

  int block_size() const { return block_size_; }
  int num_blocks() const { return static_cast<int>(blocks_.size()); }
  // Blocks that can be handed out right now (free + cached-but-unused).
  int num_available() const { return static_cast<int>(free_.size() + evictable_.size()); }
  int num_free_uncached() const { return static_cast<int>(free_.size()); }
  int num_cached_unused() const { return static_cast<int>(evictable_.size()); }
  bool prefix_caching() const { return prefix_caching_; }

  // Allocates one exclusively-owned block (refcount 1). Evicts the least
  // recently used cached block if the free list is empty. Returns -1 if no
  // block is available.
  BlockId allocate();

  void incref(BlockId b);
  // Drops one reference. At zero the block goes to the evictable LRU list if
  // it is in the prefix cache, otherwise straight back to the free list.
  void decref(BlockId b);
  int refcount(BlockId b) const { return blocks_[b].ref; }

  // Releases every block in `table` (in reverse order so that the tail of a
  // sequence is evicted before its head) and clears it.
  void free_table(std::vector<BlockId>& table);

  // Prefix-cache lookup. Walks `tokens` in block-size chunks and returns the
  // physical blocks of the longest cached prefix, with a reference taken on
  // each. At most `max_tokens` tokens are matched (rounded down to full
  // blocks), which lets the caller leave the last prompt token uncached so it
  // still gets logits. `out_hashes` receives the chained key of each block.
  int match_prefix(std::span<const TokenId> tokens, int max_tokens, std::vector<BlockId>& out_blocks,
                   std::vector<uint64_t>& out_hashes);

  // Registers a block that has just become full (all block_size tokens have
  // K/V computed) under its chained key. No-op if prefix caching is off or an
  // identical block is already registered (two requests can compute the same
  // prefix concurrently; the duplicate is simply not shared).
  void register_full_block(BlockId b, uint64_t key, uint64_t parent_key, std::span<const TokenId> tokens);

  // Ensures the caller owns block `table[idx]` exclusively before writing to
  // it. If the block is shared, allocates a new block, appends the required
  // copy to `copies`, swaps it into the table and drops the old reference.
  // Returns false if no block was available for the copy.
  bool make_writable(std::vector<BlockId>& table, size_t idx, std::vector<BlockCopy>& copies);

  static uint64_t hash_block(uint64_t parent_key, std::span<const TokenId> tokens);
  static constexpr uint64_t kRootKey = 0x9E3779B97F4A7C15ull;

  // Counters for /stats.
  uint64_t prefix_query_tokens = 0;  // tokens looked up in the prefix cache
  uint64_t prefix_hit_tokens = 0;    // tokens served from it
  uint64_t num_cow_copies = 0;
  uint64_t num_evictions = 0;

 private:
  struct Block {
    int ref = 0;
    bool cached = false;  // registered in hash map
    uint64_t key = 0;
    uint64_t parent_key = 0;
    std::vector<TokenId> tokens;  // kept to verify hash hits (no silent collisions)
    std::list<BlockId>::iterator lru_it;
    bool in_lru = false;
  };

  void evict_from_cache(BlockId b);

  int block_size_;
  bool prefix_caching_;
  std::vector<Block> blocks_;
  std::vector<BlockId> free_;           // never-cached free blocks (stack)
  std::list<BlockId> evictable_;        // cached, ref==0; front = least recently used
  std::unordered_map<uint64_t, BlockId> cache_;  // chained key -> block
};

}  // namespace ps

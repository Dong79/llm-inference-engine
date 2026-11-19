#include "block_manager.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace ps {

namespace {
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
}  // namespace

BlockManager::BlockManager(int num_blocks, int block_size, bool enable_prefix_caching)
    : block_size_(block_size), prefix_caching_(enable_prefix_caching), blocks_(num_blocks) {
  if (num_blocks <= 0 || block_size <= 0) throw std::invalid_argument("BlockManager: bad sizes");
  free_.reserve(num_blocks);
  // Push in reverse so that allocation hands out 0, 1, 2, ... (easier to debug).
  for (int i = num_blocks - 1; i >= 0; --i) free_.push_back(i);
  cache_.reserve(static_cast<size_t>(num_blocks) * 2);
}

uint64_t BlockManager::hash_block(uint64_t parent_key, std::span<const TokenId> tokens) {
  uint64_t h = splitmix64(parent_key ^ (static_cast<uint64_t>(tokens.size()) << 32));
  for (TokenId t : tokens) h = splitmix64(h ^ static_cast<uint64_t>(static_cast<uint32_t>(t)));
  return h;
}

BlockId BlockManager::allocate() {
  BlockId b;
  if (!free_.empty()) {
    b = free_.back();
    free_.pop_back();
  } else if (!evictable_.empty()) {
    // Reclaim the least recently used cached block.
    b = evictable_.front();
    evictable_.pop_front();
    blocks_[b].in_lru = false;
    evict_from_cache(b);
    ++num_evictions;
  } else {
    return -1;
  }
  assert(blocks_[b].ref == 0);
  blocks_[b].ref = 1;
  return b;
}

void BlockManager::incref(BlockId b) {
  Block& blk = blocks_[b];
  if (blk.ref == 0) {
    // Reviving a cached block that nobody was using.
    if (!blk.in_lru) throw std::logic_error("incref on a free block");
    evictable_.erase(blk.lru_it);
    blk.in_lru = false;
  }
  ++blk.ref;
}

void BlockManager::decref(BlockId b) {
  Block& blk = blocks_[b];
  if (blk.ref <= 0) throw std::logic_error("decref on unreferenced block");
  if (--blk.ref > 0) return;
  if (blk.cached) {
    evictable_.push_back(b);  // most recently used end
    blk.lru_it = std::prev(evictable_.end());
    blk.in_lru = true;
  } else {
    free_.push_back(b);
  }
}

void BlockManager::free_table(std::vector<BlockId>& table) {
  for (auto it = table.rbegin(); it != table.rend(); ++it) decref(*it);
  table.clear();
}

int BlockManager::match_prefix(std::span<const TokenId> tokens, int max_tokens, std::vector<BlockId>& out_blocks,
                               std::vector<uint64_t>& out_hashes) {
  if (!prefix_caching_) return 0;
  const size_t limit = std::min<size_t>(tokens.size(), static_cast<size_t>(std::max(0, max_tokens)));
  const size_t n_full = limit / block_size_;
  uint64_t parent = kRootKey;
  size_t matched = 0;
  for (size_t i = 0; i < n_full; ++i) {
    auto chunk = tokens.subspan(i * block_size_, block_size_);
    const uint64_t key = hash_block(parent, chunk);
    auto it = cache_.find(key);
    if (it == cache_.end()) break;
    const Block& blk = blocks_[it->second];
    // Verify: a hash collision must never hand back the wrong K/V.
    if (blk.parent_key != parent || !std::equal(chunk.begin(), chunk.end(), blk.tokens.begin(), blk.tokens.end()))
      break;
    incref(it->second);
    out_blocks.push_back(it->second);
    out_hashes.push_back(key);
    parent = key;
    matched += block_size_;
  }
  return static_cast<int>(matched);
}

void BlockManager::register_full_block(BlockId b, uint64_t key, uint64_t parent_key, std::span<const TokenId> tokens) {
  if (!prefix_caching_) return;
  Block& blk = blocks_[b];
  if (blk.cached) return;
  if (cache_.count(key)) return;  // an identical block is already cached
  blk.cached = true;
  blk.key = key;
  blk.parent_key = parent_key;
  blk.tokens.assign(tokens.begin(), tokens.end());
  cache_.emplace(key, b);
}

void BlockManager::evict_from_cache(BlockId b) {
  Block& blk = blocks_[b];
  if (!blk.cached) return;
  auto it = cache_.find(blk.key);
  if (it != cache_.end() && it->second == b) cache_.erase(it);
  blk.cached = false;
  blk.tokens.clear();
}

bool BlockManager::make_writable(std::vector<BlockId>& table, size_t idx, std::vector<BlockCopy>& copies) {
  const BlockId b = table[idx];
  Block& blk = blocks_[b];
  if (blk.ref == 1) {
    // Sole owner. If it was cached, its contents are about to change, so it
    // must leave the prefix cache.
    if (blk.cached) evict_from_cache(b);
    return true;
  }
  // Shared: copy-on-write.
  const BlockId nb = allocate();
  if (nb < 0) return false;
  copies.push_back({b, nb});
  table[idx] = nb;
  decref(b);
  ++num_cow_copies;
  return true;
}

}  // namespace ps

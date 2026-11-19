// Unit tests for the paged KV block manager (no Torch needed).
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "block_manager.hpp"

using namespace ps;

static int g_failures = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

static std::vector<TokenId> iota_tokens(int n, int start = 0) {
  std::vector<TokenId> v(n);
  std::iota(v.begin(), v.end(), start);
  return v;
}

// Simulates what the engine does after computing a sequence's K/V:
// registers every full block under its chained key.
static void register_all_full(BlockManager& bm, const std::vector<BlockId>& table, const std::vector<TokenId>& toks) {
  const int bs = bm.block_size();
  uint64_t parent = BlockManager::kRootKey;
  for (size_t i = 0; i < toks.size() / bs && i < table.size(); ++i) {
    std::span<const TokenId> chunk(toks.data() + i * bs, bs);
    uint64_t key = BlockManager::hash_block(parent, chunk);
    bm.register_full_block(table[i], key, parent, chunk);
    parent = key;
  }
}

static void test_alloc_free() {
  BlockManager bm(4, 16, false);
  CHECK(bm.num_available() == 4);
  std::vector<BlockId> t;
  for (int i = 0; i < 4; ++i) t.push_back(bm.allocate());
  CHECK(bm.allocate() == -1);
  CHECK(bm.num_available() == 0);
  bm.free_table(t);
  CHECK(bm.num_available() == 4);
  CHECK(t.empty());
}

static void test_prefix_match_and_chain() {
  BlockManager bm(16, 4, true);
  auto toks = iota_tokens(10);  // 2 full blocks + 2 tokens
  std::vector<BlockId> ta;
  for (int i = 0; i < 3; ++i) ta.push_back(bm.allocate());
  register_all_full(bm, ta, toks);

  // Same prefix -> both full blocks hit, refcount 2.
  std::vector<BlockId> tb;
  std::vector<uint64_t> hb;
  int m = bm.match_prefix(toks, 9, tb, hb);
  CHECK(m == 8);
  CHECK(tb.size() == 2 && tb[0] == ta[0] && tb[1] == ta[1]);
  CHECK(bm.refcount(ta[0]) == 2);

  // max_tokens caps the match (the caller keeps the last prompt token uncached).
  std::vector<BlockId> tc;
  std::vector<uint64_t> hc;
  CHECK(bm.match_prefix(toks, 7, tc, hc) == 4);
  bm.free_table(tc);

  // Same tokens in block 1 but different block 0 -> chained key must miss block 1.
  auto other = toks;
  other[0] = 999;
  std::vector<BlockId> td;
  std::vector<uint64_t> hd;
  CHECK(bm.match_prefix(other, 9, td, hd) == 0);

  // A suffix block alone (block 1's tokens at position 0) must not match.
  std::vector<TokenId> shifted(toks.begin() + 4, toks.end());
  CHECK(bm.match_prefix(shifted, 6, td, hd) == 0);

  bm.free_table(tb);
  bm.free_table(ta);
  CHECK(bm.num_available() == 16);
}

static void test_cached_blocks_survive_until_evicted() {
  BlockManager bm(3, 4, true);
  auto toks = iota_tokens(8);
  std::vector<BlockId> ta{bm.allocate(), bm.allocate()};
  register_all_full(bm, ta, toks);
  bm.free_table(ta);  // request finished
  CHECK(bm.num_cached_unused() == 2);
  CHECK(bm.num_available() == 3);

  // Next turn of the conversation hits the cache.
  std::vector<BlockId> tb;
  std::vector<uint64_t> hb;
  CHECK(bm.match_prefix(toks, 100, tb, hb) == 8);
  CHECK(bm.num_cached_unused() == 0);
  bm.free_table(tb);

  // Memory pressure: allocating 3 blocks reclaims both cached ones.
  std::vector<BlockId> tc{bm.allocate(), bm.allocate(), bm.allocate()};
  CHECK(tc[2] >= 0);
  CHECK(bm.num_evictions == 2);
  std::vector<BlockId> td;
  std::vector<uint64_t> hd;
  CHECK(bm.match_prefix(toks, 100, td, hd) == 0);  // evicted -> gone
  bm.free_table(tc);
}

static void test_lru_order_evicts_tail_first() {
  BlockManager bm(3, 2, true);
  auto toks = iota_tokens(4);
  std::vector<BlockId> ta{bm.allocate(), bm.allocate()};
  register_all_full(bm, ta, toks);
  BlockId head = ta[0];
  bm.free_table(ta);  // releases tail first -> tail is LRU
  // Need two blocks: one free + one eviction. The tail should go, head stays.
  std::vector<BlockId> tb{bm.allocate(), bm.allocate()};
  CHECK(bm.num_evictions == 1);
  std::vector<BlockId> tc;
  std::vector<uint64_t> hc;
  CHECK(bm.match_prefix(toks, 100, tc, hc) == 2);
  CHECK(tc.size() == 1 && tc[0] == head);
  bm.free_table(tc);
  bm.free_table(tb);
}

static void test_copy_on_write() {
  BlockManager bm(8, 4, true);
  std::vector<BlockId> parent{bm.allocate(), bm.allocate()};  // second block partially filled
  // Fork: child shares both blocks.
  std::vector<BlockId> child = parent;
  for (BlockId b : child) bm.incref(b);
  CHECK(bm.refcount(parent[1]) == 2);

  std::vector<BlockCopy> copies;
  CHECK(bm.make_writable(child, 1, copies));
  CHECK(copies.size() == 1 && copies[0].src == parent[1] && copies[0].dst == child[1]);
  CHECK(child[1] != parent[1]);
  CHECK(bm.refcount(parent[1]) == 1 && bm.refcount(child[1]) == 1);

  // Parent is now the sole owner: no copy needed.
  copies.clear();
  CHECK(bm.make_writable(parent, 1, copies));
  CHECK(copies.empty());
  CHECK(bm.num_cow_copies == 1);
  bm.free_table(parent);
  bm.free_table(child);
  CHECK(bm.num_available() == 8);
}

static void test_duplicate_registration() {
  BlockManager bm(4, 2, true);
  auto toks = iota_tokens(2);
  std::vector<BlockId> a{bm.allocate()}, b{bm.allocate()};
  register_all_full(bm, a, toks);
  register_all_full(bm, b, toks);  // computed concurrently -> not registered twice
  bm.free_table(a);
  bm.free_table(b);
  CHECK(bm.num_cached_unused() == 1);
  CHECK(bm.num_free_uncached() == 3);
}

int main() {
  test_alloc_free();
  test_prefix_match_and_chain();
  test_cached_blocks_survive_until_evicted();
  test_lru_order_evicts_tail_first();
  test_copy_on_write();
  test_duplicate_registration();
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("test_block_manager: all tests passed\n");
  return 0;
}

// pagedserve-dev: developer tool for tests and offline benchmarks.
//
//   logits      Run one prompt through the paged model (chunked prefill, then
//               token-by-token decode, twice in one batch with different block
//               tables) and dump logits for every position.
//   offline     Drive the engine directly (no HTTP) on a JSON list of requests
//               or a synthetic workload; prints outputs and throughput.
//   attn-check  Compare the CUDA paged-attention kernel with the reference.
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "attention.hpp"
#include "cli.hpp"
#ifdef PAGEDSERVE_WITH_CUDA
#include "cuda/paged_attention.hpp"
#endif
#include "engine.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "nlohmann/json.hpp"

using namespace ps;
using json = nlohmann::json;

namespace {

std::vector<int32_t> parse_ids(const std::string& s) {
  std::vector<int32_t> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) v.push_back(std::stoi(item));
  return v;
}

int cmd_logits(Args& a) {
  const std::string dir = a.str("model");
  const auto tokens = parse_ids(a.str("tokens"));
  const int chunk = a.i("chunk", 1 << 30);
  const int prefill = a.i("prefill", static_cast<int>(tokens.size()));
  const int bs = a.i("block-size", 16);
  const std::string out_path = a.str("out");
  torch::Device dev(a.str("device", "cpu"));
  const std::string dt = a.str("dtype", "float32");
  a.check_unused();
  const auto dtype = dt == "float16" ? torch::kFloat16 : dt == "bfloat16" ? torch::kBFloat16 : torch::kFloat32;

  ModelConfig cfg = ModelConfig::from_dir(dir);
  Model model(cfg, dir, dev, dtype);
  const int L = static_cast<int>(tokens.size());
  const int nb = (L + bs - 1) / bs;
  KVCache cache(cfg.num_layers, 2 * nb + 1, bs, cfg.num_kv_heads, cfg.head_dim, dev, dtype);
  // Two copies of the sequence with scrambled, disjoint block tables: proves
  // the result does not depend on where blocks physically live.
  std::vector<BlockId> ta(nb), tb(nb);
  for (int i = 0; i < nb; ++i) {
    ta[i] = 2 * nb - 2 * i;      // descending even ids
    tb[i] = 2 * i + 1;           // ascending odd ids
  }
  std::vector<torch::Tensor> rows;
  double max_dup_diff = 0;
  int pos = 0;
  while (pos < L) {
    const int n = pos < prefill ? std::min(chunk, prefill - pos) : 1;
    std::vector<SeqChunk> chunks = {
        {tokens, pos, n, ta, true, true},
        {tokens, pos, n, tb, true, true},
    };
    ModelInput in = build_model_input(chunks, bs, dev);
    auto logits = model.forward(in, cache).to(torch::kCPU);  // [2n, V]
    auto la = logits.narrow(0, 0, n), lb = logits.narrow(0, n, n);
    max_dup_diff = std::max(max_dup_diff, (la - lb).abs().max().item<double>());
    rows.push_back(la);
    pos += n;
  }
  auto all = torch::cat(rows, 0).contiguous();
  std::ofstream f(out_path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(all.data_ptr<float>()), all.numel() * sizeof(float));
  std::cout << json{{"rows", all.size(0)}, {"vocab", all.size(1)}, {"max_dup_diff", max_dup_diff}}.dump() << "\n";
  return 0;
}

struct OfflineReq {
  std::shared_ptr<Request> req;
  int arrive_step = 0;
  std::vector<std::vector<TokenId>> ids;
  std::vector<std::string> text;
  std::vector<std::string> finish;
  bool submitted = false;
  bool done = false;
  double submit_t = 0, first_t = -1, end_t = 0;
};

int cmd_offline(Args& a) {
  const std::string req_path = a.str("requests");
  const std::string out_path = a.str("out");
  const int synthetic = a.i("synthetic", 0);
  const std::string plen = a.str("prompt-len", "64:512");
  const std::string olen = a.str("output-len", "16:256");
  const unsigned seed = static_cast<unsigned>(a.i("seed", 0));
  EngineConfig cfg = parse_engine_args(a);
  a.check_unused();

  Engine engine(cfg);
  std::cerr << engine.describe();
  std::vector<OfflineReq> reqs;
  auto make = [&](std::vector<TokenId> prompt, SamplingParams p, int arrive) {
    OfflineReq r;
    r.req = std::make_shared<Request>();
    r.req->id = "req-" + std::to_string(reqs.size());
    r.req->prompt = std::move(prompt);
    r.req->params = p;
    r.arrive_step = arrive;
    r.ids.resize(p.n);
    r.text.resize(p.n);
    r.finish.resize(p.n);
    reqs.push_back(std::move(r));
  };

  if (synthetic > 0) {
    // Random prompts and output lengths; ignore_eos makes lengths exact.
    std::mt19937 rng(seed);
    auto range = [](const std::string& s) {
      auto c = s.find(':');
      return std::make_pair(std::stoi(s.substr(0, c)), std::stoi(s.substr(c + 1)));
    };
    auto [plo, phi] = range(plen);
    auto [olo, ohi] = range(olen);
    const int vocab = std::min(engine.tokenizer().vocab_size(), engine.model_config().vocab_size);
    for (int i = 0; i < synthetic; ++i) {
      std::vector<TokenId> prompt(std::uniform_int_distribution<int>(plo, phi)(rng));
      for (auto& t : prompt) t = std::uniform_int_distribution<int>(0, vocab - 1)(rng);
      SamplingParams p;
      p.temperature = 0;
      p.ignore_eos = true;
      p.max_tokens = std::uniform_int_distribution<int>(olo, ohi)(rng);
      make(std::move(prompt), p, 0);
    }
  } else {
    std::ifstream f(req_path);
    if (!f) throw std::runtime_error("cannot open --requests file");
    json j = json::parse(f);
    for (const auto& r : j) {
      std::vector<TokenId> prompt = r.contains("prompt_ids") ? r["prompt_ids"].get<std::vector<TokenId>>()
                                                             : engine.tokenizer().encode(r.at("prompt").get<std::string>());
      SamplingParams p;
      p.temperature = r.value("temperature", 0.0f);
      p.top_p = r.value("top_p", 1.0f);
      p.top_k = r.value("top_k", -1);
      p.max_tokens = r.value("max_tokens", 16);
      p.n = r.value("n", 1);
      p.ignore_eos = r.value("ignore_eos", true);
      if (r.contains("stop")) p.stop = r["stop"].get<std::vector<std::string>>();
      make(std::move(prompt), p, r.value("arrive_step", 0));
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  auto now = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  int step = 0;
  size_t finished = 0;
  uint64_t gen_tokens = 0;
  while (finished < reqs.size()) {
    bool any_pending = false;
    int next_arrival = INT32_MAX;
    for (auto& r : reqs) {
      if (r.submitted) continue;
      if (r.arrive_step <= step) {
        engine.add_request(r.req);
        r.submitted = true;
        r.submit_t = now();
      } else {
        any_pending = true;
        next_arrival = std::min(next_arrival, r.arrive_step);
      }
    }
    const int processed = engine.step();
    if (processed == 0 && any_pending) step = next_arrival - 1;  // idle: jump ahead
    ++step;
    for (auto& r : reqs) {
      if (!r.submitted || r.done) continue;
      bool done = false;
      auto evs = r.req->channel->pop(std::chrono::milliseconds(0), done);
      for (auto& e : evs) {
        if (r.first_t < 0) r.first_t = now();
        r.ids[e.index].insert(r.ids[e.index].end(), e.token_ids.begin(), e.token_ids.end());
        r.text[e.index] += e.text;
        if (e.finish_reason) r.finish[e.index] = *e.finish_reason;
        gen_tokens += e.num_tokens;
      }
      if (done) {
        r.done = true;
        r.end_t = now();
        ++finished;
      }
    }
    if (processed == 0 && !any_pending && finished < reqs.size() && !engine.has_work())
      throw std::runtime_error("engine idle but requests unfinished");
  }
  const double elapsed = now();
  auto st = engine.stats();

  json results = json::array();
  std::vector<double> ttft;
  for (auto& r : reqs) {
    json choices = json::array();
    for (size_t k = 0; k < r.ids.size(); ++k)
      choices.push_back({{"token_ids", r.ids[k]}, {"text", r.text[k]}, {"finish_reason", r.finish[k]}});
    auto u = r.req->channel->usage();
    results.push_back({{"choices", choices},
                       {"usage", {{"prompt_tokens", u.prompt_tokens}, {"completion_tokens", u.completion_tokens},
                                  {"cached_tokens", u.cached_tokens}}}});
    ttft.push_back(r.first_t - r.submit_t);
  }
  std::sort(ttft.begin(), ttft.end());
  json summary = {{"requests", reqs.size()},
                  {"elapsed_s", elapsed},
                  {"engine_steps", st.steps},
                  {"generated_tokens", gen_tokens},
                  {"output_tokens_per_s", gen_tokens / elapsed},
                  {"prefill_tokens", st.prefill_tokens},
                  {"avg_batch_tokens", st.steps ? double(st.prefill_tokens + st.generated_tokens) / st.steps : 0},
                  {"ttft_p50_s", ttft[ttft.size() / 2]},
                  {"ttft_p95_s", ttft[std::min(ttft.size() - 1, ttft.size() * 95 / 100)]},
                  {"preemptions", st.num_preemptions},
                  {"cow_copies", st.num_cow_copies},
                  {"prefix_hit_tokens", st.prefix_hit_tokens},
                  {"prefix_query_tokens", st.prefix_query_tokens}};
  std::cerr << summary.dump(2) << "\n";
  if (!out_path.empty()) {
    std::ofstream f(out_path);
    f << json{{"results", results}, {"summary", summary}}.dump() << "\n";
  }
  return 0;
}

int cmd_attn_check(Args& a) {
  a.check_unused();
#ifndef PAGEDSERVE_WITH_CUDA
  std::cerr << "built without PAGEDSERVE_CUDA; nothing to check\n";
  return 0;
#else
  if (!torch::cuda::is_available()) {
    std::cerr << "no CUDA device\n";
    return 1;
  }
  torch::manual_seed(0);
  torch::Device dev(torch::kCUDA);
  int failures = 0;
  struct Case { int nseq, hq, hkv, d, bs, max_ctx; torch::ScalarType dt; };
  std::vector<Case> cases = {{1, 8, 8, 64, 16, 1, torch::kFloat32},     {6, 4, 2, 32, 4, 90, torch::kFloat32},     {5, 8, 2, 64, 16, 300, torch::kFloat32},
                             {7, 14, 2, 64, 16, 1000, torch::kBFloat16}, {3, 32, 8, 128, 16, 2048, torch::kFloat16},
                             {9, 16, 8, 128, 32, 777, torch::kBFloat16}, {4, 16, 4, 256, 8, 129, torch::kFloat16}};
  for (const auto& c : cases) {
    auto opt = torch::TensorOptions().device(dev).dtype(c.dt);
    const int max_blocks = (c.max_ctx + c.bs - 1) / c.bs;
    const int num_blocks = c.nseq * max_blocks + 3;
    auto kc = torch::randn({num_blocks, c.bs, c.hkv, c.d}, opt);
    auto vc = torch::randn({num_blocks, c.bs, c.hkv, c.d}, opt);
    auto q = torch::randn({c.nseq, c.hq, c.d}, opt);
    auto perm = torch::randperm(num_blocks, torch::kInt32).narrow(0, 0, c.nseq * max_blocks);
    auto bt = perm.view({c.nseq, max_blocks}).to(dev);
    std::vector<int32_t> lens(c.nseq);
    for (int i = 0; i < c.nseq; ++i) lens[i] = 1 + (c.max_ctx - 1) * (i + 1) / c.nseq;
    auto cl = torch::tensor(lens, torch::kInt32).to(dev);
    const double scale = 1.0 / std::sqrt(c.d);
    auto ref = paged_decode_reference(q, kc, vc, bt, cl, scale).to(torch::kFloat32);
    auto got = paged_attention_decode_cuda(q, kc, vc, bt, cl, c.max_ctx, scale).to(torch::kFloat32);
    const double err = (ref - got).abs().max().item<double>();
    const double tol = c.dt == torch::kFloat32 ? 1e-4 : 2e-2;
    std::cout << "seqs=" << c.nseq << " hq=" << c.hq << " hkv=" << c.hkv << " d=" << c.d << " bs=" << c.bs
              << " ctx<=" << c.max_ctx << " dtype=" << c10::toString(c.dt) << "  max_err=" << err
              << (err <= tol ? "  OK" : "  FAIL") << "\n";
    failures += err > tol;
  }
  return failures ? 1 : 0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: pagedserve-dev <logits|offline|attn-check> [options]\n\n" << kEngineHelp;
    return 2;
  }
  const std::string cmd = argv[1];
  try {
    Args a(argc, argv, 2);
    if (cmd == "logits") return cmd_logits(a);
    if (cmd == "offline") return cmd_offline(a);
    if (cmd == "attn-check") return cmd_attn_check(a);
    std::cerr << "unknown command " << cmd << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

#include "tokenizer.hpp"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <tuple>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "utf8.hpp"

namespace ps {

using json = nlohmann::json;

namespace {

// The GPT-2 pre-tokenizer regex used by ByteLevel(use_regex=true).
constexpr const char* kGpt2Pattern =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool file_exists(const std::string& path) { return std::ifstream(path).good(); }

std::string escape_regex(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (std::string_view("\\^$.|?*+()[]{}").find(c) != std::string_view::npos) out += '\\';
    out += c;
  }
  return out;
}

}  // namespace

// One regex split step (HF `Split` pre-tokenizer).
struct Tokenizer::Splitter {
  pcre2_code* code = nullptr;
  bool remove_matches = false;  // behavior "Removed" (else "Isolated")
  bool invert = false;

  Splitter(const std::string& pattern, bool removed, bool inv) : remove_matches(removed), invert(inv) {
    int err = 0;
    PCRE2_SIZE off = 0;
    code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.c_str()), pattern.size(),
                         PCRE2_UTF | PCRE2_UCP | PCRE2_MATCH_INVALID_UTF, &err, &off, nullptr);
    if (!code) {
      PCRE2_UCHAR msg[256];
      pcre2_get_error_message(err, msg, sizeof(msg));
      throw std::runtime_error("bad pre-tokenizer regex at " + std::to_string(off) + ": " +
                               reinterpret_cast<char*>(msg));
    }
    pcre2_jit_compile(code, PCRE2_JIT_COMPLETE);  // optional speedup; falls back if unavailable
  }
  ~Splitter() { pcre2_code_free(code); }

  void split(std::string_view s, std::vector<std::string_view>& out) const {
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(code, nullptr);
    size_t pos = 0, last = 0;
    auto emit = [&](size_t a, size_t b, bool is_match) {
      if (b <= a) return;
      const bool keep = invert ? (is_match || !remove_matches) : (!is_match || !remove_matches);
      if (keep) out.push_back(s.substr(a, b - a));
    };
    while (pos <= s.size()) {
      int rc = pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(s.data()), s.size(), pos, 0, md, nullptr);
      if (rc < 0) break;
      PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
      size_t ms = ov[0], me = ov[1];
      if (me == ms) {  // empty match: step one code point forward
        size_t i = pos;
        if (i >= s.size()) break;
        utf8::next(s, i);
        pos = i;
        continue;
      }
      emit(last, ms, false);
      emit(ms, me, true);
      last = pos = me;
    }
    emit(last, s.size(), false);
    pcre2_match_data_free(md);
  }
};

Tokenizer::~Tokenizer() = default;

std::unique_ptr<Tokenizer> Tokenizer::from_dir(const std::string& dir) {
  auto tok = std::unique_ptr<Tokenizer>(new Tokenizer());
  json j = json::parse(read_file(dir + "/tokenizer.json"));

  // GPT-2 byte <-> unicode alphabet.
  {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<bool> direct(256, false);
    for (int b : bs) direct[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
      tok->byte_to_unicode_[b] = utf8::encode(cp);
    }
  }
  std::unordered_map<uint32_t, unsigned char> unicode_to_byte;
  for (int b = 0; b < 256; ++b) {
    size_t i = 0;
    unicode_to_byte[utf8::next(tok->byte_to_unicode_[b], i)] = static_cast<unsigned char>(b);
  }

  // Model.
  const json& model = j.at("model");
  if (model.value("type", "BPE") != "BPE") throw std::runtime_error("only BPE tokenizers are supported");
  const json& dec = j.value("decoder", json());
  auto is_byte_level = [](const json& d) {
    if (d.is_null()) return false;
    if (d.value("type", "") == "ByteLevel") return true;
    if (d.value("type", "") == "Sequence")
      for (const auto& x : d.at("decoders"))
        if (x.value("type", "") == "ByteLevel") return true;
    return false;
  };
  if (!is_byte_level(dec))
    throw std::runtime_error(
        "unsupported tokenizer: only byte-level BPE (GPT-2 / Qwen / Llama-3 style) is supported, "
        "not SentencePiece-style (Llama-2, TinyLlama)");
  tok->ignore_merges_ = model.value("ignore_merges", false);

  TokenId max_id = -1;
  for (auto& [k, v] : model.at("vocab").items()) {
    TokenId id = v.get<TokenId>();
    tok->vocab_[k] = id;
    max_id = std::max(max_id, id);
  }
  int rank = 0;
  for (const auto& m : model.at("merges")) {
    std::string key;
    if (m.is_string())
      key = m.get<std::string>();
    else
      key = m.at(0).get<std::string>() + " " + m.at(1).get<std::string>();
    tok->merge_rank_.emplace(std::move(key), rank++);
  }
  const json added = j.value("added_tokens", json::array());
  for (const auto& a : added) max_id = std::max(max_id, a.at("id").get<TokenId>());

  tok->id_to_bytes_.assign(max_id + 1, std::string());
  tok->special_.assign(max_id + 1, false);
  for (const auto& [s, id] : tok->vocab_) {
    std::string bytes;
    size_t i = 0;
    while (i < s.size()) {
      const size_t start = i;
      uint32_t cp = utf8::next(s, i);
      auto it = unicode_to_byte.find(cp);
      if (it != unicode_to_byte.end())
        bytes += static_cast<char>(it->second);
      else
        bytes.append(s.substr(start, i - start));
    }
    tok->id_to_bytes_[id] = std::move(bytes);
  }
  for (const auto& a : added) {
    AddedToken t{a.at("content").get<std::string>(), a.at("id").get<TokenId>(), a.value("special", false)};
    tok->id_to_bytes_[t.id] = t.content;
    tok->special_[t.id] = t.special;
    tok->added_lookup_[t.content] = t.id;
    tok->added_.push_back(std::move(t));
  }
  std::sort(tok->added_.begin(), tok->added_.end(),
            [](const AddedToken& x, const AddedToken& y) { return x.content.size() > y.content.size(); });

  // Pre-tokenizer.
  std::function<void(const json&)> add_pretok = [&](const json& p) {
    if (p.is_null()) return;
    const std::string type = p.value("type", "");
    if (type == "Sequence") {
      for (const auto& x : p.at("pretokenizers")) add_pretok(x);
    } else if (type == "Split") {
      const json& pat = p.at("pattern");
      std::string re = pat.contains("Regex") ? pat["Regex"].get<std::string>()
                                             : escape_regex(pat.at("String").get<std::string>());
      const std::string behavior = p.value("behavior", "Isolated");
      if (behavior != "Isolated" && behavior != "Removed")
        throw std::runtime_error("unsupported Split behavior: " + behavior);
      tok->splitters_.push_back(std::make_unique<Splitter>(re, behavior == "Removed", p.value("invert", false)));
    } else if (type == "ByteLevel") {
      tok->add_prefix_space_ = p.value("add_prefix_space", false);
      if (p.value("use_regex", true)) tok->splitters_.push_back(std::make_unique<Splitter>(kGpt2Pattern, false, false));
    } else if (type == "Digits") {
      const char* re = p.value("individual_digits", false) ? R"(\p{N})" : R"(\p{N}+)";
      tok->splitters_.push_back(std::make_unique<Splitter>(re, false, false));
    } else {
      throw std::runtime_error("unsupported pre-tokenizer: " + type);
    }
  };
  add_pretok(j.value("pre_tokenizer", json()));

  // BOS from the post-processor (TemplateProcessing), e.g. Llama-3.
  std::function<void(const json&)> find_bos = [&](const json& p) {
    if (p.is_null()) return;
    const std::string type = p.value("type", "");
    if (type == "Sequence") {
      for (const auto& x : p.at("processors")) find_bos(x);
    } else if (type == "TemplateProcessing") {
      const json& single = p.at("single");
      if (!single.empty() && single[0].contains("SpecialToken")) {
        const std::string name = single[0]["SpecialToken"].at("id").get<std::string>();
        const json& st = p.at("special_tokens");
        if (st.contains(name)) tok->bos_id_ = st[name].at("ids").at(0).get<TokenId>();
      }
    }
  };
  find_bos(j.value("post_processor", json()));

  // Chat template string (used only to auto-detect the template family).
  const std::string cfg_path = dir + "/tokenizer_config.json";
  if (file_exists(cfg_path)) {
    json cfg = json::parse(read_file(cfg_path));
    if (cfg.contains("chat_template")) {
      const json& ct = cfg["chat_template"];
      if (ct.is_string()) {
        tok->chat_template_ = ct.get<std::string>();
      } else if (ct.is_array()) {
        for (const auto& t : ct)
          if (t.value("name", "") == "default") tok->chat_template_ = t.value("template", "");
      }
    }
  }
  if (tok->chat_template_.empty() && file_exists(dir + "/chat_template.jinja"))
    tok->chat_template_ = read_file(dir + "/chat_template.jinja");
  return tok;
}

const std::string& Tokenizer::token_bytes(TokenId id) const {
  static const std::string empty;
  if (id < 0 || id >= static_cast<TokenId>(id_to_bytes_.size())) return empty;
  return id_to_bytes_[id];
}

bool Tokenizer::is_special(TokenId id) const {
  return id >= 0 && id < static_cast<TokenId>(special_.size()) && special_[id];
}

std::optional<TokenId> Tokenizer::token_to_id(const std::string& token) const {
  if (auto it = added_lookup_.find(token); it != added_lookup_.end()) return it->second;
  if (auto it = vocab_.find(token); it != vocab_.end()) return it->second;
  return std::nullopt;
}

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool parse_special) const {
  std::vector<TokenId> out;
  size_t seg_start = 0, i = 0;
  while (parse_special && i < text.size()) {
    const AddedToken* hit = nullptr;
    for (const auto& a : added_) {  // longest first
      if (text.compare(i, a.content.size(), a.content) == 0) {
        hit = &a;
        break;
      }
    }
    if (!hit) {
      ++i;
      continue;
    }
    encode_segment(text.substr(seg_start, i - seg_start), out);
    out.push_back(hit->id);
    i += hit->content.size();
    seg_start = i;
  }
  encode_segment(text.substr(seg_start), out);
  return out;
}

void Tokenizer::encode_segment(std::string_view segment, std::vector<TokenId>& out) const {
  if (segment.empty()) return;
  std::string with_prefix;
  if (add_prefix_space_ && segment.front() != ' ') {
    with_prefix = " " + std::string(segment);
    segment = with_prefix;
  }
  std::vector<std::string_view> pieces{segment};
  std::vector<std::string_view> next;
  for (const auto& sp : splitters_) {
    next.clear();
    for (auto p : pieces) sp->split(p, next);
    pieces.swap(next);
  }
  std::string mapped;
  for (auto p : pieces) {
    mapped.clear();
    for (unsigned char c : p) mapped += byte_to_unicode_[c];
    bpe_word(mapped, out);
  }
}

// BPE over one pre-token, O(n log n) with a min-heap keyed by (rank, position),
// which reproduces HF's merge order exactly (lowest rank first, leftmost on ties).
void Tokenizer::bpe_word(const std::string& word, std::vector<TokenId>& out) const {
  if (ignore_merges_) {
    if (auto it = vocab_.find(word); it != vocab_.end()) {
      out.push_back(it->second);
      return;
    }
  }
  struct Sym {
    std::string s;
    int prev, next;
    bool alive;
  };
  std::vector<Sym> syms;
  size_t i = 0;
  while (i < word.size()) {
    const size_t start = i;
    utf8::next(word, i);
    syms.push_back({word.substr(start, i - start), static_cast<int>(syms.size()) - 1,
                    static_cast<int>(syms.size()) + 1, true});
  }
  if (syms.empty()) return;
  syms.back().next = -1;

  auto rank_of = [&](int a, int b) -> int {
    auto it = merge_rank_.find(syms[a].s + " " + syms[b].s);
    return it == merge_rank_.end() ? -1 : it->second;
  };
  using Item = std::tuple<int, int, int>;  // rank, left index, right index
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
  for (int k = 0; k + 1 < static_cast<int>(syms.size()); ++k)
    if (int r = rank_of(k, k + 1); r >= 0) heap.emplace(r, k, k + 1);

  while (!heap.empty()) {
    auto [r, a, b] = heap.top();
    heap.pop();
    if (!syms[a].alive || !syms[b].alive || syms[a].next != b) continue;
    if (rank_of(a, b) != r) continue;  // stale entry
    syms[a].s += syms[b].s;
    syms[b].alive = false;
    syms[a].next = syms[b].next;
    if (syms[b].next >= 0) syms[syms[b].next].prev = a;
    if (syms[a].prev >= 0)
      if (int r2 = rank_of(syms[a].prev, a); r2 >= 0) heap.emplace(r2, syms[a].prev, a);
    if (syms[a].next >= 0)
      if (int r2 = rank_of(a, syms[a].next); r2 >= 0) heap.emplace(r2, a, syms[a].next);
  }
  for (int k = 0; k >= 0; k = syms[k].next) {
    auto it = vocab_.find(syms[k].s);
    if (it != vocab_.end()) {
      out.push_back(it->second);
    } else {
      // Should not happen for byte-level vocabularies; fall back per character.
      size_t p = 0;
      while (p < syms[k].s.size()) {
        const size_t st = p;
        utf8::next(syms[k].s, p);
        if (auto c = vocab_.find(syms[k].s.substr(st, p - st)); c != vocab_.end()) out.push_back(c->second);
      }
    }
  }
}

std::string Tokenizer::decode(std::span<const TokenId> ids, bool skip_special) const {
  std::string bytes;
  for (TokenId id : ids) {
    if (skip_special && is_special(id)) continue;
    bytes += token_bytes(id);
  }
  return utf8::sanitize(bytes);
}

}  // namespace ps

// Byte-level BPE tokenizer that reads a Hugging Face tokenizer.json.
//
// Covers the GPT-2 family of tokenizers: GPT-2, Qwen2/2.5/3, Llama-3.x,
// SmolLM2, Mistral-Nemo and most recent models. SentencePiece-style
// tokenizers (Llama-2, TinyLlama) use a different pre-tokenizer and are
// rejected with a clear error.
//
// Pipeline (same as HF `tokenizers`):
//   1. split out added/special tokens such as <|im_start|>
//   2. pre-tokenize each remaining segment with the model's own regex
//      (run by PCRE2 with Unicode properties, so \p{L}, \p{N} match exactly)
//   3. map bytes to the GPT-2 byte alphabet, then apply BPE merges
// Thread-safe for concurrent encode/decode calls.
#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ps {

using TokenId = int32_t;

class Tokenizer {
 public:
  // Loads <dir>/tokenizer.json and, if present, <dir>/tokenizer_config.json.
  static std::unique_ptr<Tokenizer> from_dir(const std::string& dir);
  ~Tokenizer();

  // Encodes text. Added/special tokens appearing literally in the text are
  // recognised when `parse_special` is true (needed for chat templates).
  std::vector<TokenId> encode(std::string_view text, bool parse_special = true) const;

  // Decodes to UTF-8 text. Special tokens are dropped when `skip_special`.
  std::string decode(std::span<const TokenId> ids, bool skip_special = true) const;

  // Raw bytes of a single token (may be a partial UTF-8 sequence). Empty for
  // ids outside the tokenizer's vocabulary.
  const std::string& token_bytes(TokenId id) const;
  bool is_special(TokenId id) const;
  std::optional<TokenId> token_to_id(const std::string& token) const;
  int vocab_size() const { return static_cast<int>(id_to_bytes_.size()); }

  // Token the HF post-processor would prepend to a raw prompt (e.g. Llama-3's
  // <|begin_of_text|>); empty for tokenizers that add nothing (Qwen).
  std::optional<TokenId> bos_id() const { return bos_id_; }
  const std::string& chat_template_source() const { return chat_template_; }

 private:
  Tokenizer() = default;
  struct Splitter;
  struct AddedToken {
    std::string content;
    TokenId id;
    bool special;
  };

  void encode_segment(std::string_view segment, std::vector<TokenId>& out) const;
  void bpe_word(const std::string& word_bytes, std::vector<TokenId>& out) const;

  std::unordered_map<std::string, TokenId> vocab_;       // byte-level string -> id
  std::unordered_map<std::string, int> merge_rank_;      // "a b" -> rank
  std::vector<std::string> id_to_bytes_;                 // id -> raw bytes
  std::vector<bool> special_;
  std::vector<AddedToken> added_;                        // sorted longest first
  std::unordered_map<std::string, TokenId> added_lookup_;
  std::vector<std::unique_ptr<Splitter>> splitters_;
  bool add_prefix_space_ = false;
  bool ignore_merges_ = false;
  std::optional<TokenId> bos_id_;
  std::string chat_template_;
  std::string byte_to_unicode_[256];
};

}  // namespace ps

// Chat templates: turn OpenAI `messages` into a prompt string.
//
// Hugging Face ships templates as Jinja. Rather than embed a Jinja engine,
// the two families that cover most open models are hard-coded and detected
// from the template source: ChatML (Qwen, SmolLM2) and Llama-3 headers.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace ps {

class Tokenizer;

struct ChatMessage {
  std::string role;
  std::string content;
};

enum class ChatFormat { ChatML, Llama3, Plain };

class ChatTemplate {
 public:
  // name: "auto" | "chatml" | "llama3" | "plain"
  static ChatTemplate create(const std::string& name, const Tokenizer& tok);

  std::string apply(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true) const;
  // Token that ends an assistant turn and must stop generation.
  std::optional<std::string> end_of_turn() const;
  ChatFormat format() const { return format_; }
  const char* name() const;

 private:
  ChatFormat format_ = ChatFormat::Plain;
};

}  // namespace ps

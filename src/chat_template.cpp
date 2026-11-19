#include "chat_template.hpp"

#include <stdexcept>

#include "tokenizer.hpp"

namespace ps {

namespace {
std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}
}  // namespace

ChatTemplate ChatTemplate::create(const std::string& name, const Tokenizer& tok) {
  ChatTemplate t;
  if (name == "chatml") {
    t.format_ = ChatFormat::ChatML;
  } else if (name == "llama3") {
    t.format_ = ChatFormat::Llama3;
  } else if (name == "plain") {
    t.format_ = ChatFormat::Plain;
  } else if (name == "auto") {
    const std::string& src = tok.chat_template_source();
    if (src.find("<|im_start|>") != std::string::npos)
      t.format_ = ChatFormat::ChatML;
    else if (src.find("<|start_header_id|>") != std::string::npos)
      t.format_ = ChatFormat::Llama3;
    else
      t.format_ = ChatFormat::Plain;
  } else {
    throw std::runtime_error("unknown chat template: " + name);
  }
  return t;
}

const char* ChatTemplate::name() const {
  switch (format_) {
    case ChatFormat::ChatML: return "chatml";
    case ChatFormat::Llama3: return "llama3";
    default: return "plain";
  }
}

std::optional<std::string> ChatTemplate::end_of_turn() const {
  switch (format_) {
    case ChatFormat::ChatML: return "<|im_end|>";
    case ChatFormat::Llama3: return "<|eot_id|>";
    default: return std::nullopt;
  }
}

std::string ChatTemplate::apply(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
  std::string out;
  switch (format_) {
    case ChatFormat::ChatML:
      for (const auto& m : messages) out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
      if (add_generation_prompt) out += "<|im_start|>assistant\n";
      break;
    case ChatFormat::Llama3:
      out += "<|begin_of_text|>";
      for (const auto& m : messages)
        out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + trim(m.content) + "<|eot_id|>";
      if (add_generation_prompt) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
      break;
    case ChatFormat::Plain:
      for (const auto& m : messages) out += m.role + ": " + m.content + "\n";
      if (add_generation_prompt) out += "assistant:";
      break;
  }
  return out;
}

}  // namespace ps

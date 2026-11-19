#include "openai_api.hpp"

#include <atomic>
#include <chrono>
#include <random>

#include "chat_template.hpp"
#include "engine.hpp"

namespace ps {

namespace {

std::string random_id(const char* prefix) {
  static std::atomic<uint64_t> counter{0};
  thread_local std::mt19937_64 rng(std::random_device{}());
  char buf[64];
  snprintf(buf, sizeof(buf), "%s-%012llx%04llx", prefix, static_cast<unsigned long long>(rng() & 0xFFFFFFFFFFFFull),
           static_cast<unsigned long long>(counter++ & 0xFFFF));
  return buf;
}

[[noreturn]] void fail(const std::string& msg, int status = 400) { throw ApiError{status, msg}; }

std::string message_text(const json& content) {
  if (content.is_string()) return content.get<std::string>();
  if (content.is_null()) return "";
  if (content.is_array()) {  // [{"type": "text", "text": "..."}]
    std::string out;
    for (const auto& part : content) {
      if (part.value("type", "") != "text") fail("only text content parts are supported");
      out += part.value("text", "");
    }
    return out;
  }
  fail("message content must be a string or an array of text parts");
}

}  // namespace

ApiRequest parse_request(const json& body, bool chat, const Engine& engine) {
  if (!body.is_object()) fail("request body must be a JSON object");
  ApiRequest r;
  r.chat = chat;
  r.req = std::make_shared<Request>();
  r.req->id = random_id(chat ? "chatcmpl" : "cmpl");
  r.model = body.value("model", engine.model_name());
  r.created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                  .count();
  const Tokenizer& tok = engine.tokenizer();

  if (chat) {
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
      fail("'messages' must be a non-empty array");
    std::vector<ChatMessage> msgs;
    for (const auto& m : body["messages"]) {
      if (!m.is_object() || !m.contains("role")) fail("each message needs a 'role'");
      msgs.push_back({m["role"].get<std::string>(), message_text(m.value("content", json()))});
    }
    r.req->prompt = tok.encode(engine.chat_template().apply(msgs, true));
  } else {
    if (!body.contains("prompt")) fail("'prompt' is required");
    const json& p = body["prompt"];
    if (p.is_string()) {
      if (auto bos = tok.bos_id()) r.req->prompt.push_back(*bos);
      auto ids = tok.encode(p.get<std::string>());
      r.req->prompt.insert(r.req->prompt.end(), ids.begin(), ids.end());
    } else if (p.is_array() && !p.empty() && p[0].is_number_integer()) {
      r.req->prompt = p.get<std::vector<TokenId>>();  // pre-tokenized prompt
    } else {
      fail("'prompt' must be a string or an array of token ids");
    }
  }
  const int prompt_len = static_cast<int>(r.req->prompt.size());
  const int max_len = engine.max_model_len();
  if (prompt_len == 0) fail("prompt is empty");
  for (TokenId t : r.req->prompt)
    if (t < 0 || t >= engine.model_config().vocab_size) fail("token id out of range");

  SamplingParams& sp = r.req->params;
  auto num = [&](const char* k, double def) {
    if (!body.contains(k) || body[k].is_null()) return def;
    if (!body[k].is_number()) fail(std::string("'") + k + "' must be a number");
    return body[k].get<double>();
  };
  sp.temperature = static_cast<float>(num("temperature", 1.0));
  sp.top_p = static_cast<float>(num("top_p", 1.0));
  sp.top_k = static_cast<int>(num("top_k", -1));
  sp.n = static_cast<int>(num("n", 1));
  sp.ignore_eos = body.value("ignore_eos", false);
  if (sp.temperature < 0 || sp.temperature > 2) fail("'temperature' must be in [0, 2]");
  if (sp.top_p <= 0 || sp.top_p > 1) fail("'top_p' must be in (0, 1]");
  if (sp.n < 1 || sp.n > 16) fail("'n' must be between 1 and 16");
  const double requested = body.contains("max_completion_tokens") ? num("max_completion_tokens", -1)
                                                                   : num("max_tokens", -1);
  if (requested == -1) {
    sp.max_tokens = max_len - prompt_len;
  } else {
    if (requested < 1) fail("'max_tokens' must be at least 1");
    sp.max_tokens = static_cast<int>(requested);
  }
  if (prompt_len + sp.max_tokens > max_len || prompt_len >= max_len)
    fail("This model's maximum context length is " + std::to_string(max_len) + " tokens. However, you requested " +
         std::to_string(prompt_len + sp.max_tokens) + " tokens (" + std::to_string(prompt_len) + " in the prompt, " +
         std::to_string(sp.max_tokens) + " for the completion).");
  if (body.contains("stop") && !body["stop"].is_null()) {
    const json& s = body["stop"];
    if (s.is_string()) {
      sp.stop.push_back(s.get<std::string>());
    } else if (s.is_array()) {
      for (const auto& x : s) sp.stop.push_back(x.get<std::string>());
    } else {
      fail("'stop' must be a string or array of strings");
    }
    if (sp.stop.size() > 4) fail("at most 4 stop sequences are allowed");
  }
  r.stream = body.value("stream", false);
  if (body.contains("stream_options") && body["stream_options"].is_object())
    r.include_usage = body["stream_options"].value("include_usage", false);
  return r;
}

json error_body(const ApiError& e) {
  return {{"error", {{"message", e.message}, {"type", e.type}, {"param", nullptr}, {"code", nullptr}}}};
}

json usage_json(const Usage& u) {
  return {{"prompt_tokens", u.prompt_tokens},
          {"completion_tokens", u.completion_tokens},
          {"total_tokens", u.prompt_tokens + u.completion_tokens},
          {"prompt_tokens_details", {{"cached_tokens", u.cached_tokens}}}};
}

json stream_chunk(const ApiRequest& r, int index, const std::string* text, const std::string* finish_reason,
                  bool role_header) {
  json choice = {{"index", index}, {"logprobs", nullptr},
                 {"finish_reason", finish_reason ? json(*finish_reason) : json(nullptr)}};
  if (r.chat) {
    json delta = json::object();
    if (role_header) delta["role"] = "assistant";
    if (text) delta["content"] = *text;
    if (role_header && !text) delta["content"] = "";
    choice["delta"] = delta;
  } else {
    choice["text"] = text ? *text : "";
  }
  return {{"id", r.req->id},
          {"object", r.chat ? "chat.completion.chunk" : "text_completion"},
          {"created", r.created},
          {"model", r.model},
          {"choices", json::array({choice})}};
}

json full_response(const ApiRequest& r, const std::vector<std::string>& texts,
                   const std::vector<std::string>& finish_reasons, const Usage& usage) {
  json choices = json::array();
  for (size_t i = 0; i < texts.size(); ++i) {
    json c = {{"index", i}, {"logprobs", nullptr}, {"finish_reason", finish_reasons[i]}};
    if (r.chat)
      c["message"] = {{"role", "assistant"}, {"content", texts[i]}};
    else
      c["text"] = texts[i];
    choices.push_back(c);
  }
  return {{"id", r.req->id},
          {"object", r.chat ? "chat.completion" : "text_completion"},
          {"created", r.created},
          {"model", r.model},
          {"choices", choices},
          {"usage", usage_json(usage)}};
}

}  // namespace ps

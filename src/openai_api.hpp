// OpenAI-compatible request parsing and response formatting.
//
// Supported: POST /v1/chat/completions, POST /v1/completions with
// model, messages|prompt, max_tokens|max_completion_tokens, temperature,
// top_p, top_k (extension), n, stop, stream, stream_options.include_usage,
// ignore_eos (extension, for benchmarks). Unknown fields are ignored.
#pragma once

#include <memory>
#include <string>

#include "nlohmann/json.hpp"
#include "sequence.hpp"

namespace ps {

class Engine;
using json = nlohmann::json;

struct ApiError {
  int status = 400;
  std::string message;
  std::string type = "invalid_request_error";
};

struct ApiRequest {
  std::shared_ptr<Request> req;
  bool chat = true;
  bool stream = false;
  bool include_usage = false;
  std::string model;
  int64_t created = 0;
};

// Throws ApiError on invalid input. Tokenization happens here, on the HTTP
// thread, so the engine thread never spends time on it.
ApiRequest parse_request(const json& body, bool chat, const Engine& engine);

json error_body(const ApiError& e);
json usage_json(const Usage& u);

// One SSE `data:` payload for a streamed event.
json stream_chunk(const ApiRequest& r, int index, const std::string* text, const std::string* finish_reason,
                  bool role_header);

// Full non-streaming response body.
json full_response(const ApiRequest& r, const std::vector<std::string>& texts,
                   const std::vector<std::string>& finish_reasons, const Usage& usage);

}  // namespace ps

// pagedserve: OpenAI-compatible LLM server.
//
//   pagedserve --model ./Qwen2.5-0.5B-Instruct --port 8000
//
// HTTP threads (cpp-httplib pool) parse and tokenize requests, hand them to
// the engine thread, then stream tokens back as Server-Sent Events as they
// arrive on the request's channel. A dropped connection aborts the request so
// its KV blocks are freed on the next engine step.
#include <csignal>
#include <cstdio>
#include <iostream>

#include "cli.hpp"
#include "engine.hpp"
#include "httplib/httplib.h"
#include "openai_api.hpp"

using namespace ps;

namespace {

httplib::Server* g_server = nullptr;

void on_signal(int) {
  if (g_server) g_server->stop();
}

void send_json(httplib::Response& res, const json& j, int status = 200) {
  res.status = status;
  res.set_content(j.dump(), "application/json");
}

bool write_sse(httplib::DataSink& sink, const std::string& payload) {
  const std::string msg = "data: " + payload + "\n\n";
  return sink.write(msg.data(), msg.size());
}

void handle_completion(Engine& engine, const httplib::Request& hreq, httplib::Response& res, bool chat) {
  ApiRequest api;
  try {
    api = parse_request(json::parse(hreq.body), chat, engine);
    engine.add_request(api.req);
  } catch (const ApiError& e) {
    return send_json(res, error_body(e), e.status);
  } catch (const json::exception& e) {
    return send_json(res, error_body({400, std::string("invalid JSON: ") + e.what()}), 400);
  } catch (const std::exception& e) {
    return send_json(res, error_body({400, e.what()}), 400);
  }
  auto req = api.req;
  const int n = req->params.n;

  if (!api.stream) {
    std::vector<std::string> texts(n), finish(n, "stop");
    bool done = false;
    while (!done) {
      for (auto& e : req->channel->pop(std::chrono::milliseconds(100), done)) {
        texts[e.index] += e.text;
        if (e.finish_reason) finish[e.index] = *e.finish_reason;
      }
      if (!done && hreq.is_connection_closed()) {
        req->aborted = true;
        engine.abort_request(req->id);
        return;
      }
    }
    return send_json(res, full_response(api, texts, finish, req->channel->usage()));
  }

  res.set_header("Cache-Control", "no-cache");
  res.set_header("X-Accel-Buffering", "no");
  auto api_ptr = std::make_shared<ApiRequest>(api);
  res.set_chunked_content_provider(
      "text/event-stream",
      [api_ptr, &engine](size_t, httplib::DataSink& sink) {
        const ApiRequest& a = *api_ptr;
        auto& r = *a.req;
        auto abort = [&] {
          r.aborted = true;
          engine.abort_request(r.id);
          return false;
        };
        if (a.chat)  // OpenAI sends the role first, once per choice
          for (int i = 0; i < r.params.n; ++i)
            if (!write_sse(sink, stream_chunk(a, i, nullptr, nullptr, true).dump())) return abort();
        bool done = false;
        while (!done) {
          auto events = r.channel->pop(std::chrono::milliseconds(50), done);
          if (events.empty() && !done) {
            if (!sink.is_writable()) return abort();  // client went away while we wait
            continue;
          }
          for (const auto& e : events) {
            if (!e.text.empty() && !write_sse(sink, stream_chunk(a, e.index, &e.text, nullptr, false).dump()))
              return abort();
            if (e.finish_reason &&
                !write_sse(sink, stream_chunk(a, e.index, nullptr, &*e.finish_reason, false).dump()))
              return abort();
          }
        }
        if (a.include_usage) {
          json u = {{"id", r.id},
                    {"object", a.chat ? "chat.completion.chunk" : "text_completion"},
                    {"created", a.created},
                    {"model", a.model},
                    {"choices", json::array()},
                    {"usage", usage_json(r.channel->usage())}};
          if (!write_sse(sink, u.dump())) return false;
        }
        write_sse(sink, "[DONE]");
        sink.done();
        return true;
      },
      [api_ptr, &engine](bool success) {
        // Connection closed before we finished: make sure the engine stops.
        if (!success && !api_ptr->req->aborted) {
          api_ptr->req->aborted = true;
          engine.abort_request(api_ptr->req->id);
        }
      });
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  Args args(argc, argv);
  if (args.flag("help") || argc == 1) {
    std::cout << "usage: pagedserve --model DIR [--host 0.0.0.0] [--port 8000] [--http-threads 512]\n\n"
              << kEngineHelp;
    return 0;
  }
  try {
    const std::string host = args.str("host", "0.0.0.0");
    const int port = args.i("port", 8000);
    const int http_threads = args.i("http-threads", 512);
    EngineConfig cfg = parse_engine_args(args);
    args.check_unused();

    Engine engine(cfg);
    std::cerr << engine.describe();
    engine.start();

    httplib::Server server;
    g_server = &server;
    // Each open stream holds a thread while it waits for tokens, so the pool
    // grows on demand up to http_threads.
    server.new_task_queue = [http_threads] { return new httplib::ThreadPool(16, http_threads); };
    server.set_write_timeout(30, 0);

    server.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& r) {
      handle_completion(engine, q, r, true);
    });
    server.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& r) {
      handle_completion(engine, q, r, false);
    });
    server.Get("/v1/models", [&](const httplib::Request&, httplib::Response& r) {
      send_json(r, {{"object", "list"},
                    {"data", json::array({{{"id", engine.model_name()},
                                            {"object", "model"},
                                            {"owned_by", "pagedserve"},
                                            {"max_model_len", engine.max_model_len()}}})}});
    });
    server.Get("/health", [](const httplib::Request&, httplib::Response& r) { r.set_content("ok", "text/plain"); });
    server.Get("/stats", [&](const httplib::Request&, httplib::Response& r) {
      const EngineStats s = engine.stats();
      send_json(r, {{"num_running", s.num_running},
                    {"num_waiting", s.num_waiting},
                    {"kv_blocks_total", s.kv_blocks_total},
                    {"kv_blocks_free", s.kv_blocks_free},
                    {"kv_blocks_cached", s.kv_blocks_cached},
                    {"kv_blocks_in_use", s.kv_blocks_total - s.kv_blocks_free - s.kv_blocks_cached},
                    {"prefix_query_tokens", s.prefix_query_tokens},
                    {"prefix_hit_tokens", s.prefix_hit_tokens},
                    {"prefix_hit_rate", s.prefix_query_tokens ? double(s.prefix_hit_tokens) / s.prefix_query_tokens : 0.0},
                    {"num_preemptions", s.num_preemptions},
                    {"num_cow_copies", s.num_cow_copies},
                    {"num_evictions", s.num_evictions},
                    {"steps", s.steps},
                    {"prefill_tokens", s.prefill_tokens},
                    {"generated_tokens", s.generated_tokens},
                    {"requests_finished", s.requests_finished},
                    {"busy_seconds", s.busy_seconds}});
    });

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::cerr << "listening on http://" << host << ":" << port << "\n";
    if (!server.listen(host, port)) {
      std::cerr << "failed to listen on " << host << ":" << port << "\n";
      return 1;
    }
    engine.stop();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

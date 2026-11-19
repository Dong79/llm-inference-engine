#include "config.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace ps {

using json = nlohmann::json;

namespace {
json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return json::parse(ss.str());
}

void add_ids(const json& v, std::vector<int>& out) {
  if (v.is_number_integer()) out.push_back(v.get<int>());
  if (v.is_array())
    for (const auto& x : v)
      if (x.is_number_integer()) out.push_back(x.get<int>());
}

void parse_rope(const json& r, ModelConfig& c) {
  if (!r.is_object()) return;
  if (r.contains("rope_theta")) c.rope_theta = r["rope_theta"].get<double>();
  std::string type = r.value("rope_type", r.value("type", std::string("default")));
  if (type == "default") return;
  if (type != "llama3" && type != "linear")
    throw std::runtime_error("unsupported rope scaling type: " + type + " (supported: llama3, linear)");
  c.rope_scaling.type = type;
  c.rope_scaling.factor = r.value("factor", 1.0);
  c.rope_scaling.low_freq_factor = r.value("low_freq_factor", 1.0);
  c.rope_scaling.high_freq_factor = r.value("high_freq_factor", 4.0);
  c.rope_scaling.original_max_position_embeddings = r.value("original_max_position_embeddings", 8192);
}
}  // namespace

ModelConfig ModelConfig::from_dir(const std::string& dir) {
  json j = read_json(dir + "/config.json");
  ModelConfig c;
  if (j.contains("architectures") && !j["architectures"].empty())
    c.architecture = j["architectures"][0].get<std::string>();
  static const char* kSupported[] = {"LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM",
                                     "Qwen3ForCausalLM"};
  if (std::find(std::begin(kSupported), std::end(kSupported), c.architecture) == std::end(kSupported))
    throw std::runtime_error("unsupported architecture '" + c.architecture +
                             "' (supported: Llama, Mistral, Qwen2, Qwen3)");
  c.vocab_size = j.at("vocab_size").get<int>();
  c.hidden_size = j.at("hidden_size").get<int>();
  c.intermediate_size = j.at("intermediate_size").get<int>();
  c.num_layers = j.at("num_hidden_layers").get<int>();
  c.num_heads = j.at("num_attention_heads").get<int>();
  c.num_kv_heads = j.value("num_key_value_heads", c.num_heads);
  c.head_dim = (j.contains("head_dim") && !j["head_dim"].is_null()) ? j["head_dim"].get<int>()
                                                                     : c.hidden_size / c.num_heads;
  c.rms_norm_eps = j.value("rms_norm_eps", 1e-6);
  c.rope_theta = j.value("rope_theta", 10000.0);
  c.max_position_embeddings = j.value("max_position_embeddings", 4096);
  c.tie_word_embeddings = j.value("tie_word_embeddings", false);
  // Older configs use rope_scaling; transformers v5 uses rope_parameters.
  if (j.contains("rope_scaling")) parse_rope(j["rope_scaling"], c);
  if (j.contains("rope_parameters")) parse_rope(j["rope_parameters"], c);
  c.dtype = j.value("torch_dtype", j.value("dtype", std::string("float32")));
  if (j.contains("eos_token_id")) add_ids(j["eos_token_id"], c.eos_token_ids);
  if (std::ifstream(dir + "/generation_config.json").good()) {
    json g = read_json(dir + "/generation_config.json");
    if (g.contains("eos_token_id")) add_ids(g["eos_token_id"], c.eos_token_ids);
  }
  std::sort(c.eos_token_ids.begin(), c.eos_token_ids.end());
  c.eos_token_ids.erase(std::unique(c.eos_token_ids.begin(), c.eos_token_ids.end()), c.eos_token_ids.end());
  if (c.num_heads % c.num_kv_heads != 0) throw std::runtime_error("num_heads must be a multiple of num_kv_heads");
  return c;
}

}  // namespace ps

// pagedserve-tokenize: encode/decode a JSON array of strings with the C++
// tokenizer. Used by tests/test_tokenizer.py to check parity with HF.
//
//   pagedserve-tokenize <model_or_tokenizer_dir> < strings.json
//   -> {"ids": [[...]], "decoded": [...], "bos": <id|null>, "vocab_size": N}
#include <iostream>
#include <iterator>

#include "nlohmann/json.hpp"
#include "tokenizer.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <tokenizer_dir> < strings.json\n";
    return 2;
  }
  try {
    auto tok = ps::Tokenizer::from_dir(argv[1]);
    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    auto strings = nlohmann::json::parse(input);
    nlohmann::json out;
    out["ids"] = nlohmann::json::array();
    out["decoded"] = nlohmann::json::array();
    for (const auto& s : strings) {
      auto ids = tok->encode(s.get<std::string>());
      out["ids"].push_back(ids);
      out["decoded"].push_back(tok->decode(ids));
    }
    out["bos"] = tok->bos_id() ? nlohmann::json(*tok->bos_id()) : nlohmann::json(nullptr);
    out["vocab_size"] = tok->vocab_size();
    std::cout << out.dump() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

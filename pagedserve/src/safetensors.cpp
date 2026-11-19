#include "safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace ps {

using json = nlohmann::json;

namespace {

torch::ScalarType parse_dtype(const std::string& s) {
  if (s == "F32") return torch::kFloat32;
  if (s == "F16") return torch::kFloat16;
  if (s == "BF16") return torch::kBFloat16;
  if (s == "F64") return torch::kFloat64;
  if (s == "I64") return torch::kInt64;
  if (s == "I32") return torch::kInt32;
  if (s == "I16") return torch::kInt16;
  if (s == "I8") return torch::kInt8;
  if (s == "U8") return torch::kUInt8;
  if (s == "BOOL") return torch::kBool;
  throw std::runtime_error("unsupported safetensors dtype " + s);
}

// RAII read-only mmap of a whole file.
struct MappedFile {
  void* data = MAP_FAILED;
  size_t size = 0;
  explicit MappedFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    ::fstat(fd, &st);
    size = static_cast<size_t>(st.st_size);
    data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (data == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
  }
  ~MappedFile() {
    if (data != MAP_FAILED) ::munmap(data, size);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
};

void load_file(const std::string& path, torch::Device device, torch::ScalarType float_dtype, WeightMap& out) {
  MappedFile mf(path);
  if (mf.size < 8) throw std::runtime_error("truncated safetensors file " + path);
  uint64_t header_len = 0;
  std::memcpy(&header_len, mf.data, 8);  // little-endian on all supported hosts
  if (8 + header_len > mf.size) throw std::runtime_error("bad safetensors header in " + path);
  const char* base = static_cast<const char*>(mf.data);
  json header = json::parse(std::string(base + 8, header_len));
  const char* payload = base + 8 + header_len;
  for (auto& [name, info] : header.items()) {
    if (name == "__metadata__") continue;
    const auto dtype = parse_dtype(info.at("dtype").get<std::string>());
    std::vector<int64_t> shape = info.at("shape").get<std::vector<int64_t>>();
    const auto offs = info.at("data_offsets").get<std::vector<uint64_t>>();
    void* ptr = const_cast<char*>(payload + offs[0]);
    torch::Tensor t = torch::from_blob(ptr, shape, torch::TensorOptions().dtype(dtype));
    const auto target = torch::isFloatingType(dtype) ? float_dtype : dtype;
    // .to() is a no-op when nothing changes, so clone to detach from the mmap.
    torch::Tensor owned = t.to(device, target);
    if (owned.data_ptr() == t.data_ptr()) owned = owned.clone();
    out[name] = owned;
  }
}

}  // namespace

WeightMap load_safetensors_dir(const std::string& dir, torch::Device device, torch::ScalarType float_dtype) {
  WeightMap out;
  const std::string index_path = dir + "/model.safetensors.index.json";
  std::ifstream idx(index_path);
  if (idx) {
    std::stringstream ss;
    ss << idx.rdbuf();
    json j = json::parse(ss.str());
    std::set<std::string> files;
    for (auto& [k, v] : j.at("weight_map").items()) files.insert(v.get<std::string>());
    for (const auto& f : files) load_file(dir + "/" + f, device, float_dtype, out);
  } else {
    load_file(dir + "/model.safetensors", device, float_dtype, out);
  }
  return out;
}

}  // namespace ps

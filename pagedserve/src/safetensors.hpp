// Minimal safetensors reader.
//
// Format: 8-byte little-endian header length N, then N bytes of JSON
// {"name": {"dtype": "BF16", "shape": [...], "data_offsets": [begin, end]}, ...},
// then the raw tensor bytes. Files are mmapped and each tensor is copied to
// the target device/dtype, so the mapping can be dropped afterwards.
#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>

namespace ps {

using WeightMap = std::unordered_map<std::string, torch::Tensor>;

// Loads model.safetensors, or every shard listed in
// model.safetensors.index.json. Floating-point tensors are converted to
// `float_dtype`; integer tensors keep their type.
WeightMap load_safetensors_dir(const std::string& dir, torch::Device device, torch::ScalarType float_dtype);

}  // namespace ps

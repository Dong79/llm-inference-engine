cmake_minimum_required(VERSION 3.22)
project(pagedserve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PAGEDSERVE_CUDA "Build the custom CUDA paged-attention decode kernel (needs nvcc)" OFF)
set(TORCH_ROOT "" CACHE PATH "LibTorch root (has include/, lib/, share/cmake). Default: the Python torch package")

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)

if(NOT TORCH_ROOT)
  execute_process(
    COMMAND python3 -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT EXISTS "${TORCH_ROOT}/include/torch")
  message(FATAL_ERROR "LibTorch not found. Pass -DTORCH_ROOT=/path/to/libtorch or pip install torch.")
endif()
message(STATUS "LibTorch: ${TORCH_ROOT}")

if(PAGEDSERVE_CUDA)
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
  endif()
endif()

# A CUDA build of LibTorch insists on finding a CUDA toolkit in its CMake
# config. Without a toolkit (e.g. a CPU-only dev box with the pip wheel) we
# link the libraries directly instead.
find_package(CUDAToolkit QUIET)
add_library(torch_deps INTERFACE)
if(EXISTS "${TORCH_ROOT}/lib/libtorch_cuda.so" AND NOT CUDAToolkit_FOUND)
  message(STATUS "No CUDA toolkit: linking LibTorch manually (GPU still usable through LibTorch ops)")
  execute_process(
    COMMAND python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(TORCH_CXX11_ABI STREQUAL "")
    set(TORCH_CXX11_ABI 1)
  endif()
  target_include_directories(torch_deps SYSTEM INTERFACE
    "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")
  target_compile_definitions(torch_deps INTERFACE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
  target_link_directories(torch_deps INTERFACE "${TORCH_ROOT}/lib")
  # --no-as-needed keeps libtorch_cuda loaded so its CUDA kernels register.
  target_link_libraries(torch_deps INTERFACE
    -Wl,--no-as-needed torch torch_cuda torch_cpu c10 -Wl,--as-needed)
  target_link_options(torch_deps INTERFACE "-Wl,-rpath,${TORCH_ROOT}/lib")
else()
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_ROOT}/share/cmake")
  find_package(Torch REQUIRED)
  target_link_libraries(torch_deps INTERFACE ${TORCH_LIBRARIES})
  target_include_directories(torch_deps SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
  target_link_options(torch_deps INTERFACE "-Wl,-rpath,${TORCH_ROOT}/lib")
endif()

# Tokenizer + block manager have no Torch dependency.
add_library(pagedserve_light STATIC
  src/block_manager.cpp
  src/tokenizer.cpp
  src/chat_template.cpp)
target_include_directories(pagedserve_light PUBLIC src third_party)
target_link_libraries(pagedserve_light PUBLIC PkgConfig::PCRE2)
target_compile_options(pagedserve_light PRIVATE -Wall -Wextra)

add_library(pagedserve_core STATIC
  src/config.cpp
  src/safetensors.cpp
  src/kv_cache.cpp
  src/attention.cpp
  src/model.cpp
  src/sampler.cpp
  src/scheduler.cpp
  src/engine.cpp)
target_link_libraries(pagedserve_core PUBLIC pagedserve_light torch_deps Threads::Threads)
target_compile_options(pagedserve_core PRIVATE -Wall -Wextra -Wno-unused-parameter)

if(PAGEDSERVE_CUDA)
  target_sources(pagedserve_core PRIVATE src/cuda/paged_attention.cu)
  target_compile_definitions(pagedserve_core PUBLIC PAGEDSERVE_WITH_CUDA)
  target_link_libraries(pagedserve_core PUBLIC CUDA::cudart)
  set_target_properties(pagedserve_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

add_executable(pagedserve src/server_main.cpp src/openai_api.cpp)
target_link_libraries(pagedserve PRIVATE pagedserve_core)

add_executable(pagedserve-dev tools/devtool.cpp)
target_link_libraries(pagedserve-dev PRIVATE pagedserve_core)

add_executable(pagedserve-tokenize tools/tokenize_main.cpp)
target_link_libraries(pagedserve-tokenize PRIVATE pagedserve_light)

enable_testing()
add_executable(test_block_manager tests/test_block_manager.cpp)
target_link_libraries(test_block_manager PRIVATE pagedserve_light)
add_test(NAME block_manager COMMAND test_block_manager)

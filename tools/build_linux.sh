#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$repo_dir" -B "$repo_dir/build" -DCMAKE_BUILD_TYPE=Release \
  -DPHYSX_ROOT="$repo_dir/deps/PhysX/physx" \
  -DPHYSX_LINUX_LIB_DIR="$repo_dir/deps/physx-sdk/bin/linux.x86_64/release" \
  -DORT_ROOT="$repo_dir/deps/onnxruntime" \
  -DFETCHCONTENT_SOURCE_DIR_GLAD="$repo_dir/deps/glad" \
  -DFETCHCONTENT_SOURCE_DIR_GLFW="$repo_dir/deps/glfw" \
  -DFETCHCONTENT_SOURCE_DIR_GLM="$repo_dir/deps/glm" \
  -DFETCHCONTENT_SOURCE_DIR_PUGIXML="$repo_dir/deps/pugixml" \
  -DFETCHCONTENT_SOURCE_DIR_YAML-CPP="$repo_dir/deps/yaml-cpp"
cmake --build "$repo_dir/build" -j "${BUILD_JOBS:-6}"
ctest --test-dir "$repo_dir/build" --output-on-failure

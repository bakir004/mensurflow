#!/usr/bin/env bash
# Linux/macOS shortcut: configure a Release build, run correctness tests, and run benchmarks.
# OpenCL headers and the ICD loader are fetched automatically when not installed.
set -euo pipefail

# Resolve paths relative to this script, not the caller's working directory.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$project_dir/build-auto"

# Configuration detects available toolchains before the build target is executed.
cmake -S "$project_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --config Release --parallel
ctest --test-dir "$build_dir" --output-on-failure
cmake --build "$build_dir" --config Release --target run_benchmarks

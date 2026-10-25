#!/usr/bin/env bash
# Linux/macOS shortcut: configure a Release build and run both benchmarks.
# Build all targets and run CTest separately for the complete correctness suite.
set -euo pipefail

# Resolve paths relative to this script, not the caller's working directory.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$project_dir/build-auto"

# Configuration detects available toolchains before the build target is executed.
cmake -S "$project_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --config Release --target run_benchmarks --parallel

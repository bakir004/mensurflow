#!/usr/bin/env bash
# Linux/macOS shortcut: configure a Release build and run both API benchmarks.
# Correctness tests are separate: build all targets, then run ctest in build-auto.
# Exit on errors, unset variables, or failing commands within a pipeline.
set -euo pipefail

# Resolve paths relative to this script, not the caller's working directory.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$project_dir/build-auto"

# Configuration detects optional GPU toolchains and generates build instructions.
cmake -S "$project_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
# Build benchmark dependencies in parallel, then execute the run_benchmarks target.
cmake --build "$build_dir" --config Release --target run_benchmarks --parallel

#!/usr/bin/env bash
# Linux/macOS shortcut: configure a Release build, run correctness tests, and run benchmarks.
# OpenCL headers and the ICD loader are fetched automatically when not installed.
set -euo pipefail

# Resolve paths relative to this script, not the caller's working directory.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$project_dir/build-auto"

# Install the tracked formatter hook for this clone, but preserve custom hooks.
# Source archives without Git metadata can still build normally.
if [[ -e "$project_dir/.git" ]]; then
    hooks_path="$(git -C "$project_dir" config --get core.hooksPath || true)"
    if [[ -z "$hooks_path" || "$hooks_path" == ".githooks" ]]; then
        git -C "$project_dir" config --local core.hooksPath .githooks
        echo "Git pre-commit formatting hook enabled (.githooks)."
    else
        echo "Keeping custom core.hooksPath: $hooks_path; integrate .githooks/pre-commit manually."
    fi
fi

# Configuration detects available toolchains before the build target is executed.
# Prefer Ninja for faster incremental builds, but keep an existing build directory's generator.
generator=()
if [[ ! -f "$build_dir/CMakeCache.txt" ]] && command -v ninja >/dev/null; then
    generator=(-G Ninja)
fi
cmake -S "$project_dir" -B "$build_dir" ${generator[@]+"${generator[@]}"} -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --config Release --parallel
ctest --test-dir "$build_dir" --output-on-failure
cmake --build "$build_dir" --config Release --target run_benchmarks

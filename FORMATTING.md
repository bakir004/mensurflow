# Code Formatting

Code is formatted automatically with `clang-format` by a git pre-commit hook, so formatting is the same for everyone regardless of IDE. The style is defined in `.clang-format`.

## Requirements

1. **Git** – on Windows, install Git for Windows, which includes the Bash needed by the hook.
2. **clang-format** – installed and available on your `PATH`.

No Python, `pre-commit` package, or VS Code extension is required. Use the same clang-format version across the team for consistent results.

### Installing clang-format

- Windows: `winget install LLVM.LLVM` (make sure LLVM's `bin` directory is on `PATH`)
- Ubuntu/Debian: `sudo apt install clang-format`
- macOS: `brew install clang-format`

Verify:

```
git --version
clang-format --version
```

## Enabling the hook

After pulling the repo, run the build script once to set up the git hook:

- `build.sh` (Linux/macOS/Git Bash), or
- `build.bat` (Windows)

Alternatively, enable the hook without building:

```
git config --local core.hooksPath .githooks
```

From then on, staged code is formatted automatically whenever you commit.

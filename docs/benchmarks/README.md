# Benchmark results: one file per person per run

Everyone who runs the benchmarks adds **one file here**, so results from different machines sit side
by side and nobody edits the same big file at the same time.

- [`docs/BENCHMARKING.md`](../BENCHMARKING.md): how the benchmarks are written (the rules).
- [`docs/BENCHMARKS.md`](../BENCHMARKS.md): the shared summary and comparisons between machines.
- **This folder**: your own result file, one per person, CPU, and day.
- `benchmarks/` (in the repository root) is the benchmark **source code**, not results.

Filled-in examples: [`asja-i7-10510u-2026-11-01.md`](asja-i7-10510u-2026-11-01.md) (a laptop) and
[`asja-ryzen9-5900x-2026-11-01.md`](asja-ryzen9-5900x-2026-11-01.md) (a desktop with an NVIDIA GPU).

## Quick start

### 1. Get the code
```
git switch main
git pull
```

### 2. Build, CPU only, Release. Pick the line that matches your setup.

All three build **without** `MENSURFLOW_NATIVE` (so every machine uses the same instruction set) and
without the GPU backends (so the tables only contain the CPU). Use a short folder name such as `bench`:
Windows limits path lengths and long build folder names can break the build.

**Windows, Visual Studio (MSVC)** (use `"Visual Studio 17 2022"` for 2022):
```
cmake -S . -B bench -G "Visual Studio 18 2026" -A x64 -DMENSURFLOW_CUDA=OFF -DMENSURFLOW_NATIVE=OFF
cmake --build bench --config Release
ctest --test-dir bench -C Release
```
The programs are in `bench\Release\`.

**Windows, MinGW from MSYS2: use the UCRT64 compiler**, not the older `mingw64` one (its `fmaf` is not
fused, which fails the `axpy` tests). Install it once in an "MSYS2 UCRT64" terminal with
`pacman -S mingw-w64-ucrt-x86_64-gcc`. Then, in Git Bash, assuming MSYS2 is in `C:\msys64`:
```
export PATH=/c/msys64/ucrt64/bin:/c/msys64/mingw64/bin:$PATH
cmake -S . -B bench -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe \
      -DCMAKE_MAKE_PROGRAM=C:/msys64/mingw64/bin/mingw32-make.exe \
      -DMENSURFLOW_CUDA=OFF -DMENSURFLOW_NATIVE=OFF
cmake --build bench -j 8
ctest --test-dir bench
```
The programs are in `bench\`. Keep the `export PATH` line in every new terminal: the programs need the
DLLs in `ucrt64\bin`.

**Linux and macOS** (not tested by us yet; tell us what you had to change):
```
cmake -S . -B bench -DCMAKE_BUILD_TYPE=Release -DMENSURFLOW_CUDA=OFF -DMENSURFLOW_ROCM=OFF \
      -DMENSURFLOW_METAL=OFF -DMENSURFLOW_NATIVE=OFF
cmake --build bench -j 8
ctest --test-dir bench
```
Apple's compiler has no OpenMP; install one (for example `brew install libomp`) or the parallel rows will
run serially. The header of the output says `OpenMP: yes` or `no`.

All 14 tests must pass before you measure.

### 3. Get the machine quiet
Plugged in, High performance power plan, browsers and other heavy programs closed. Antivirus: pause it
for the runs, or exclude the `bench` folder. Do not touch the machine while it runs.

### 4. Run: twice, with a pause
Use **Git Bash** (or Linux and macOS terminals), and the path of your build (`bench/` or `bench/Release/`):
```
./bench/cpu_parallel_benchmark.exe > run1.txt     # no .exe on Linux and macOS
sleep 180                                         # let the CPU cool down: 3 minutes
./bench/cpu_parallel_benchmark.exe > run2.txt
sleep 180
./bench/matmul_benchmark.exe > matmul.txt
```
Each run takes a few minutes. To try it quickly first, name some operations:
`./bench/cpu_parallel_benchmark.exe neg relu`.

In **Windows PowerShell** write the output with `| Out-File -Encoding ascii run1.txt`, not `>`:
older PowerShell versions write `>` files as UTF-16, which `awk` cannot read.

Open `run1.txt` and check the header: `build: Release`, your CPU, `OpenMP: yes`. If it says `NOT Release`,
you built the wrong configuration.

### 5. Make the tables (in Git Bash; `awk` ships with it, but not with PowerShell)
```
awk -f docs/benchmarks/compare_runs.awk run1.txt run2.txt                             # how noisy
awk -v mode=simd -f docs/benchmarks/summarize_runs.awk run1.txt run2.txt              # SIMD table
awk -v mode=threads -v size=100000 -f docs/benchmarks/summarize_runs.awk run1.txt run2.txt
awk -v mode=threads -v size=10000000 -f docs/benchmarks/summarize_runs.awk run1.txt run2.txt
awk -v mode=matmul -f docs/benchmarks/summarize_runs.awk matmul.txt                   # matmul table
```
Each prints a markdown table (or the stability summary) ready to paste into your result file.

### 6. Write your result file
1. Copy [`TEMPLATE.md`](TEMPLATE.md) to `<yourname>-<cpu>-<YYYY-MM-DD>.md`: your name, then the CPU in a
   short lowercase form (the model without the brand, spaces replaced by hyphens), then the date. For
   example `asja-i7-10510u-2026-11-01.md` or `asja-ryzen9-5900x-2026-11-01.md`. One file per person, CPU,
   and day.
2. Fill in every field: who and when, machine, build, conditions, what you ran, the tables from step 5,
   stability, observations, problems. Write `unknown` if you could not find out.
3. Keep it short: summary tables, not the 1,500-line raw output. If you want to keep the raw output,
   save it next to your file as `<same name>-run1.txt` (optional).
4. Commit your file and push it.

## Getting the machine details

The benchmark header already prints the CPU model, thread counts, compiler, OpenMP version, AVX2, and
whether the build is Release. Copy that block. What it does not print:

**Windows (PowerShell)**
```
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed, SMBIOSMemoryType
(Get-CimInstance Win32_OperatingSystem).Caption
powercfg /getactivescheme
(Get-CimInstance Win32_Battery).BatteryStatus
```
RAM `Capacity` is in bytes and `Speed` in MHz; list every stick (a mismatched pair runs at the slower
speed). `BatteryStatus` is `2` on AC power.

**Linux**
```
lscpu
free -h
cat /etc/os-release
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor     # should be "performance"
```
RAM speed: `sudo dmidecode -t memory | grep -i speed`.

**macOS**
```
sysctl -n machdep.cpu.brand_string
sysctl hw.physicalcpu hw.logicalcpu hw.memsize
sw_vers
pmset -g batt
```

## When something goes wrong

- **`An Application Control policy has blocked this file`**, or `ctest` says `Not Run` for a test: Windows
  blocked a freshly built `.exe`. Delete that `.exe` and build again (`cmake --build bench`); a rebuilt
  file is usually allowed. Do not turn protection off. Antivirus can cause the same.
- **A row where max is several times the median**: the machine was busy. Close more programs and rerun.
- **Scalar rows equal to sequential rows**: the scalar baseline is vectorized anyway. This is known to
  happen with MSVC (it ignores the flag we use). Say which compiler you used and treat the `vs scalar`
  column as not valid; use `vs seq`.
- **2- and 4-thread rows differ a lot between your two runs**: thread placement. See BENCHMARKING.md; the
  1-thread and all-thread rows are the reliable ones.
- **A failed check** ("result differs from the reference"): do not report numbers. Report the failure.
- **Compile error**: copy the first error message into your result file under "Problems" and tell the team.

# Benchmark results: one file per person per run

Everyone who runs the benchmarks adds **one file here**, so results from different machines sit side
by side and nobody edits the same big file at the same time.

- [`docs/BENCHMARKING.md`](../BENCHMARKING.md): how the benchmarks are written and run (the rules).
- [`docs/BENCHMARKS.md`](../BENCHMARKS.md): the shared summary and comparisons between machines.
- **This folder**: your own result file (form below), one per person and run.
- `benchmarks/` (in the repository root) is the benchmark **source code**, not results.

## Adding your result

1. Build and run the benchmarks as described in [BENCHMARKING.md](../BENCHMARKING.md): Release build,
   no `MENSURFLOW_NATIVE`, plugged in, High performance, other programs closed, a 3-minute pause
   before each run, and run `cpu_parallel_benchmark` **twice**.
2. Copy [`TEMPLATE.md`](TEMPLATE.md) to `<yourname>-<machine>-<YYYY-MM-DD>.md`, for example
   `aldin-nitro-2026-11-03.md`. One file per machine and day.
3. Fill in every field of the form: who and when, the machine, the build, the conditions, what you ran,
   the results, and how stable the two runs were. Leave nothing blank; write `unknown` if you could not
   find out. A filled-in example: [`asja-laptop-2026-11-01.md`](asja-laptop-2026-11-01.md).
4. Paste your two runs into files and compare them to measure the noise:

   ```
   awk -f docs/benchmarks/compare_runs.awk run1.txt run2.txt
   ```

   (`awk` comes with Git Bash on Windows, and with Linux and macOS.) Paste its output into the
   "Stability" part of the form.
5. Keep the result file short. Put a **summary table** for a few sizes, not the 1,500-line raw output.
   If you want to keep the raw output, save it next to your file as `<same name>-run1.txt` (optional).
6. Commit only your own file (and the raw outputs, if any), open a pull request, and add a one-line
   entry to the list in BENCHMARKS.md if your machine is new.

## Getting the machine details

The benchmark header already prints the CPU model, thread counts, compiler, OpenMP version, AVX2, and
whether the build is Release. Copy that block. What it does not print:

**Windows (PowerShell)**
```
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed, SMBIOSMemoryType
(Get-CimInstance Win32_OperatingSystem).Caption
powercfg /getactivescheme
```
RAM `Capacity` is in bytes and `Speed` in MHz; list every stick (a mismatched pair runs at the slower
speed). The same PowerShell commands show whether you are on AC: `(Get-CimInstance Win32_Battery).BatteryStatus` is `2` on AC.

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

## What goes wrong, and how to tell

- **A row where max is several times the median**: the machine was busy. Close more programs, rerun.
- **Scalar rows equal to sequential rows**: the scalar baseline is vectorized anyway. This is known to
  happen with MSVC (it ignores the flag we use). Say which compiler you used, and treat the scalar rows
  as not valid for MSVC.
- **2- and 4-thread rows differ a lot between your two runs**: thread placement. See the notes in
  BENCHMARKING.md; 1-thread and all-thread rows are the reliable ones.
- **A failed check** ("result differs from the reference"): do not report numbers. Report the failure.

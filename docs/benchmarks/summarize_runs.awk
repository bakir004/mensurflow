# Turns the raw output of the benchmarks into the markdown tables of the result form
# (docs/benchmarks/TEMPLATE.md), so nobody has to copy rows by hand.
#
#   SIMD table, scalar vs sequential at n = 1,000,000 (two runs of cpu_parallel_benchmark):
#     awk -v mode=simd -f docs/benchmarks/summarize_runs.awk run1.txt run2.txt
#
#   Thread table at one size (default 100000; the form uses 100000 and 10000000):
#     awk -v mode=threads -v size=10000000 -f docs/benchmarks/summarize_runs.awk run1.txt run2.txt
#
#   Matrix multiplication, GFLOP/s per size (one run of matmul_benchmark):
#     awk -v mode=matmul -f docs/benchmarks/summarize_runs.awk matmul.txt
#
# Cells read "run 1 / run 2". Speedup = sequential time / parallel time. "all threads" is the largest
# thread count in the output; "best" is the fastest speedup among the counts of 2 and more. Pass a
# different size with -v size=... and a different SIMD size with -v simd_size=...
#
# Reads the standard table printed by the benchmark harness: n impl threads median ms ... in
# columns 1-4, and the rate in the column four from the end.

FNR == 1 { file++ }
/^== / {
    op = $2
    if (!(op in seen)) { seen[op] = 1; order[++ops] = op }
    next
}
/^ +[0-9]+ +(scalar|sequential|parallel) / {
    n = $1; impl = $2; thr = (impl == "parallel") ? $3 : "-"
    median[file, op, n, impl, thr] = $4
    rate[file, op, n, impl, thr] = $(NF - 3)
    if (impl == "parallel") threads_seen[$3] = 1
    if (file > runs) runs = file
}

function speedup(f, o, n, t,    s, p) {
    s = median[f, o, n, "sequential", "-"]; p = median[f, o, n, "parallel", t]
    return (s > 0 && p > 0) ? s / p : 0
}
function cell(a, b, fmt) { return (runs > 1) ? sprintf(fmt " / " fmt, a, b) : sprintf(fmt, a) }

END {
    if (mode == "matmul") {
        print "| n | scalar | sequential | " threads_header() " |"
        # one run: GFLOP/s of every implementation per size
        ncols = 0
        for (t in threads_seen) tlist[++ncols] = t + 0
        asort_numeric(tlist, ncols)
        line = "|---|---|---|"; for (i = 1; i <= ncols; i++) line = line "---|"; print line
        for (key in median) { split(key, p, SUBSEP); sizes[p[3]] = 1 }
        nsz = 0; for (s in sizes) szlist[++nsz] = s + 0
        asort_numeric(szlist, nsz)
        for (i = 1; i <= nsz; i++) {
            n = szlist[i]; o = order[1]
            row = "| " n " | " rate[1, o, n, "scalar", "-"] " | " rate[1, o, n, "sequential", "-"] " |"
            for (j = 1; j <= ncols; j++) row = row " " rate[1, o, n, "parallel", tlist[j]] " |"
            print row
        }
        exit
    }
    if (mode == "simd") {
        n = (simd_size == "") ? 1000000 : simd_size
        for (i = 1; i <= ops; i++) {
            o = order[i]
            printf "| %s | %s | %s | %s |\n", o,
                cell(median[1, o, n, "scalar", "-"], median[2, o, n, "scalar", "-"], "%s"),
                cell(median[1, o, n, "sequential", "-"], median[2, o, n, "sequential", "-"], "%s"),
                cell(median[1, o, n, "scalar", "-"] / median[1, o, n, "sequential", "-"],
                     median[2, o, n, "scalar", "-"] / median[2, o, n, "sequential", "-"], "%.1fx")
        }
        exit
    }
    # mode threads
    n = (size == "") ? 100000 : size
    maxt = 0; for (t in threads_seen) if (t + 0 > maxt) maxt = t + 0
    for (i = 1; i <= ops; i++) {
        o = order[i]
        for (f = 1; f <= runs; f++) {
            best[f] = 0
            for (t in threads_seen) if (t + 0 >= 2) { s = speedup(f, o, n, t); if (s > best[f]) best[f] = s }
        }
        printf "| %s | %s | %s | %s | %s | %s |\n", o, n,
            cell(median[1, o, n, "sequential", "-"], median[2, o, n, "sequential", "-"], "%s"),
            cell(speedup(1, o, n, 1), speedup(2, o, n, 1), "%.2fx"),
            cell(speedup(1, o, n, maxt), speedup(2, o, n, maxt), "%.2fx"),
            cell(best[1], best[2], "%.2fx")
    }
}

function threads_header(    t, c, list, i, s) {
    c = 0; for (t in threads_seen) list[++c] = t + 0
    asort_numeric(list, c)
    s = ""
    for (i = 1; i <= c; i++) s = s (i > 1 ? " | " : "") list[i] (list[i] == 1 ? " thread" : " threads")
    return s
}

# Sorts the first n entries of arr as numbers (insertion sort; mawk and BSD awk lack asort).
function asort_numeric(arr, n,    i, j, v) {
    for (i = 2; i <= n; i++) {
        v = arr[i]
        for (j = i - 1; j >= 1 && arr[j] > v; j--) arr[j + 1] = arr[j]
        arr[j + 1] = v
    }
}

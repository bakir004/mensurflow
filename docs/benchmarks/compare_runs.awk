# Measures how much two runs of cpu_parallel_benchmark (or matmul_benchmark) disagree.
#
#   awk -f docs/benchmarks/compare_runs.awk run1.txt run2.txt
#
# For every row (op, size, implementation, thread count) it compares the median time of the two
# runs, and prints how many rows differ by more than 10 % and by more than 25 %, per implementation
# and per thread count, and whether run 2 was mostly slower or faster. Rows under 1 microsecond are
# skipped (too short to compare). A machine that was quiet has few rows over 10 %. Many rows where
# run 2 is slower mean the CPU heated up between the runs; many for 2 and 4 threads mean thread
# placement changed.
#
# Reads the median column ($4) of the standard table printed by the benchmark harness.

FNR == 1 { file++ }
/^== / { op = $2; next }
/^ +[0-9]+ +(scalar|sequential|parallel) / {
    key = op "|" $1 "|" $2 "|" $3
    if (file == 1) first[key] = $4; else second[key] = $4
}
END {
    for (k in first) {
        if (!(k in second)) continue
        if (first[k] * 1e3 < 1 || second[k] * 1e3 < 1) { skipped++; continue }
        split(k, p, "|")
        group = (p[3] == "parallel") ? "parallel t=" p[4] : p[3]
        total[group]++
        ratio = second[k] / first[k]
        diff = (ratio > 1) ? ratio - 1 : 1 / ratio - 1
        if (diff > 0.10) over10[group]++
        if (diff > 0.25) over25[group]++
        if (ratio > 1.10) slower[group]++
        if (ratio < 1 / 1.10) faster[group]++
    }
    printf "%-14s %6s %9s %9s %12s %12s\n", "group", "rows", ">10%", ">25%", "run2 slower", "run2 faster"
    for (g in total) {
        printf "%-14s %6d %9d %9d %12d %12d\n", g, total[g], over10[g] + 0, over25[g] + 0, slower[g] + 0, faster[g] + 0 | "sort"
    }
    close("sort")
    printf "(%d rows under 1 microsecond skipped)\n", skipped + 0
}

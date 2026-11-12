#!/usr/bin/env python3
"""Pair two native random_indexing_benchmark logs into a Markdown reference comparison.

The reference itself is native LibTorch in random_indexing_benchmark.cpp. This reporting script
uses only Python's standard library; it does not time Python or require NumPy/PyTorch imports.
"""

import argparse
from decimal import Decimal, DecimalException
from pathlib import Path
import re


IMPLEMENTATIONS = ("mensurflow", "pytorch")
SECTION = re.compile(r"^== (.+) ==$")
SHAPE = re.compile(r"[1-9]\d*(?:x[1-9]\d*)*\Z")


def read_log(path):
    """Read median/min/max milliseconds by (operation, shape, threads, implementation)."""
    operation = None
    results = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        section = SECTION.fullmatch(line)
        if section:
            operation = section.group(1)
            continue
        fields = line.split()
        if len(fields) < 2 or fields[1] not in IMPLEMENTATIONS:
            continue
        location = f"{path}:{line_number}"
        if operation is None or len(fields) != 10 or not SHAPE.fullmatch(fields[0]):
            raise ValueError(f"{location}: expected a ten-column native comparison row")
        try:
            threads = int(fields[2])
            times = tuple(Decimal(value) for value in fields[3:6])
        except (ValueError, DecimalException) as error:
            raise ValueError(f"{location}: invalid thread count or timing") from error
        median, minimum, maximum = times
        if threads <= 0 or any(not value.is_finite() or value <= 0 for value in times):
            raise ValueError(f"{location}: expected positive thread count and finite timings")
        if not minimum <= median <= maximum:
            raise ValueError(f"{location}: expected min <= median <= max")
        key = operation, fields[0], threads
        row = results.setdefault(key, {})
        if fields[1] in row:
            raise ValueError(f"{location}: duplicate {fields[1]} row for {key}")
        row[fields[1]] = times
    if not results:
        raise ValueError(f"{path}: no native reference comparison rows found")
    for key, row in results.items():
        missing = set(IMPLEMENTATIONS) - row.keys()
        if missing:
            raise ValueError(f"{path}: missing {', '.join(sorted(missing))} for {key}")
    return results


def report(first, second):
    if first.keys() != second.keys():
        raise ValueError("runs must contain identical operation/shape/thread-count rows")
    lines = [
        "All times are milliseconds; paired values are run 1 / run 2.",
        "`torch/ours > 1` means MensurFlow was faster. Allocation and validation are timed on both sides.",
        "Random streams differ; distribution bounds/moments and dropout formulas are checked independently.",
        "Embedding backward shape describes its upstream gradient; its output is 512×32.",
        "",
        "| op | shape | threads | ours median | ours min | torch median | torch min | torch/ours | max drift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    flagged = 0
    for key, first_row in first.items():
        second_row = second[key]
        ours_first, ours_second = (row["mensurflow"] for row in (first_row, second_row))
        torch_first, torch_second = (row["pytorch"] for row in (first_row, second_row))
        drift = max(abs(a[0] - b[0]) / min(a[0], b[0]) * 100
                    for a, b in ((ours_first, ours_second), (torch_first, torch_second)))
        drift_text = f"{drift:.1f}%"
        if drift > 10:
            flagged += 1
            drift_text = f"**{drift_text}**"
        operation, shape, threads = key
        lines.append(
            f"| {operation} | {shape} | {threads} | "
            f"{ours_first[0]:.6g} / {ours_second[0]:.6g} | "
            f"{ours_first[1]:.6g} / {ours_second[1]:.6g} | "
            f"{torch_first[0]:.6g} / {torch_second[0]:.6g} | "
            f"{torch_first[1]:.6g} / {torch_second[1]:.6g} | "
            f"{torch_first[0] / ours_first[0]:.2f}x / "
            f"{torch_second[0] / ours_second[0]:.2f}x | {drift_text} |"
        )
    lines.extend(("", f"Rows exceeding 10% median timing drift: **{flagged} / {len(first)}**. "
                  "Raw logs retain maximum times and throughput for every row."))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_one", type=Path)
    parser.add_argument("run_two", type=Path)
    arguments = parser.parse_args()
    try:
        print(report(read_log(arguments.run_one), read_log(arguments.run_two)))
    except (OSError, ValueError, DecimalException) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two native activation benchmark logs and print a Markdown report."""

import argparse
from decimal import Decimal, DecimalException
from pathlib import Path
import re


IMPLEMENTATIONS = ("mensurflow", "pytorch")
SECTION = re.compile(r"^== (.+) ==$")
INTEGER = re.compile(r"[+-]?\d+\Z")
DRIFT_WARNING = Decimal("10")


def describe(key):
    operation, size, threads = key
    return f"{operation}, n={size}, threads={threads}"


def read_log(path):
    """Return paired median times indexed by (operation, size, thread count)."""
    results = {}
    operation = None
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        section = SECTION.fullmatch(line)
        if section:
            operation = section.group(1)
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        # Header notes may begin with numbers (e.g. "8 bytes/element" or "500 ms").
        # Only a named implementation can be a timing row before the first section.
        if operation is None and fields[1] not in IMPLEMENTATIONS:
            continue
        # Ignore metadata and column headings, but diagnose malformed data rows.
        if not INTEGER.fullmatch(fields[0]) and fields[1] not in IMPLEMENTATIONS:
            continue
        location = f"{path}:{line_number}"
        if operation is None:
            raise ValueError(f"{location}: timing row has no operation section")
        if len(fields) != 9 or fields[1] not in IMPLEMENTATIONS:
            raise ValueError(f"{location}: expected a nine-column native timing row")
        try:
            size = int(fields[0])
            threads = int(fields[2])
            median = Decimal(fields[3])
        except (ValueError, DecimalException) as error:
            raise ValueError(f"{location}: invalid size, thread count, or median") from error
        if size <= 0 or threads <= 0:
            raise ValueError(f"{location}: size and thread count must be positive")
        if not median.is_finite() or median <= 0:
            raise ValueError(f"{location}: median must be positive and finite")
        key = (operation, size, threads)
        row = results.setdefault(key, {})
        implementation = fields[1]
        if implementation in row:
            raise ValueError(f"{location}: duplicate {implementation} row for {describe(key)}")
        row[implementation] = median
    if not results:
        raise ValueError(f"{path}: no native activation timing rows found")
    for key, row in results.items():
        missing = set(IMPLEMENTATIONS) - row.keys()
        if missing:
            raise ValueError(f"{path}: missing {', '.join(sorted(missing))} counterpart for "
                             f"{describe(key)}")
    return results


def timing_drift(first, second):
    return abs(first - second) / min(first, second) * 100


def report(first, second):
    if first.keys() != second.keys():
        differences = []
        for label, missing in (("run 2", first.keys() - second.keys()),
                               ("run 1", second.keys() - first.keys())):
            if missing:
                differences.append(f"missing from {label}: " +
                                   "; ".join(describe(key) for key in sorted(missing)))
        raise ValueError("log keys do not match; " + " | ".join(differences))

    lines = [
        "All times are medians in milliseconds; paired values are run 1 / run 2.",
        "`torch/ours > 1` means MensurFlow was faster for the measured calls.",
        "Drift is `abs(a-b) / min(a,b) * 100`, taking the larger drift of the two libraries.",
        "",
        "| op | n | threads | ours median ms | torch median ms | torch/ours | max timing drift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    flagged = 0
    largest_drift = Decimal(0)
    for key, run_one in first.items():
        run_two = second[key]
        own_one, own_two = (run["mensurflow"] for run in (run_one, run_two))
        torch_one, torch_two = (run["pytorch"] for run in (run_one, run_two))
        drift = max(timing_drift(own_one, own_two), timing_drift(torch_one, torch_two))
        largest_drift = max(largest_drift, drift)
        drift_text = f"{drift:.2f}%"
        if drift > DRIFT_WARNING:
            flagged += 1
            drift_text = f"**{drift_text}**"
        operation, size, threads = key
        operation = operation.replace("|", "\\|")
        lines.append(f"| {operation} | {size} | {threads} | {own_one:.6g} / {own_two:.6g} | "
                     f"{torch_one:.6g} / {torch_two:.6g} | "
                     f"{torch_one / own_one:.2f}x / {torch_two / own_two:.2f}x | {drift_text} |")
    lines.extend(("", f"Compared {len(first)} rows. Rows exceeding 10% timing drift: "
                  f"**{flagged} / {len(first)}** (bold drift cells); "
                  f"maximum drift: {largest_drift:.2f}%."))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_one", type=Path, help="First activation_torch_benchmark log")
    parser.add_argument("run_two", type=Path, help="Second log with the same operations/sizes/threads")
    args = parser.parse_args()
    try:
        output = report(read_log(args.run_one), read_log(args.run_two))
    except (OSError, ValueError, DecimalException) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()

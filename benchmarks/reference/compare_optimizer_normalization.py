"""Joins the output of optimizer_normalization_benchmark (ours) and optimizer_normalization.py (torch)
into one Markdown table of ours / reference ratios, per op, size and thread count.

    python benchmarks/reference/compare_optimizer_normalization.py ours.txt torch.txt [ours2.txt torch2.txt]

With two pairs of logs (two runs) both medians are shown as "run1 / run2". The ratio is the
reference's median time divided by ours, so above 1.00 means we are faster.
"""

import re
import sys


def parse(path):
    rows, section = {}, None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.match(r"^(?:== (.+?) ==|-- (.+))$", line)
        if m:
            section = (m.group(1) or m.group(2)).split(" [")[0].split(",")[0].strip()
            if section.startswith("public API"):
                section = "public API"
            continue
        # Ours is whitespace-separated ("n impl threads median ..."); torch's uses "|" between cells.
        m = re.match(r"^\s*(\S+)\s+(?:sequential|parallel)\s+(\d+)\s+([\d.]+)", line)
        if m:
            rows.setdefault((section, m.group(1), "ours", int(m.group(2))), float(m.group(3)))
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 7 and re.match(r"^\d+$", cells[2]):
            rows.setdefault((section, cells[0], cells[1], int(cells[2])), float(cells[3]))
    return rows


def main():
    logs = sys.argv[1:]
    pairs = [(parse(logs[i]), parse(logs[i + 1])) for i in range(0, len(logs), 2)]
    print("| op | shape | torch impl | threads | ours ms | torch ms | torch / ours |")
    print("|---|---|---|---:|---:|---:|---:|")
    ours0, torch0 = pairs[0]
    for key, _ in sorted(torch0.items(), key=lambda kv: list(torch0).index(kv[0])):
        section, label, impl, threads = key
        ours_key = (section, label, "ours", threads)
        if ours_key not in ours0:
            continue
        o = [p[0][ours_key] for p in pairs]
        t = [p[1][key] for p in pairs]
        ratio = [tv / ov for tv, ov in zip(t, o)]
        print(f"| {section} | {label} | {impl} | {threads} | {' / '.join(f'{v:.4f}' for v in o)} | "
              f"{' / '.join(f'{v:.4f}' for v in t)} | {' / '.join(f'{v:.2f}x' for v in ratio)} |")


main()

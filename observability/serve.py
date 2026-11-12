#!/usr/bin/env python3
"""Serves the observability dashboard and runs the `observe` binary for it. Standard library only.

    python3 observability/serve.py                 # finds build/observability/observe
    python3 observability/serve.py --bin path/to/observe --port 8765

Then open http://localhost:8765. Requests are handled one at a time, so two benchmarks never run at
the same time. Every result (kernel runs, sweeps, call breakdowns, program traces) is stored in
observability/results/observe.db (SQLite, zlib-compressed JSON; not committed), so the page restores
them after a reload or a server restart.
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATABASE = HERE / "results" / "observe.db"
OLD_HISTORY = HERE / "results" / "history.jsonl"  # earlier text format, imported once
CANDIDATES = ["build/observability/observe", "build-trace/observability/observe",
              "build/observability/Release/observe.exe", "build/observability/observe.exe"]


def find_binary(explicit):
    if explicit:
        return Path(explicit).resolve()
    for candidate in CANDIDATES:
        path = ROOT / candidate
        if path.exists():
            return path
    sys.exit("observe binary not found; build it (cmake --build build --target observe) or pass --bin")


class Runner:
    def __init__(self, binary):
        self.binary = binary
        # Built next to observe from the same sources without auto-vectorization (see CMakeLists.txt).
        self.scalar = binary.with_name(binary.name.replace("observe", "observe_scalar", 1))
        self.listing = None

    def call(self, args, threads=None, scalar=False):
        env = dict(os.environ)
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
        binary = self.scalar if scalar else self.binary
        if not binary.exists():
            return {"fatal": f"{binary.name} not built (cmake --build build --target {binary.stem})"}
        result = subprocess.run([str(binary), *args], capture_output=True, text=True, env=env,
                                timeout=1800)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"fatal": f"observe exited with {result.returncode}: "
                             f"{result.stderr.strip() or result.stdout.strip()[:500]}"}


def demo_source():
    """The demo program's code from observe.cpp (between the demo markers), dedented, with its line numbers."""
    lines = (HERE / "observe.cpp").read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if "// demo:begin" in line) + 1
        end = next(i for i, line in enumerate(lines) if "// demo:end" in line)
    except StopIteration:
        return None
    body = lines[start:end]
    indent = min((len(line) - len(line.lstrip()) for line in body if line.strip()), default=0)
    return {"file": "observability/observe.cpp", "first_line": start + 1,
            "lines": [line[indent:] for line in body]}


DECLARATION = re.compile(r"^(?:Tensor|std::vector<Tensor>|DropoutResult)\s+(\w+)\(")


def op_docs():
    """Each public op's description: the comment right above its declaration in include/mensurflow/ops.

    Returns {name: {"summary", "text", "signature", "file", "line"}}. The second overload of a reduction
    (the one taking `int axis`) is listed as "sum(axis)" and so on, matching the runner's names.
    """
    docs = {}
    for path in sorted((ROOT / "include" / "mensurflow" / "ops").glob("*.hpp")):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            match = DECLARATION.match(line)
            if not match:
                continue
            comment = []
            j = i - 1
            while j >= 0 and lines[j].strip().startswith("//"):
                comment.insert(0, lines[j].strip()[2:].strip())
                j -= 1
            declaration = line
            k = i
            while ";" not in lines[k] and k + 1 < len(lines):
                k += 1
                declaration += " " + lines[k].strip()
            signature = re.sub(r"\s+", " ", declaration.split(";")[0])
            signature = re.sub(r",?\s*Execution execution = Execution::Auto", "", signature)
            name = match.group(1)
            if name in docs:
                if "int axis" not in signature:
                    continue  # another overload (split by sizes); the first one is shown
                name += "(axis)"
            text = " ".join(comment)
            summary = re.split(r"(?<=\.)\s+(?=[A-Z])", text, maxsplit=1)[0] if text else ""
            docs[name] = {"summary": summary, "text": text, "signature": signature,
                          "file": str(path.relative_to(ROOT)), "line": i + 1}
    return docs


class Store:
    """Results in one SQLite file. kind is "run" (kernels and sweeps), "call" or "program"."""

    def __init__(self, path):
        path.parent.mkdir(exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, time REAL, kind TEXT,"
                        " op TEXT, device TEXT, payload BLOB)")
        self.db.commit()
        if OLD_HISTORY.exists():
            for line in OLD_HISTORY.read_text().splitlines():
                if line.strip():
                    self.add("run", json.loads(line))
            OLD_HISTORY.rename(OLD_HISTORY.with_suffix(".jsonl.imported"))

    def add(self, kind, result):
        result.setdefault("time", time.time())
        blob = zlib.compress(json.dumps(result).encode())
        self.db.execute("INSERT INTO results (time, kind, op, device, payload) VALUES (?, ?, ?, ?, ?)",
                        (result["time"], kind, result.get("op"), result.get("device"), blob))
        self.db.commit()

    def all(self, kind):
        rows = self.db.execute("SELECT payload FROM results WHERE kind = ? ORDER BY id", (kind,))
        return [json.loads(zlib.decompress(row[0])) for row in rows]

    def latest(self, kind):
        row = self.db.execute("SELECT payload FROM results WHERE kind = ? ORDER BY id DESC LIMIT 1",
                              (kind,)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def latest_per_op(self, kind):
        rows = self.db.execute("SELECT op, payload FROM results WHERE id IN"
                               " (SELECT MAX(id) FROM results WHERE kind = ? GROUP BY op)", (kind,))
        return {op: json.loads(zlib.decompress(payload)) for op, payload in rows}

    def clear(self):
        self.db.execute("DELETE FROM results")
        self.db.commit()
        self.db.execute("VACUUM")


def make_handler(runner, store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("  " + fmt % args + "\n")

        def send(self, status, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/list"):
                if runner.listing is None or "refresh" in self.path:
                    runner.listing = runner.call(["list"])
                    runner.listing["scalar_build"] = runner.scalar.exists()
                docs = op_docs()  # read fresh, so edited header comments show up on reload
                for op in runner.listing.get("ops", []):
                    op["doc"] = docs.get(op["name"])
                self.send(200, runner.listing)
            elif self.path == "/api/history":
                self.send(200, store.all("run"))
            elif self.path == "/api/latest":
                # Each op's last call breakdown and the last program trace, so a reload brings them back.
                self.send(200, {"calls": store.latest_per_op("call"), "program": store.latest("program")})
            else:
                self.send(404, {"fatal": "not found"})

        def do_POST(self):
            request = self.body()
            threads = request.get("threads") or None
            if self.path == "/api/run":
                args = ["run", "--op", request["op"], "--device", request["device"],
                        "--execution", request.get("execution") or "default",
                        "--min-time", str(request.get("min_time", 0.25))]
                if request.get("sizes"):
                    args += ["--sizes", ",".join(str(int(s)) for s in request["sizes"])]
                variant = request.get("variant")  # None (devices view), "scalar", "simd", "threaded"
                result = runner.call(args, threads, scalar=variant == "scalar")
                result["time"] = time.time()
                result["threads_requested"] = threads
                result["variant"] = variant
                result["tab"] = request.get("tab") or "kernels"  # which tab ran it (kernels or sweep)
                result["col"] = request.get("col")  # the dashboard column (device or CPU version)
                if "fatal" not in result:
                    store.add("run", result)
                self.send(200, result)
            elif self.path == "/api/call":
                args = ["call", "--op", request["op"], "--device", request["device"],
                        "--execution", request.get("execution") or "default",
                        "--calls", str(int(request.get("calls", 25)))]
                if request.get("size"):
                    args += ["--size", str(int(request["size"]))]
                result = runner.call(args, threads)
                if "fatal" not in result:
                    store.add("call", result)
                self.send(200, result)
            elif self.path == "/api/program":
                args = ["program", "--device", request.get("device", "cpu:0"),
                        "--execution", request.get("execution") or "default",
                        "--iterations", str(request.get("iterations", 3))]
                result = runner.call(args, threads)
                result["source"] = demo_source()
                if "fatal" not in result:
                    store.add("program", result)
                self.send(200, result)
            elif self.path == "/api/history/clear":
                store.clear()
                self.send(200, {"ok": True})
            else:
                self.send(404, {"fatal": "not found"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", help="path to the observe binary")
    parser.add_argument("--port", type=int, default=8765)
    options = parser.parse_args()
    runner = Runner(find_binary(options.bin))
    server = HTTPServer(("127.0.0.1", options.port), make_handler(runner, Store(DATABASE)))
    print(f"observe binary: {runner.binary}")
    print(f"dashboard: http://localhost:{options.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

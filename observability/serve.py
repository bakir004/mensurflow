#!/usr/bin/env python3
"""Serves the observability dashboard and runs the `observe` binary for it. Standard library only.

    python3 observability/serve.py                 # finds build/observability/observe
    python3 observability/serve.py --bin path/to/observe --port 8765

Then open http://localhost:8765. Requests are handled one at a time, so two benchmarks never run at
the same time. Every run is appended to observability/results/history.jsonl (not committed).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY = HERE / "results" / "history.jsonl"
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
        self.listing = None

    def call(self, args, threads=None):
        env = dict(os.environ)
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
        result = subprocess.run([str(self.binary), *args], capture_output=True, text=True, env=env,
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


def record(entry):
    HISTORY.parent.mkdir(exist_ok=True)
    with HISTORY.open("a") as file:
        file.write(json.dumps(entry) + "\n")


def make_handler(runner):
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
                self.send(200, runner.listing)
            elif self.path == "/api/history":
                lines = HISTORY.read_text().splitlines() if HISTORY.exists() else []
                self.send(200, [json.loads(line) for line in lines if line.strip()])
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
                result = runner.call(args, threads)
                result["time"] = time.time()
                result["threads_requested"] = threads
                if "fatal" not in result:
                    record(result)
                self.send(200, result)
            elif self.path == "/api/call":
                args = ["call", "--op", request["op"], "--device", request["device"],
                        "--execution", request.get("execution") or "default",
                        "--calls", str(int(request.get("calls", 25)))]
                if request.get("size"):
                    args += ["--size", str(int(request["size"]))]
                self.send(200, runner.call(args, threads))
            elif self.path == "/api/program":
                args = ["program", "--device", request.get("device", "cpu:0"),
                        "--execution", request.get("execution") or "default",
                        "--iterations", str(request.get("iterations", 3))]
                result = runner.call(args, threads)
                result["source"] = demo_source()
                self.send(200, result)
            elif self.path == "/api/history/clear":
                if HISTORY.exists():
                    HISTORY.unlink()
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
    server = HTTPServer(("127.0.0.1", options.port), make_handler(runner))
    print(f"observe binary: {runner.binary}")
    print(f"dashboard: http://localhost:{options.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run a minimal fault demo around the existing smoke test client.

Flow:
1. run one clean request against the live swarm
2. kill one non-bootstrap peer by exact port/block-range match
3. run one more request with a hard timeout
4. save both outcomes and the killed PID metadata
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

from model_presets import safe_model_tag


BEFORE_PROMPT = "The future of robotics is"
AFTER_PROMPT = "When a peer disappears from a private swarm,"
TIMESTAMP_RE = re.compile(r"^\[(?P<ts>[^\]]+)\] (?P<message>.*)$")


def ensure_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def list_candidate_server_processes() -> list[dict]:
    processes = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmdline = (
                open(f"/proc/{pid}/cmdline", "rb")
                .read()
                .decode("utf-8", "ignore")
                .replace("\x00", " ")
                .strip()
            )
        except Exception:
            continue
        if "petals.cli.run_server" not in cmdline:
            continue
        processes.append({"pid": int(pid), "cmdline": cmdline})
    return processes


def find_server_processes(port: int, block_range: tuple[int, int]) -> list[dict]:
    start, end = block_range
    matches = []
    for process in list_candidate_server_processes():
        cmdline = process["cmdline"]
        if f"/tcp/{port}" not in cmdline:
            continue
        if f"--block_indices {start}:{end}" not in cmdline:
            continue
        matches.append(process)
    return matches


def find_server_processes_from_config(config: dict, port: int, block_range: tuple[int, int]) -> list[dict]:
    matches = []
    for process in config.get("server_processes", []):
        if process.get("port") != port:
            continue
        if tuple(process.get("blocks", [])) != block_range:
            continue
        matches.append({"pid": int(process["pid"]), "cmdline": "<from swarm_config.json>"})
    if matches:
        return matches
    return find_server_processes(port, block_range)


def kill_server(processes: list[dict], method: str) -> list[int]:
    sig = signal.SIGTERM if method == "SIGTERM" else signal.SIGKILL
    killed = []
    for process in processes:
        try:
            os.kill(process["pid"], sig)
            killed.append(process["pid"])
        except ProcessLookupError:
            continue
    return killed


def parse_smoke_output(stdout: str) -> dict:
    parsed = {
        "success": False,
        "elapsed_sec": None,
        "tokens_generated": None,
        "output_text": None,
        "route_line": None,
        "error": None,
    }

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if "Route found:" in line:
            parsed["route_line"] = line
            continue

        match = TIMESTAMP_RE.match(line)
        if not match:
            continue
        message = match.group("message")
        if message.startswith("Generation finished in "):
            try:
                parsed["elapsed_sec"] = float(message.split("Generation finished in ", 1)[1].rstrip("s"))
            except ValueError:
                pass
        elif message.startswith("Generated tokens: "):
            try:
                parsed["tokens_generated"] = int(message.split(": ", 1)[1])
            except ValueError:
                pass
        elif message.startswith("Output: "):
            parsed["output_text"] = message.split(": ", 1)[1]
        elif message == "Smoke test passed.":
            parsed["success"] = True

    return parsed


def run_smoke_subprocess(prompt: str, max_new_tokens: int, timeout_sec: int) -> dict:
    command = [
        sys.executable,
        "test_client.py",
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--traceback-timeout",
        "0",
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "elapsed_sec": timeout_sec,
            "tokens_generated": 0,
            "output_text": None,
            "route_line": None,
            "error": f"subprocess timeout after {timeout_sec}s",
            "returncode": None,
            "stdout": ensure_text(exc.stdout),
            "stderr": ensure_text(exc.stderr),
        }

    parsed = parse_smoke_output(result.stdout)
    if result.returncode != 0 and not parsed["success"]:
        parsed["error"] = f"test_client.py exited with code {result.returncode}"

    parsed["returncode"] = result.returncode
    parsed["stdout"] = ensure_text(result.stdout)
    parsed["stderr"] = ensure_text(result.stderr)
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Run a minimal fault smoke test")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--fault-method", choices=["SIGTERM", "SIGKILL"], default="SIGTERM")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    try:
        with open("swarm_config.json") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        print("ERROR: swarm_config.json not found. Run launch_swarm.py first.")
        sys.exit(1)

    if len(config["server_ports"]) < 2:
        print("ERROR: fault smoke requires at least two peers.")
        sys.exit(1)

    fault_port = config["server_ports"][-1]
    fault_range = tuple(config["block_ranges"][-1])
    target_processes = find_server_processes_from_config(config, fault_port, fault_range)

    output_path = args.output or (
        f"results/fault_smoke_{config['profile']}_{safe_model_tag(config['model_name'])}_{args.fault_method.lower()}.json"
    )

    before = run_smoke_subprocess(BEFORE_PROMPT, args.max_new_tokens, args.timeout)
    killed_pids = kill_server(target_processes, args.fault_method)
    time.sleep(3)
    after = run_smoke_subprocess(AFTER_PROMPT, args.max_new_tokens, args.timeout)

    payload = {
        "experiment": "fault_smoke",
        "config": {
            "preset": config.get("preset"),
            "model_name": config["model_name"],
            "profile": config["profile"],
            "fault_method": args.fault_method,
            "fault_port": fault_port,
            "fault_range": list(fault_range),
            "max_new_tokens": args.max_new_tokens,
            "timeout_sec": args.timeout,
        },
        "results": [
            {"phase": "before_fault", **before},
            {"phase": "after_fault", **after},
        ],
        "fault_events": [
            {
                "type": args.fault_method,
                "port": fault_port,
                "block_range": list(fault_range),
                "matched_processes": target_processes,
                "killed_pids": killed_pids,
                "time": time.time(),
            }
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Before fault: {'OK' if before['success'] else 'FAIL'}")
    print(f"After fault: {'OK' if after['success'] else 'FAIL'}")
    print(f"Matched processes: {len(target_processes)}")
    print(f"Killed PIDs: {killed_pids}")
    print(f"Saved fault smoke results to {output_path}")


if __name__ == "__main__":
    main()

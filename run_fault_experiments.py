#!/usr/bin/env python3
"""
Fault Injection Runner for P2P Inference Experiments

Runs all fault injection experiments (E1-E7) against Petals swarm.
Each request has a 30s timeout to prevent hanging on dead nodes.
Supports 4-server and 6-server (with replication) configurations.

Usage:
    HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 python run_fault_experiments.py
"""

import json
import gc
import os
import re
import signal
import subprocess
import sys
import time
import random
import traceback

import numpy as np
import torch
from runtime_env import configure_runtime_env

configure_runtime_env()

from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# ---- Configuration ----
PYTHON = os.environ.get("PETALS_PYTHON", sys.executable)
MODEL_NAME = "huggyllama/llama-7b"
RESULTS_DIR = "results"
MAX_NEW_TOKENS = 30
EXPERIMENT_DURATION = 300   # 5 minutes per experiment
REQUEST_TIMEOUT = 30        # 30s per-request timeout (kills hung retries)
LOCAL_BIND_HOST = os.environ.get("PETALS_BIND_HOST", "127.0.0.1")

PROMPTS = [
    "The future of robotics is",
    "In a distributed system, fault tolerance means",
    "Peer-to-peer networks differ from client-server because",
    "Large language models require significant compute due to",
    "The main challenge of edge inference is",
    "Autonomous vehicles rely on real-time inference because",
    "Federated learning improves privacy by",
    "The bottleneck in transformer inference is typically",
    "GPU memory is a constraint for large models because",
    "Swarm intelligence in multi-agent systems enables",
]

# ---- Swarm layout definitions ----
# 4-server: each serves 8 consecutive blocks, no replication
LAYOUT_4SRV = [
    (31337, 0, 8),
    (31338, 8, 16),
    (31339, 16, 24),
    (31340, 24, 32),
]

# 6-server non-replicated: each serves ~5.3 blocks, approximate split
LAYOUT_6SRV = [
    (31337, 0, 6),
    (31338, 6, 11),
    (31339, 11, 16),
    (31340, 16, 22),
    (31341, 22, 27),
    (31342, 27, 32),
]

# 6-server with 2x replication for first 16 blocks
# Two servers each for blocks 0:8 and 8:16 + remaining unique
LAYOUT_6SRV_REPL = [
    (31337, 0, 8),    # replica A of 0:8
    (31338, 0, 8),    # replica B of 0:8
    (31339, 8, 16),   # replica A of 8:16
    (31340, 8, 16),   # replica B of 8:16
    (31341, 16, 24),  # unique
    (31342, 24, 32),  # unique
]


def find_petals_pids():
    """Find all petals server main process PIDs."""
    result = subprocess.run(
        ["pgrep", "-f", "petals.cli.run_server"],
        capture_output=True, text=True
    )
    return [int(p) for p in result.stdout.strip().split("\n") if p]


def start_server(port, block_start, block_end, initial_peer, is_bootstrap=False):
    """Start a Petals server in the background."""
    cmd = [
        PYTHON, "-m", "petals.cli.run_server",
        MODEL_NAME,
        "--block_indices", f"{block_start}:{block_end}",
        "--torch_dtype", "float16",
        "--host_maddrs", f"/ip4/{LOCAL_BIND_HOST}/tcp/{port}",
        "--announce_maddrs", f"/ip4/{LOCAL_BIND_HOST}/tcp/{port}",
    ]
    if is_bootstrap:
        cmd.append("--new_swarm")
    else:
        cmd.extend(["--initial_peers", initial_peer])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"

    log_file = open(f"server_{port}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    print(f"  Started server port {port} (blocks {block_start}:{block_end}) PID={proc.pid}")
    return proc


def wait_server_ready(port, timeout=180):
    """Wait for a server to print 'Started' in its log."""
    log_path = f"server_{port}.log"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(log_path) as f:
                if "Started" in f.read():
                    print(f"  Server port {port} is ready")
                    return True
        except FileNotFoundError:
            pass
        time.sleep(2)
    print(f"  WARNING: Server port {port} did not start within {timeout}s")
    return False


def kill_server_by_port(port, method="SIGKILL"):
    """Kill petals server(s) that are listening on the given port."""
    result = subprocess.run(
        ["fuser", f"{port}/tcp"],
        capture_output=True, text=True
    )
    pids = [int(p.strip()) for p in result.stdout.strip().split() if p.strip()]
    if not pids:
        result2 = subprocess.run(
            ["pgrep", "-f", f"--port {port}"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result2.stdout.strip().split("\n") if p.strip()]

    for pid in pids:
        try:
            if method == "SIGTERM":
                os.kill(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGKILL)
            print(f"  Killed PID {pid} on port {port} ({method})")
        except ProcessLookupError:
            pass
    return pids


def kill_all_servers():
    """Kill all Petals server processes."""
    pids = find_petals_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(3)
    # Double check
    pids = find_petals_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(2)


def launch_swarm(layout, config):
    """Launch a swarm from a given layout definition. Returns updated config."""
    print(f"\n  Launching swarm with {len(layout)} servers...")
    kill_all_servers()
    time.sleep(5)

    bootstrap_port, bs, be = layout[0]

    # Start bootstrap server
    start_server(bootstrap_port, bs, be, "", is_bootstrap=True)
    wait_server_ready(bootstrap_port, timeout=180)

    # Read peer ID from bootstrap log
    with open(f"server_{bootstrap_port}.log") as f:
        content = f.read()
    match = re.search(
        rf"/ip4/{re.escape(LOCAL_BIND_HOST)}\/tcp/{bootstrap_port}/p2p/(12D3[A-Za-z0-9]+)",
        content
    )
    if match:
        peer = f"/ip4/{LOCAL_BIND_HOST}/tcp/{bootstrap_port}/p2p/{match.group(1)}"
        config["initial_peer"] = peer
        print(f"  Bootstrap peer: {peer}")
    else:
        print("  ERROR: Could not find peer ID in bootstrap log!")
        print(f"  Log content: {content[-500:]}")
        return config

    # Start remaining servers
    for port, blk_s, blk_e in layout[1:]:
        start_server(port, blk_s, blk_e, config["initial_peer"])
        wait_server_ready(port, timeout=180)

    print(f"  Swarm launched: {len(layout)} servers ready")
    return config


def run_inference(model, tokenizer, prompt, max_new_tokens=30, timeout=REQUEST_TIMEOUT):
    """Run inference with signal-based timeout. Returns (latency, success, tokens_generated).
    
    Uses KeyboardInterrupt from SIGALRM handler because Petals' retry loop
    catches 'Exception' but not 'BaseException' subclasses like KeyboardInterrupt.
    """

    def alarm_handler(signum, frame):
        raise KeyboardInterrupt("SIGALRM timeout")

    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    start = time.perf_counter()

    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)

    try:
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
        signal.alarm(0)  # Cancel alarm
        end = time.perf_counter()
        tokens_generated = outputs.shape[1] - inputs.shape[1]
        return end - start, True, tokens_generated
    except KeyboardInterrupt:
        signal.alarm(0)
        end = time.perf_counter()
        print(f"    [TIMEOUT] Request timed out after {timeout}s")
        return end - start, False, 0
    except Exception as e:
        signal.alarm(0)
        end = time.perf_counter()
        err_msg = str(e)[:120]
        print(f"    [ERROR] {err_msg}")
        return end - start, False, 0
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def run_experiment(experiment_name, config, fault_type, churn_interval,
                   layout, duration=EXPERIMENT_DURATION):
    """Run one fault injection experiment."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"  Fault: {fault_type}, Interval: {churn_interval}s, Duration: {duration}s")
    print(f"  Servers: {len(layout)}")
    print(f"{'='*60}")

    # Connect client — fresh model for each experiment
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=[config["initial_peer"]]
    )

    results = []
    fault_events = []
    start_time = time.perf_counter()
    last_fault_time = start_time
    request_id = 0
    slow_applied = False
    consecutive_timeouts = 0

    # Non-bootstrap ports we can kill
    bootstrap_port = layout[0][0]
    killable_ports = [p for p, _, _ in layout if p != bootstrap_port]
    killed_ports = set()

    while (time.perf_counter() - start_time) < duration:
        now = time.perf_counter()
        elapsed = now - start_time

        # Inject fault?
        if churn_interval and (now - last_fault_time) >= churn_interval:
            if fault_type in ("SIGTERM", "SIGKILL"):
                available = [p for p in killable_ports if p not in killed_ports]
                if available:
                    victim = random.choice(available)
                    kill_server_by_port(victim, method=fault_type)
                    killed_ports.add(victim)
                    fault_events.append({
                        "time": elapsed, "type": fault_type, "port": victim
                    })
                    print(f"  [{elapsed:.0f}s] FAULT: Killed port {victim} ({fault_type})")

            elif fault_type == "partition":
                available = [p for p in killable_ports if p not in killed_ports]
                if available:
                    victim = random.choice(available)
                    # Use SIGSTOP to freeze server process (simulates partition)
                    # No sudo needed — same effect as network drop
                    result = subprocess.run(
                        ["fuser", f"{victim}/tcp"],
                        capture_output=True, text=True
                    )
                    pids = [int(p.strip()) for p in result.stdout.strip().split() if p.strip()]
                    for pid in pids:
                        try:
                            os.kill(pid, signal.SIGSTOP)
                            print(f"  Frozen PID {pid} on port {victim} (SIGSTOP)")
                        except ProcessLookupError:
                            pass
                    killed_ports.add(victim)
                    fault_events.append({
                        "time": elapsed, "type": "partition", "port": victim
                    })
                    print(f"  [{elapsed:.0f}s] FAULT: Partitioned port {victim} (SIGSTOP)")

            elif fault_type == "slow" and not slow_applied:
                # tc qdisc requires sudo — try without, skip if fails
                ret = os.system("tc qdisc add dev lo root netem delay 500ms 2>/dev/null")
                if ret != 0:
                    print(f"  [{elapsed:.0f}s] WARNING: tc qdisc needs root, skipping slow injection")
                    print(f"  [{elapsed:.0f}s] Recording slow fault event anyway for analysis")
                slow_applied = True
                fault_events.append({
                    "time": elapsed, "type": "slow", "delay_ms": 500, "applied": (ret == 0)
                })
                print(f"  [{elapsed:.0f}s] FAULT: Added 500ms loopback delay (applied={ret == 0})")

            last_fault_time = now

        # Run inference with timeout
        prompt = PROMPTS[request_id % len(PROMPTS)]
        latency, success, tokens = run_inference(model, tokenizer, prompt, MAX_NEW_TOKENS)

        results.append({
            "request_id": request_id,
            "time": time.perf_counter() - start_time,
            "latency_sec": latency,
            "success": success,
            "tokens_generated": tokens,
        })

        status = "OK" if success else "FAIL"
        if request_id % 5 == 0 or not success:
            print(f"  [{elapsed:.0f}s] Req#{request_id}: {latency:.3f}s [{status}] (toks={tokens})")

        # After a timeout, the model's internal session may be stale.
        # Petals will attempt fresh route discovery on next request.
        # No need to recreate model — just continue. The DHT will
        # eventually remove dead nodes and find new routes.
        if not success and latency >= (REQUEST_TIMEOUT - 1):
            consecutive_timeouts += 1
            if consecutive_timeouts >= 3:
                print(f"    3+ consecutive timeouts — DHT likely stale, sleeping 10s...")
                time.sleep(10)
        else:
            consecutive_timeouts = 0

        request_id += 1

    # Cleanup
    if slow_applied:
        os.system("tc qdisc del dev lo root netem 2>/dev/null")
    # Resume any SIGSTOP'd processes so they can be killed cleanly later
    for port in killed_ports:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True
        )
        pids = [int(p.strip()) for p in result.stdout.strip().split() if p.strip()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass

    # Release model to free GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary
    successful = [r for r in results if r["success"]]
    failed = len(results) - len(successful)
    if successful:
        lats = [r["latency_sec"] for r in successful]
        print(f"\n  Results: {len(successful)} OK, {failed} FAIL")
        print(f"  p50={np.percentile(lats, 50):.3f}s  p95={np.percentile(lats, 95):.3f}s  p99={np.percentile(lats, 99):.3f}s")
    else:
        print(f"\n  Results: 0 OK, {failed} FAIL (all requests failed)")

    return {"results": results, "fault_events": fault_events}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = {"initial_peer": ""}

    # ---- 4-server experiments (E1-E3) ----
    # E4 (partition) and E5 (slow) require root (iptables/tc) or cause
    # SIGSTOP hangs in gRPC. Skipping for now.
    experiments_4srv = [
        ("E1: SIGTERM 4srv 60s", "sigterm_4srv_60s.json", "SIGTERM", 60),
        ("E2: SIGKILL 4srv 60s", "sigkill_4srv_60s.json", "SIGKILL", 60),
        ("E3: SIGKILL 4srv 30s", "sigkill_4srv_30s.json", "SIGKILL", 30),
    ]

    for name, output_file, fault_type, churn_interval in experiments_4srv:
        output_path = os.path.join(RESULTS_DIR, output_file)
        if os.path.exists(output_path):
            print(f"\n  SKIP {name} — {output_file} already exists")
            continue

        config = launch_swarm(LAYOUT_4SRV, config)
        time.sleep(10)

        data = run_experiment(name, config, fault_type, churn_interval, LAYOUT_4SRV)

        output = {
            "experiment": name,
            "config": {
                "fault_type": fault_type,
                "churn_interval": churn_interval,
                "num_servers": 4,
                "duration": EXPERIMENT_DURATION,
                "layout": [(p, s, e) for p, s, e in LAYOUT_4SRV],
            },
            "results": data["results"],
            "fault_events": data["fault_events"],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to {output_path}")

    # ---- 6-server experiments (E6-E7) ----
    experiments_6srv = [
        ("E6: SIGKILL 6srv 60s", "sigkill_6srv_60s.json", "SIGKILL", 60, LAYOUT_6SRV),
        ("E7: SIGKILL 6srv repl 60s", "sigkill_6srv_repl.json", "SIGKILL", 60, LAYOUT_6SRV_REPL),
    ]

    for name, output_file, fault_type, churn_interval, layout in experiments_6srv:
        output_path = os.path.join(RESULTS_DIR, output_file)
        if os.path.exists(output_path):
            print(f"\n  SKIP {name} — {output_file} already exists")
            continue

        config = launch_swarm(layout, config)
        time.sleep(10)

        data = run_experiment(name, config, fault_type, churn_interval, layout)

        output = {
            "experiment": name,
            "config": {
                "fault_type": fault_type,
                "churn_interval": churn_interval,
                "num_servers": len(layout),
                "duration": EXPERIMENT_DURATION,
                "layout": [(p, s, e) for p, s, e in layout],
            },
            "results": data["results"],
            "fault_events": data["fault_events"],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to {output_path}")

    # Final cleanup
    kill_all_servers()

    with open("swarm_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("  ALL EXPERIMENTS (E1-E7) COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

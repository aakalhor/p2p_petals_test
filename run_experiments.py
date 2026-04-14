#!/usr/bin/env python3
"""
Phase 5: Experiment Matrix Runner

Runs the full set of experiments from the instruction:
  - Baseline (4 servers, no faults)
  - E1: SIGTERM, 4 servers, 1 per 60s
  - E2: SIGKILL, 4 servers, 1 per 60s
  - E3: SIGKILL, 4 servers, 1 per 30s
  - E4: partition, 4 servers, 1 per 60s
  - E5: slow (500ms), 4 servers, applied at t=60s
  - E6: SIGKILL, 6 servers, 1 per 60s
  - E7: SIGKILL, 6 servers (2x replication), 1 per 60s

Usage:
    conda activate petals
    python run_experiments.py [--experiments all|baseline|E1|E2|...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

from swarm_manager import SwarmManager
from run_churn_experiment import run_churn_experiment


MODEL_NAME = "huggyllama/llama-7b"
RESULTS_DIR = "results"
EXPERIMENT_DURATION = 300  # 5 minutes per experiment
MAX_NEW_TOKENS = 30


# Experiment definitions
EXPERIMENTS = {
    "baseline": {
        "label": "Baseline (4 servers, no faults)",
        "num_servers": 4,
        "fault_type": None,
        "churn_interval": None,
        "output_file": "baseline_4srv.json",
        "replicated": False,
    },
    "E1": {
        "label": "E1: SIGTERM, 4 servers, 1 per 60s",
        "num_servers": 4,
        "fault_type": "SIGTERM",
        "churn_interval": 60,
        "output_file": "sigterm_4srv_60s.json",
        "replicated": False,
    },
    "E2": {
        "label": "E2: SIGKILL, 4 servers, 1 per 60s",
        "num_servers": 4,
        "fault_type": "SIGKILL",
        "churn_interval": 60,
        "output_file": "sigkill_4srv_60s.json",
        "replicated": False,
    },
    "E3": {
        "label": "E3: SIGKILL, 4 servers, 1 per 30s",
        "num_servers": 4,
        "fault_type": "SIGKILL",
        "churn_interval": 30,
        "output_file": "sigkill_4srv_30s.json",
        "replicated": False,
    },
    "E4": {
        "label": "E4: partition, 4 servers, 1 per 60s",
        "num_servers": 4,
        "fault_type": "partition",
        "churn_interval": 60,
        "output_file": "partition_4srv_60s.json",
        "replicated": False,
    },
    "E5": {
        "label": "E5: slow (500ms delay), 4 servers",
        "num_servers": 4,
        "fault_type": "slow",
        "churn_interval": 60,  # Applied once at t=60s
        "output_file": "slow_4srv.json",
        "replicated": False,
    },
    "E6": {
        "label": "E6: SIGKILL, 6 servers, 1 per 60s",
        "num_servers": 6,
        "fault_type": "SIGKILL",
        "churn_interval": 60,
        "output_file": "sigkill_6srv_60s.json",
        "replicated": False,
    },
    "E7": {
        "label": "E7: SIGKILL, 6 servers (2x replication), 1 per 60s",
        "num_servers": 6,
        "fault_type": "SIGKILL",
        "churn_interval": 60,
        "output_file": "sigkill_6srv_repl.json",
        "replicated": True,
    },
}


def launch_swarm(manager, num_servers, replicated=False, startup_wait=60):
    """
    Launch a Petals swarm.

    For 4-server configs: each server handles 8 blocks (0:8, 8:16, 16:24, 24:32)
    For 6-server non-replicated: distribute blocks across 6 (approx 5-6 blocks each)
    For 6-server replicated (E7): 4 primary + 2 replicas of first two ranges
    """
    # Server 1: Bootstrap (always blocks 0:8)
    port1 = manager.base_port
    print(f"  Starting bootstrap server on port {port1} (blocks 0:8)...")
    manager.start_server(port1, 0, 8, is_bootstrap=True)
    manager.wait_for_bootstrap(port1, timeout=300)
    time.sleep(startup_wait)

    if num_servers == 4 and not replicated:
        # Standard 4-server layout
        for i in range(1, 4):
            port = manager.base_port + i
            block_start = i * 8
            block_end = block_start + 8
            print(f"  Starting server on port {port} (blocks {block_start}:{block_end})...")
            manager.start_server(port, block_start, block_end)
            time.sleep(startup_wait)

    elif num_servers == 6 and not replicated:
        # 6 servers, ~5-6 blocks each
        block_ranges = [(0, 8), (8, 13), (13, 18), (18, 24), (24, 28), (28, 32)]
        # Bootstrap already covers 0:8, start from index 1
        for i in range(1, 6):
            port = manager.base_port + i
            bs, be = block_ranges[i]
            print(f"  Starting server on port {port} (blocks {bs}:{be})...")
            manager.start_server(port, bs, be)
            time.sleep(startup_wait)

    elif num_servers == 6 and replicated:
        # 4 primary servers (standard) + 2 replicas
        # Primary: 0:8, 8:16, 16:24, 24:32
        # Replicas: duplicate 0:8 and 8:16
        for i in range(1, 4):
            port = manager.base_port + i
            block_start = i * 8
            block_end = block_start + 8
            print(f"  Starting primary server on port {port} (blocks {block_start}:{block_end})...")
            manager.start_server(port, block_start, block_end)
            time.sleep(startup_wait)

        # Replicas
        for i in range(2):
            port = manager.base_port + 4 + i
            block_start = i * 8
            block_end = block_start + 8
            print(f"  Starting REPLICA server on port {port} (blocks {block_start}:{block_end})...")
            manager.start_server(port, block_start, block_end)
            time.sleep(startup_wait)

    manager.status()


def run_single_experiment(exp_name, exp_config):
    """Run a single experiment: launch swarm, run inference, collect data."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp_config['label']}")
    print(f"{'='*60}\n")

    manager = SwarmManager(MODEL_NAME)

    try:
        # Launch the swarm
        print("Launching swarm...")
        launch_swarm(
            manager,
            exp_config["num_servers"],
            replicated=exp_config["replicated"],
            startup_wait=60,
        )

        # Connect client
        print("Connecting inference client...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoDistributedModelForCausalLM.from_pretrained(
            MODEL_NAME,
            initial_peers=[manager.initial_peer],
        )

        # Run experiment
        if exp_config["fault_type"] is None:
            # Baseline: just run inference with no faults
            print("Running baseline inference...")
            from run_baseline import run_baseline, PROMPTS
            results_list = run_baseline(model, tokenizer, num_runs=50, max_new_tokens=MAX_NEW_TOKENS)
            data = {
                "results": [
                    {
                        "request_id": r["run"],
                        "time": sum(rr["elapsed_sec"] for rr in results_list[:r["run"]]),
                        "latency_sec": r["elapsed_sec"],
                        "success": r["success"],
                        "tokens_generated": r["tokens_generated"],
                    }
                    for r in results_list
                ],
                "fault_events": [],
            }
        else:
            # Churn experiment
            data = run_churn_experiment(
                model=model,
                tokenizer=tokenizer,
                swarm_manager=manager,
                fault_type=exp_config["fault_type"],
                churn_interval_sec=exp_config["churn_interval"],
                experiment_duration_sec=EXPERIMENT_DURATION,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # Save results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, exp_config["output_file"])
        output_data = {
            "experiment": exp_name,
            "config": exp_config,
            "results": data["results"],
            "fault_events": data["fault_events"],
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved to {output_path}")

        # Print quick summary
        successful = [r for r in data["results"] if r["success"]]
        if successful:
            latencies = [r["latency_sec"] for r in successful]
            failures = len(data["results"]) - len(successful)
            print(f"  Summary: {len(successful)} successful, {failures} failed")
            print(f"  Latency p50={np.percentile(latencies, 50):.3f}s "
                  f"p95={np.percentile(latencies, 95):.3f}s "
                  f"p99={np.percentile(latencies, 99):.3f}s")

    finally:
        print("\nShutting down swarm for this experiment...")
        manager.shutdown_all()
        # Wait for ports to free up
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment matrix")
    parser.add_argument(
        "--experiments", nargs="+",
        default=["all"],
        help="Which experiments to run. Options: all, baseline, E1, E2, E3, E4, E5, E6, E7",
    )
    args = parser.parse_args()

    if "all" in args.experiments:
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = args.experiments
        for e in experiments_to_run:
            if e not in EXPERIMENTS:
                print(f"ERROR: Unknown experiment '{e}'. Options: {list(EXPERIMENTS.keys())}")
                sys.exit(1)

    print(f"Will run {len(experiments_to_run)} experiment(s): {experiments_to_run}")
    print(f"Estimated total time: ~{len(experiments_to_run) * 10} minutes\n")

    for i, exp_name in enumerate(experiments_to_run):
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(experiments_to_run)}] {exp_name}")
        print(f"{'#'*60}")
        run_single_experiment(exp_name, EXPERIMENTS[exp_name])

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Results in: {RESULTS_DIR}/")
    print(f"  Run: python analyze_results.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

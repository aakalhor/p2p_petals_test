#!/usr/bin/env python3
"""
Phase 3: Baseline Measurements

Run N=50+ inference requests and collect per-token latency,
end-to-end generation time, and throughput (tokens/sec).

Usage:
    python run_baseline.py [--num_runs 50] [--max_tokens 30]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from runtime_env import configure_runtime_env

configure_runtime_env()

from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

from model_presets import safe_model_tag


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


def run_baseline(model, tokenizer, num_runs=50, max_new_tokens=30):
    """Run baseline inference measurements."""
    results = []

    for i in range(num_runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

        start = time.perf_counter()
        try:
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
            end = time.perf_counter()
            elapsed = end - start
            num_tokens = outputs.shape[1] - inputs.shape[1]
            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
            success = True
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            end = time.perf_counter()
            elapsed = end - start
            num_tokens = 0
            tokens_per_sec = 0
            success = False
            generated_text = f"ERROR: {str(e)}"

        result = {
            "run": i,
            "prompt": prompt,
            "elapsed_sec": elapsed,
            "tokens_generated": num_tokens,
            "tokens_per_sec": tokens_per_sec,
            "success": success,
        }
        results.append(result)
        status = "OK" if success else "FAIL"
        print(f"  Run {i:3d}/{num_runs}: {elapsed:.3f}s, {tokens_per_sec:.2f} tok/s [{status}]")

    return results


def print_summary(results):
    """Print summary statistics."""
    successful = [r for r in results if r["success"]]
    if not successful:
        print("\nNo successful runs!")
        return

    latencies = [r["elapsed_sec"] for r in successful]
    throughputs = [r["tokens_per_sec"] for r in successful]
    failures = len(results) - len(successful)

    print(f"\n{'='*50}")
    print(f"  BASELINE SUMMARY ({len(results)} total runs)")
    print(f"{'='*50}")
    print(f"  Successful runs: {len(successful)}/{len(results)} ({failures} failures)")
    print(f"  Latency (seconds):")
    print(f"    p50:  {np.percentile(latencies, 50):.3f}s")
    print(f"    p95:  {np.percentile(latencies, 95):.3f}s")
    print(f"    p99:  {np.percentile(latencies, 99):.3f}s")
    print(f"    mean: {np.mean(latencies):.3f}s")
    print(f"    std:  {np.std(latencies):.3f}s")
    print(f"  Throughput:")
    print(f"    mean: {np.mean(throughputs):.2f} tok/s")
    print(f"    min:  {np.min(throughputs):.2f} tok/s")
    print(f"    max:  {np.max(throughputs):.2f} tok/s")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Baseline inference measurements")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of inference runs")
    parser.add_argument("--max_tokens", type=int, default=30, help="Max new tokens per run")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    # Load swarm config
    try:
        with open("swarm_config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("ERROR: swarm_config.json not found. Run launch_swarm.py first.")
        sys.exit(1)

    model_name = config["model_name"]
    initial_peers = config.get("initial_peers") or [config["initial_peer"]]
    model_tag = safe_model_tag(model_name)
    output_path = args.output or f"results/baseline_{config['profile']}_{model_tag}.json"

    print(f"Model: {model_name}")
    print(f"Initial peer: {initial_peers[0]}")
    print(f"Runs: {args.num_runs}, Max tokens: {args.max_tokens}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Connecting to swarm...")
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_name,
        initial_peers=initial_peers,
    )

    print(f"\nRunning {args.num_runs} baseline inference requests...\n")
    results = run_baseline(model, tokenizer, args.num_runs, args.max_tokens)

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output_data = {
        "experiment": "baseline",
        "config": {
            "preset": config.get("preset"),
            "model_name": model_name,
            "num_runs": args.num_runs,
            "max_new_tokens": args.max_tokens,
            "profile": config["profile"],
            "num_servers": config["num_servers"],
        },
        "results": results,
        "fault_events": [],  # No faults in baseline
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print_summary(results)


if __name__ == "__main__":
    main()

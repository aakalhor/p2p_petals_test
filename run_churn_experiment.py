#!/usr/bin/env python3
"""
Phase 4: Churn Experiment Runner

Continuously runs inference while injecting faults at regular intervals.
Returns per-request latency measurements with fault event timestamps.

This module is imported by run_experiments.py.
"""

import time
import random


def run_churn_experiment(
    model,
    tokenizer,
    swarm_manager,
    fault_type="SIGKILL",       # SIGTERM, SIGKILL, partition, slow
    churn_interval_sec=60,      # inject a fault every N seconds
    experiment_duration_sec=300, # total experiment time
    max_new_tokens=30,
    prompts=None,
):
    """
    Continuously run inference while injecting faults at regular intervals.

    Args:
        model: The distributed model
        tokenizer: The tokenizer
        swarm_manager: SwarmManager instance
        fault_type: One of "SIGTERM", "SIGKILL", "partition", "slow"
        churn_interval_sec: Inject a fault every N seconds
        experiment_duration_sec: Total experiment duration
        max_new_tokens: Max tokens to generate per request
        prompts: List of prompts to cycle through

    Returns:
        dict with "results" and "fault_events" lists
    """
    if prompts is None:
        prompts = [
            "The future of robotics is",
            "In a distributed system, fault tolerance means",
            "Peer-to-peer networks differ from client-server because",
            "Large language models require significant compute due to",
            "The main challenge of edge inference is",
        ]

    results = []
    fault_events = []
    start_time = time.perf_counter()
    last_fault_time = start_time
    request_id = 0
    slow_applied = False

    print(f"  [Churn] Starting experiment: fault_type={fault_type}, "
          f"churn_interval={churn_interval_sec}s, duration={experiment_duration_sec}s")

    while (time.perf_counter() - start_time) < experiment_duration_sec:
        now = time.perf_counter()
        elapsed_experiment = now - start_time

        # --- Inject fault if it's time ---
        if (now - last_fault_time) >= churn_interval_sec:
            if fault_type in ("SIGTERM", "SIGKILL"):
                killable = swarm_manager.get_alive_ports()
                # Don't kill the bootstrap node (it holds the DHT)
                bootstrap_port = swarm_manager.base_port
                killable = [p for p in killable if p != bootstrap_port]
                if killable:
                    victim_port = random.choice(killable)
                    swarm_manager.kill_server(victim_port, method=fault_type)
                    fault_events.append({
                        "time": elapsed_experiment,
                        "type": fault_type,
                        "port": victim_port,
                    })
                    print(f"  [Churn] t={elapsed_experiment:.1f}s: Killed port {victim_port} ({fault_type})")
                else:
                    print(f"  [Churn] t={elapsed_experiment:.1f}s: No killable servers left (bootstrap only)")

            elif fault_type == "partition":
                alive = swarm_manager.get_alive_ports()
                bootstrap_port = swarm_manager.base_port
                candidates = [p for p in alive if p != bootstrap_port]
                if candidates:
                    victim_port = random.choice(candidates)
                    swarm_manager.partition_server(victim_port)
                    fault_events.append({
                        "time": elapsed_experiment,
                        "type": "partition",
                        "port": victim_port,
                    })
                    print(f"  [Churn] t={elapsed_experiment:.1f}s: Partitioned port {victim_port}")

            elif fault_type == "slow":
                if not slow_applied:
                    swarm_manager.add_latency(500)
                    slow_applied = True
                    fault_events.append({
                        "time": elapsed_experiment,
                        "type": "slow",
                        "delay_ms": 500,
                    })
                    print(f"  [Churn] t={elapsed_experiment:.1f}s: Added 500ms loopback delay")

            last_fault_time = now

        # --- Run one inference request ---
        prompt = prompts[request_id % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

        req_start = time.perf_counter()
        try:
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
            req_end = time.perf_counter()
            success = True
            tokens_generated = outputs.shape[1] - inputs.shape[1]
        except Exception as e:
            req_end = time.perf_counter()
            success = False
            tokens_generated = 0

        latency = req_end - req_start
        results.append({
            "request_id": request_id,
            "time": req_start - start_time,
            "latency_sec": latency,
            "success": success,
            "tokens_generated": tokens_generated,
        })

        status = "OK" if success else "FAIL"
        if request_id % 10 == 0:
            print(f"  [Churn] Request {request_id}: {latency:.3f}s [{status}] "
                  f"(t={req_start - start_time:.1f}s)")
        request_id += 1

    # Cleanup: remove latency if applied
    if slow_applied:
        swarm_manager.remove_latency()

    successful = sum(1 for r in results if r["success"])
    print(f"  [Churn] Done: {len(results)} requests, {successful} successful, "
          f"{len(fault_events)} fault events")

    return {"results": results, "fault_events": fault_events}

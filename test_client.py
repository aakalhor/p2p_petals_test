#!/usr/bin/env python3
"""
Quick smoke test: connect to the swarm and generate a few tokens.

Usage:
    python test_client.py
    python test_client.py --max-new-tokens 16 --debug-log smoke_trace.log
"""

import argparse
import faulthandler
import json
import sys
import time
from pathlib import Path

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch
from petals import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer


DEFAULT_PROMPT = "The future of robotics is"


def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the private Petals swarm")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to generate from")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Number of new tokens to generate")
    parser.add_argument(
        "--debug-log",
        default="test_client.trace.log",
        help="Path for faulthandler trace output if the client stalls",
    )
    parser.add_argument(
        "--traceback-timeout",
        type=int,
        default=300,
        help="Seconds before dumping Python stack traces for a stuck client",
    )
    return parser.parse_args()


def configure_debug_logging(path, timeout):
    debug_path = Path(path)
    debug_handle = debug_path.open("w")
    faulthandler.enable(file=debug_handle, all_threads=True)
    if timeout > 0:
        faulthandler.dump_traceback_later(timeout, repeat=True, file=debug_handle)
    return debug_handle


def main():
    args = parse_args()
    debug_handle = configure_debug_logging(args.debug_log, args.traceback_timeout)

    try:
        try:
            with open("swarm_config.json") as handle:
                config = json.load(handle)
        except FileNotFoundError:
            log("ERROR: swarm_config.json not found. Run launch_swarm.py first.")
            sys.exit(1)

        model_name = config["model_name"]
        initial_peers = config.get("initial_peers") or [config["initial_peer"]]

        log(f"Model: {model_name}")
        log(f"Initial peer: {initial_peers[0]}")
        log(f"Debug traces: {args.debug_log}")
        log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        log("Connecting to swarm and loading distributed model...")
        model = AutoDistributedModelForCausalLM.from_pretrained(
            model_name,
            initial_peers=initial_peers,
        )
        model.eval()
        log("Distributed model is ready.")

        prompt = args.prompt
        log(f"Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
        log(f"Prompt tokens: {inputs.shape[1]}")

        log(f"Generating {args.max_new_tokens} tokens...")
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - start

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_tokens = outputs.shape[1] - inputs.shape[1]
        log(f"Generation finished in {elapsed:.2f}s")
        log(f"Generated tokens: {generated_tokens}")
        log(f"Output: {text}")
        log("Smoke test passed.")
    finally:
        faulthandler.cancel_dump_traceback_later()
        debug_handle.close()


if __name__ == "__main__":
    main()

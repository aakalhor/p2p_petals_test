#!/usr/bin/env python3
"""
Quick smoke test: connect to the swarm and generate a few tokens.

Usage:
    conda activate petals
    python test_client.py
"""

import json
import sys
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM


def main():
    # Load swarm config
    try:
        with open("swarm_config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("ERROR: swarm_config.json not found. Run launch_swarm.py first.")
        sys.exit(1)

    model_name = config["model_name"]
    initial_peer = config["initial_peer"]

    print(f"Model: {model_name}")
    print(f"Initial peer: {initial_peer}")
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Connecting to swarm and loading distributed model...")
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_name,
        initial_peers=[initial_peer],
    )

    prompt = "The future of robotics is"
    print(f"\nPrompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

    print("Generating...")
    outputs = model.generate(inputs, max_new_tokens=30)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {text}")
    print("\n✓ Smoke test passed!")


if __name__ == "__main__":
    main()

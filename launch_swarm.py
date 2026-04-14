#!/usr/bin/env python3
"""
Launch a private Petals swarm with 4 servers on localhost.

Each server handles 8 of the 32 transformer blocks of Llama-2-7b-hf.
After all servers are running, saves the swarm config for other scripts.

Usage:
    conda activate petals
    python launch_swarm.py
"""

import json
import time
import sys
from swarm_manager import SwarmManager

MODEL_NAME = "huggyllama/llama-7b"
BASE_PORT = 31337
BLOCKS_PER_SERVER = 8
TOTAL_BLOCKS = 32
NUM_SERVERS = TOTAL_BLOCKS // BLOCKS_PER_SERVER  # 4
STARTUP_WAIT_PER_SERVER = 60  # seconds to wait for each non-bootstrap server


def main():
    print("=" * 60)
    print("  P2P Inference — Launching Private Swarm")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Servers: {NUM_SERVERS} (blocks per server: {BLOCKS_PER_SERVER})")
    print("=" * 60)

    manager = SwarmManager(MODEL_NAME, base_port=BASE_PORT)

    # --- Server 1: Bootstrap (blocks 0-7) ---
    port1 = BASE_PORT
    print(f"\n[1/{NUM_SERVERS}] Starting bootstrap server on port {port1} (blocks 0:{BLOCKS_PER_SERVER})...")
    manager.start_server(port1, 0, BLOCKS_PER_SERVER, is_bootstrap=True)

    # Wait for the bootstrap node to announce its peer ID
    try:
        peer_addr = manager.wait_for_bootstrap(port1, timeout=300)
    except TimeoutError as e:
        print(f"\nERROR: {e}")
        print("Check server logs: server_{port1}.log")
        manager.shutdown_all()
        sys.exit(1)

    # --- Servers 2-4 ---
    for i in range(1, NUM_SERVERS):
        port = BASE_PORT + i
        block_start = i * BLOCKS_PER_SERVER
        block_end = block_start + BLOCKS_PER_SERVER
        print(f"\n[{i+1}/{NUM_SERVERS}] Starting server on port {port} (blocks {block_start}:{block_end})...")
        manager.start_server(port, block_start, block_end, is_bootstrap=False)
        print(f"  Waiting {STARTUP_WAIT_PER_SERVER}s for server to load model blocks...")
        time.sleep(STARTUP_WAIT_PER_SERVER)

    # Check all servers alive
    manager.status()

    alive = manager.get_alive_ports()
    if len(alive) < NUM_SERVERS:
        dead = manager.get_dead_ports()
        print(f"WARNING: {len(dead)} server(s) died during startup. Ports: {dead}")
        print("Check logs: server_<port>.log")
    else:
        print("All servers are running!")

    # Save swarm config for other scripts
    config = {
        "model_name": MODEL_NAME,
        "initial_peer": peer_addr,
        "base_port": BASE_PORT,
        "num_servers": NUM_SERVERS,
        "blocks_per_server": BLOCKS_PER_SERVER,
        "server_ports": [BASE_PORT + i for i in range(NUM_SERVERS)],
    }
    with open("swarm_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSwarm config saved to swarm_config.json")
    print("\nSwarm is ready! You can now run:")
    print("  python test_client.py          # Quick smoke test")
    print("  python run_baseline.py         # Baseline measurements")
    print("  python run_experiments.py      # Full experiment matrix")

    # Keep running — servers are background processes
    print("\nPress Ctrl+C to shut down the swarm.")
    try:
        while True:
            time.sleep(10)
            dead = manager.get_dead_ports()
            if dead:
                print(f"WARNING: Server(s) on ports {dead} have died.")
    except KeyboardInterrupt:
        print("\nShutting down swarm...")
        manager.shutdown_all()


if __name__ == "__main__":
    main()

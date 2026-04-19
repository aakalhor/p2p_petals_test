#!/usr/bin/env python3
"""
Launch a private local Petals swarm for llama-7b on one machine.

Usage:
    python launch_swarm.py
    python launch_swarm.py --profile compact
"""

import argparse
import json
import sys
import time

from swarm_manager import SwarmManager


MODEL_NAME = "huggyllama/llama-7b"
BASE_PORT = 31337
SERVER_START_TIMEOUT = 180
DEFAULT_SERVER_ARGS = [
    "--num_handlers",
    "1",
    "--prefetch_batches",
    "1",
    "--sender_threads",
    "1",
    "--attn_cache_tokens",
    "4096",
    "--balance_quality",
    "0.0",
]

SWARM_PROFILES = {
    "presentation": {
        "description": "4 local peers, 8 blocks each",
        "block_ranges": [(0, 8), (8, 16), (16, 24), (24, 32)],
    },
    "compact": {
        "description": "2 local peers, lower process overhead",
        "block_ranges": [(0, 16), (16, 32)],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a private local Petals swarm")
    parser.add_argument("--profile", choices=sorted(SWARM_PROFILES), default="presentation")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--base-port", type=int, default=BASE_PORT)
    parser.add_argument("--bind-host", default="127.0.0.1", help="Host used for listen multiaddrs")
    parser.add_argument("--announce-host", help="Host advertised to clients and peers; defaults to --bind-host")
    parser.add_argument("--startup-timeout", type=int, default=SERVER_START_TIMEOUT)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra argument to pass through to petals.cli.run_server. Repeat for multiple values.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    profile = SWARM_PROFILES[args.profile]
    block_ranges = profile["block_ranges"]
    num_servers = len(block_ranges)
    announce_host = args.announce_host or args.bind_host
    exit_code = 0

    print("=" * 60)
    print("  P2P Inference - Launching Private Swarm")
    print(f"  Model: {args.model}")
    print(f"  Profile: {args.profile} ({profile['description']})")
    print(f"  Host: {args.bind_host}")
    print(f"  Announce host: {announce_host}")
    print(f"  Servers: {num_servers}")
    print(f"  Server args: {' '.join(DEFAULT_SERVER_ARGS + args.server_arg)}")
    print("=" * 60)

    manager = SwarmManager(
        args.model,
        base_port=args.base_port,
        bind_host=args.bind_host,
        announce_host=announce_host,
        extra_server_args=DEFAULT_SERVER_ARGS + args.server_arg,
    )

    try:
        bootstrap_port = args.base_port
        bootstrap_start, bootstrap_end = block_ranges[0]
        print(
            f"\n[1/{num_servers}] Starting bootstrap server on port {bootstrap_port} "
            f"(blocks {bootstrap_start}:{bootstrap_end})..."
        )
        manager.start_server(bootstrap_port, bootstrap_start, bootstrap_end, is_bootstrap=True)
        peer_addr = manager.wait_for_bootstrap(bootstrap_port, timeout=args.startup_timeout)
        manager.wait_for_server_start(bootstrap_port, timeout=args.startup_timeout)

        for index, (block_start, block_end) in enumerate(block_ranges[1:], start=1):
            port = args.base_port + index
            print(
                f"\n[{index + 1}/{num_servers}] Starting server on port {port} "
                f"(blocks {block_start}:{block_end})..."
            )
            manager.start_server(port, block_start, block_end, is_bootstrap=False)
            manager.wait_for_server_start(port, timeout=args.startup_timeout)

        manager.status()
        alive = manager.get_alive_ports()
        if len(alive) < num_servers:
            dead = manager.get_dead_ports()
            print(f"WARNING: {len(dead)} server(s) died during startup. Ports: {dead}")
            print("Check logs: server_<port>.log")
        else:
            print("All servers are running.")

        config = {
            "model_name": args.model,
            "initial_peer": peer_addr,
            "initial_peers": [peer_addr],
            "base_port": args.base_port,
            "bind_host": args.bind_host,
            "announce_host": announce_host,
            "profile": args.profile,
            "num_servers": num_servers,
            "block_ranges": block_ranges,
            "server_ports": [args.base_port + i for i in range(num_servers)],
            "server_args": DEFAULT_SERVER_ARGS + args.server_arg,
        }
        with open("swarm_config.json", "w") as handle:
            json.dump(config, handle, indent=2)

        print("\nSwarm config saved to swarm_config.json")
        print("\nSwarm is ready. Smoke test:")
        print("  python test_client.py")
        print("\nPress Ctrl+C to shut down the swarm.")

        while True:
            time.sleep(10)
            dead = manager.get_dead_ports()
            if dead:
                print(f"WARNING: Server(s) on ports {dead} have died.")
    except KeyboardInterrupt:
        print("\nShutting down swarm...")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("Check server_<port>.log for details.")
        exit_code = 1
    finally:
        manager.shutdown_all()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

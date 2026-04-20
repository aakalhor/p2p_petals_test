#!/usr/bin/env python3
"""
Launch a private local Petals swarm on one machine.

Usage:
    python launch_swarm.py
    python launch_swarm.py --profile single
    python launch_swarm.py --preset llama-7b-legacy --profile compact
"""

import argparse
import json
import time

from model_presets import DEFAULT_MODEL_PRESET, MODEL_PRESETS, build_profile, resolve_model_config
from swarm_manager import SwarmManager


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


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a private local Petals swarm")
    parser.add_argument("--profile", choices=["single", "compact", "presentation"], default="single")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default=DEFAULT_MODEL_PRESET)
    parser.add_argument("--model", help="Override the model name from the preset")
    parser.add_argument("--total-blocks", type=int, help="Override the total block count from the preset")
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
    model_config = resolve_model_config(args.preset, model_name=args.model, total_blocks=args.total_blocks)
    profile = build_profile(args.profile, model_config["total_blocks"])
    block_ranges = profile["block_ranges"]
    announce_host = args.announce_host or args.bind_host
    exit_code = 0

    print("=" * 60)
    print("  P2P Inference - Launching Private Swarm")
    print(f"  Preset: {model_config['preset']} ({model_config['summary']})")
    print(f"  Model: {model_config['model_name']}")
    print(f"  Total blocks: {model_config['total_blocks']}")
    print(f"  Profile: {profile['name']} ({profile['description']})")
    print(f"  Host: {args.bind_host}")
    print(f"  Announce host: {announce_host}")
    print(f"  Servers: {profile['num_servers']}")
    print(f"  Server args: {' '.join(DEFAULT_SERVER_ARGS + args.server_arg)}")
    print("=" * 60)

    manager = SwarmManager(
        model_config["model_name"],
        base_port=args.base_port,
        bind_host=args.bind_host,
        announce_host=announce_host,
        extra_server_args=DEFAULT_SERVER_ARGS + args.server_arg,
    )

    try:
        bootstrap_port = args.base_port
        bootstrap_start, bootstrap_end = block_ranges[0]
        print(
            f"\n[1/{profile['num_servers']}] Starting bootstrap server on port {bootstrap_port} "
            f"(blocks {bootstrap_start}:{bootstrap_end})..."
        )
        manager.start_server(bootstrap_port, bootstrap_start, bootstrap_end, is_bootstrap=True)
        peer_addr = manager.wait_for_bootstrap(bootstrap_port, timeout=args.startup_timeout)
        manager.wait_for_server_start(bootstrap_port, timeout=args.startup_timeout)

        for index, (block_start, block_end) in enumerate(block_ranges[1:], start=1):
            port = args.base_port + index
            print(
                f"\n[{index + 1}/{profile['num_servers']}] Starting server on port {port} "
                f"(blocks {block_start}:{block_end})..."
            )
            manager.start_server(port, block_start, block_end, is_bootstrap=False)
            manager.wait_for_server_start(port, timeout=args.startup_timeout)

        manager.status()
        alive = manager.get_alive_ports()
        if len(alive) < profile["num_servers"]:
            dead = manager.get_dead_ports()
            print(f"WARNING: {len(dead)} server(s) died during startup. Ports: {dead}")
            print("Check logs: server_<port>.log")
        else:
            print("All servers are running.")

        config = {
            "preset": model_config["preset"],
            "model_name": model_config["model_name"],
            "model_summary": model_config["summary"],
            "total_blocks": model_config["total_blocks"],
            "initial_peer": peer_addr,
            "initial_peers": [peer_addr],
            "base_port": args.base_port,
            "bind_host": args.bind_host,
            "announce_host": announce_host,
            "profile": profile["name"],
            "profile_description": profile["description"],
            "num_servers": profile["num_servers"],
            "block_ranges": block_ranges,
            "server_ports": [args.base_port + i for i in range(profile["num_servers"])],
            "server_args": DEFAULT_SERVER_ARGS + args.server_arg,
            "server_processes": [
                {
                    "port": port,
                    "pid": manager.servers[port]["process"].pid,
                    "blocks": list(manager.servers[port]["blocks"]),
                    "is_bootstrap": manager.servers[port]["is_bootstrap"],
                }
                for port in sorted(manager.servers)
            ],
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

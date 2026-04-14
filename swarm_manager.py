"""
SwarmManager: Manages Petals server processes for fault injection experiments.

Handles launching, killing, partitioning, and adding latency to Petals servers.
"""

import subprocess
import signal
import time
import os
import re


class SwarmManager:
    """Manages Petals server processes for fault injection experiments."""

    def __init__(self, model_name, base_port=31337, torch_dtype="float16"):
        self.model_name = model_name
        self.base_port = base_port
        self.torch_dtype = torch_dtype
        self.servers = {}  # port -> {"process": Popen, "blocks": (start, end), "is_bootstrap": bool}
        self.initial_peer = None  # Set after bootstrap server starts

    def start_server(self, port, block_start, block_end, is_bootstrap=False):
        """Start a Petals server process serving the given block range."""
        cmd = [
            "python", "-m", "petals.cli.run_server",
            self.model_name,
            "--num_blocks", str(block_end - block_start),
            "--block_indices", f"{block_start}:{block_end}",
            "--port", str(port),
            "--torch_dtype", self.torch_dtype,
        ]
        if is_bootstrap:
            cmd.append("--new_swarm")
        else:
            if self.initial_peer is None:
                raise RuntimeError("Bootstrap server not started yet — no initial_peer available.")
            cmd.extend(["--initial_peers", self.initial_peer])

        log_file = open(f"server_{port}.log", "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # Create new process group for clean kills
        )
        self.servers[port] = {
            "process": proc,
            "blocks": (block_start, block_end),
            "is_bootstrap": is_bootstrap,
            "log_file": log_file,
        }
        print(f"[SwarmManager] Started server on port {port} serving blocks {block_start}:{block_end} (PID: {proc.pid})")
        return proc

    def wait_for_bootstrap(self, port, timeout=300):
        """
        Wait for the bootstrap server to print its multiaddress, then extract and store it.
        Reads from the server log file.
        """
        log_path = f"server_{port}.log"
        start = time.time()
        print(f"[SwarmManager] Waiting for bootstrap server on port {port} to announce peer ID...")

        while time.time() - start < timeout:
            try:
                with open(log_path, "r") as f:
                    content = f.read()
                # Look for multiaddress pattern: /ip4/.../tcp/.../p2p/...
                match = re.search(r"(/ip4/[\d.]+/tcp/\d+/p2p/\S+)", content)
                if match:
                    self.initial_peer = match.group(1)
                    print(f"[SwarmManager] Bootstrap peer address: {self.initial_peer}")
                    return self.initial_peer
                # Also check for "running" or ready indicators
                if "ready" in content.lower() or "running" in content.lower():
                    # Try to find peer ID another way
                    match2 = re.search(r"(Qm\S+|12D3\S+)", content)
                    if match2:
                        peer_id = match2.group(1)
                        self.initial_peer = f"/ip4/127.0.0.1/tcp/{port}/p2p/{peer_id}"
                        print(f"[SwarmManager] Bootstrap peer address (constructed): {self.initial_peer}")
                        return self.initial_peer
            except FileNotFoundError:
                pass
            time.sleep(2)

        raise TimeoutError(f"Bootstrap server on port {port} did not announce peer ID within {timeout}s. "
                           f"Check server_{port}.log for errors.")

    def kill_server(self, port, method="SIGKILL"):
        """Kill a server process. method: SIGTERM, SIGKILL"""
        if port not in self.servers:
            print(f"[SwarmManager] No server on port {port}")
            return
        proc = self.servers[port]["process"]
        if proc.poll() is not None:
            print(f"[SwarmManager] Server on port {port} already dead")
            return
        if method == "SIGTERM":
            proc.send_signal(signal.SIGTERM)
        elif method == "SIGKILL":
            proc.kill()
        else:
            raise ValueError(f"Unknown kill method: {method}")
        print(f"[SwarmManager] Killed server on port {port} with {method} (PID: {proc.pid})")

    def partition_server(self, port):
        """Simulate network partition using iptables (requires sudo)."""
        os.system(f"sudo iptables -A INPUT -p tcp --dport {port} -j DROP")
        os.system(f"sudo iptables -A OUTPUT -p tcp --sport {port} -j DROP")
        print(f"[SwarmManager] Partitioned server on port {port}")

    def heal_partition(self, port):
        """Remove iptables rules for a given port."""
        os.system(f"sudo iptables -D INPUT -p tcp --dport {port} -j DROP")
        os.system(f"sudo iptables -D OUTPUT -p tcp --sport {port} -j DROP")
        print(f"[SwarmManager] Healed partition on port {port}")

    def add_latency(self, delay_ms=500):
        """Add latency to loopback interface (affects all local servers). Requires sudo."""
        os.system(f"sudo tc qdisc add dev lo root netem delay {delay_ms}ms")
        print(f"[SwarmManager] Added {delay_ms}ms delay to loopback")

    def remove_latency(self):
        """Remove netem delay from loopback."""
        os.system("sudo tc qdisc del dev lo root netem 2>/dev/null")
        print("[SwarmManager] Removed loopback delay")

    def restart_server(self, port):
        """Restart a killed server with the same block range."""
        if port not in self.servers:
            raise ValueError(f"No record of server on port {port}")
        info = self.servers[port]
        block_start, block_end = info["blocks"]
        # Close old log file
        if "log_file" in info and info["log_file"]:
            info["log_file"].close()
        self.start_server(port, block_start, block_end, is_bootstrap=False)

    def get_alive_ports(self):
        """Return list of ports whose server processes are still running."""
        return [p for p, info in self.servers.items() if info["process"].poll() is None]

    def get_dead_ports(self):
        """Return list of ports whose server processes have exited."""
        return [p for p, info in self.servers.items() if info["process"].poll() is not None]

    def shutdown_all(self):
        """Gracefully shut down all servers."""
        print("[SwarmManager] Shutting down all servers...")
        for port, info in self.servers.items():
            proc = info["process"]
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                print(f"  Sent SIGTERM to server on port {port} (PID: {proc.pid})")
        # Wait for graceful shutdown
        time.sleep(5)
        for port, info in self.servers.items():
            proc = info["process"]
            if proc.poll() is None:
                proc.kill()
                print(f"  Force-killed server on port {port} (PID: {proc.pid})")
            if "log_file" in info and info["log_file"]:
                info["log_file"].close()
        # Clean up any iptables rules and netem
        self.remove_latency()
        print("[SwarmManager] All servers shut down.")

    def status(self):
        """Print status of all servers."""
        print("\n=== Swarm Status ===")
        for port, info in sorted(self.servers.items()):
            proc = info["process"]
            alive = proc.poll() is None
            blocks = info["blocks"]
            status_str = "ALIVE" if alive else f"DEAD (exit code: {proc.returncode})"
            bootstrap_str = " [BOOTSTRAP]" if info.get("is_bootstrap") else ""
            print(f"  Port {port}: blocks {blocks[0]}:{blocks[1]} — {status_str}{bootstrap_str}")
        print(f"  Initial peer: {self.initial_peer}")
        print("====================\n")

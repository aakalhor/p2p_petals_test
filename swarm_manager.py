"""
SwarmManager: Manages local Petals server processes for smoke tests and experiments.
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


class SwarmManager:
    """Manages Petals server processes for local swarm experiments."""

    def __init__(
        self,
        model_name,
        base_port=31337,
        torch_dtype="float16",
        python_executable=None,
        bind_host="127.0.0.1",
        announce_host=None,
        log_dir=".",
        extra_server_args=None,
    ):
        self.model_name = model_name
        self.base_port = base_port
        self.torch_dtype = torch_dtype
        self.python_executable = python_executable or sys.executable
        self.bind_host = bind_host
        self.announce_host = announce_host or bind_host
        self.log_dir = Path(log_dir)
        self.extra_server_args = list(extra_server_args or [])
        self.servers = {}
        self.initial_peer = None

    def _log_path(self, port):
        return self.log_dir / f"server_{port}.log"

    def _server_multiaddr(self, port, host=None):
        host = host or self.announce_host
        return f"/ip4/{host}/tcp/{port}"

    def _build_command(self, port, block_start, block_end, is_bootstrap):
        cmd = [
            self.python_executable,
            "-m",
            "petals.cli.run_server",
            self.model_name,
            "--block_indices",
            f"{block_start}:{block_end}",
            "--torch_dtype",
            self.torch_dtype,
        ]

        if self.bind_host:
            cmd.extend(["--host_maddrs", self._server_multiaddr(port, self.bind_host)])
            cmd.extend(["--announce_maddrs", self._server_multiaddr(port, self.announce_host)])
        else:
            cmd.extend(["--port", str(port)])

        if is_bootstrap:
            cmd.append("--new_swarm")
        else:
            if self.initial_peer is None:
                raise RuntimeError("Bootstrap server not started yet - no initial_peer available.")
            cmd.extend(["--initial_peers", self.initial_peer])

        cmd.extend(self.extra_server_args)
        return cmd

    def _spawn(self, cmd, log_file):
        kwargs = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "env": {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
                "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
                "TMP": os.environ.get("TMP", os.environ.get("TMPDIR", "/tmp")),
                "TEMP": os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp")),
                "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
                "HUGGINGFACE_HUB_CACHE": os.environ.get(
                    "HUGGINGFACE_HUB_CACHE",
                    str(Path.home() / ".cache" / "huggingface" / "hub"),
                ),
            },
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(cmd, **kwargs)

    def _signal_process(self, proc, sig):
        if proc.poll() is not None:
            return

        if os.name == "nt":
            if sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.send_signal(sig)
            return

        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            pass

    def start_server(self, port, block_start, block_end, is_bootstrap=False):
        """Start a Petals server process serving the given block range."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_path(port)
        log_file = open(log_path, "w")
        cmd = self._build_command(port, block_start, block_end, is_bootstrap)
        proc = self._spawn(cmd, log_file)
        self.servers[port] = {
            "process": proc,
            "blocks": (block_start, block_end),
            "is_bootstrap": is_bootstrap,
            "log_file": log_file,
            "log_path": log_path,
        }
        print(f"[SwarmManager] Started server on port {port} serving blocks {block_start}:{block_end} (PID: {proc.pid})")
        return proc

    def _read_log(self, port):
        info = self.servers.get(port)
        log_path = info["log_path"] if info else self._log_path(port)
        try:
            return log_path.read_text(errors="ignore")
        except FileNotFoundError:
            return ""

    def _extract_multiaddrs(self, content, port=None):
        matches = re.findall(r"(/ip4/[0-9.]+/tcp/\d+/p2p/[A-Za-z0-9]+)", content)
        if port is None:
            return matches
        expected = f"/tcp/{port}/p2p/"
        return [match for match in matches if expected in match]

    def wait_for_bootstrap(self, port, timeout=300):
        """Wait for the bootstrap server to announce a usable peer multiaddr."""
        start = time.time()
        print(f"[SwarmManager] Waiting for bootstrap server on port {port} to announce peer ID...")

        while time.time() - start < timeout:
            info = self.servers.get(port)
            if info and info["process"].poll() is not None:
                raise RuntimeError(
                    f"Bootstrap server on port {port} exited early with code {info['process'].returncode}. "
                    f"Check {info['log_path']}."
                )

            content = self._read_log(port)
            maddrs = self._extract_multiaddrs(content, port=port)
            if maddrs:
                preferred_prefix = self._server_multiaddr(port, self.announce_host) + "/p2p/"
                for addr in maddrs:
                    if addr.startswith(preferred_prefix):
                        self.initial_peer = addr
                        print(f"[SwarmManager] Bootstrap peer address: {self.initial_peer}")
                        return self.initial_peer
                self.initial_peer = maddrs[0]
                print(f"[SwarmManager] Bootstrap peer address: {self.initial_peer}")
                return self.initial_peer

            peer_match = re.search(r"(Qm[1-9A-HJ-NP-Za-km-z]+|12D3Koo[A-Za-z0-9]+)", content)
            if peer_match and ("Running a server on" in content or "Announced that blocks" in content):
                self.initial_peer = f"{self._server_multiaddr(port, self.announce_host)}/p2p/{peer_match.group(1)}"
                print(f"[SwarmManager] Bootstrap peer address (constructed): {self.initial_peer}")
                return self.initial_peer

            time.sleep(1)

        raise TimeoutError(
            f"Bootstrap server on port {port} did not announce peer ID within {timeout}s. "
            f"Check {self._log_path(port)} for errors."
        )

    def wait_for_server_start(self, port, timeout=180):
        """Wait until a server finishes startup and is ready for inference."""
        start = time.time()
        print(f"[SwarmManager] Waiting for server on port {port} to finish startup...")

        while time.time() - start < timeout:
            info = self.servers.get(port)
            if info is None:
                raise ValueError(f"No record of server on port {port}")

            proc = info["process"]
            content = self._read_log(port)
            if "Started" in content:
                return True
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server on port {port} exited with code {proc.returncode}. Check {info['log_path']}."
                )
            time.sleep(1)

        raise TimeoutError(f"Server on port {port} did not finish startup within {timeout}s.")

    def kill_server(self, port, method="SIGKILL"):
        """Kill a server process. method: SIGTERM, SIGKILL."""
        if port not in self.servers:
            print(f"[SwarmManager] No server on port {port}")
            return
        proc = self.servers[port]["process"]
        if proc.poll() is not None:
            print(f"[SwarmManager] Server on port {port} already dead")
            return
        if method == "SIGTERM":
            self._signal_process(proc, signal.SIGTERM)
        elif method == "SIGKILL":
            self._signal_process(proc, signal.SIGKILL)
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
        if info["log_file"]:
            info["log_file"].close()
        self.start_server(port, block_start, block_end, is_bootstrap=info.get("is_bootstrap", False))

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
                self._signal_process(proc, signal.SIGTERM)
                print(f"  Sent SIGTERM to server on port {port} (PID: {proc.pid})")
        time.sleep(5)
        for port, info in self.servers.items():
            proc = info["process"]
            if proc.poll() is None:
                self._signal_process(proc, signal.SIGKILL)
                print(f"  Force-killed server on port {port} (PID: {proc.pid})")
            if info.get("log_file"):
                info["log_file"].close()
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
            print(f"  Port {port}: blocks {blocks[0]}:{blocks[1]} - {status_str}{bootstrap_str}")
        print(f"  Initial peer: {self.initial_peer}")
        print("====================\n")

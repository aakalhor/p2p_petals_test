#!/usr/bin/env python3
"""Find the peer ID of the running bootstrap server by connecting to the p2pd daemon."""
import subprocess
import re
import sys

# Check the log from the early (failed) run that DID print the peer ID
# The peer ID is deterministic based on the identity key
# Let's try to read from the p2pd socket

# Strategy: Kill the old conda-wrapped process and start fresh without conda run
# using the actual python path directly

python_path = "/home/gellert/miniforge3/envs/petals/bin/python"
result = subprocess.run(
    [python_path, "-c", """
import hivemind
import asyncio

async def get_peer_id():
    # Create a DHT client that connects to the bootstrap server
    dht = hivemind.DHT(initial_peers=[], host_maddrs=['/ip4/0.0.0.0/tcp/0'], start=True)
    print(f"My peer ID: {dht.peer_id}")
    
    # Try to get visible peers
    visible = dht.get_visible_maddrs()
    for peer_id, addrs in visible.items():
        print(f"Visible peer: {peer_id} at {addrs}")
    dht.shutdown()

asyncio.run(get_peer_id())
"""],
    capture_output=True, text=True, timeout=30
)

print("STDOUT:", result.stdout)
if result.stderr:
    # Look for peer IDs in stderr (hivemind logs there)
    for line in result.stderr.split('\n'):
        if '12D3' in line or 'Qm' in line or 'peer' in line.lower():
            print("STDERR:", line)

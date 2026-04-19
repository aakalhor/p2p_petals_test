# P2P Petals Test

Private Petals swarm experiments for `huggyllama/llama-7b`, focused on launching multiple local Petals peers and measuring fault-tolerance behavior.

## Current Status

This repo is presentation-ready for:

- private swarm startup on one machine
- peer discovery and route construction
- archived experiment results from prior runs
- a clear discussion of the remaining runtime limitation

This repo is not currently validated for:

- successful end-to-end token generation in the current Windows/WSL or rented A40 environments

In both environments, the swarm starts correctly, the client connects, and Petals finds a valid route, but the first live inference span fails with `ConnectionResetError`, `BrokenPipeError`, or `TimeoutError`.

## What Was Fixed

- Removed the invalid `num_blocks + block_indices` combination from server startup.
- Fixed bootstrap peer extraction so the parsed multiaddr no longer includes a trailing quote.
- Switched local swarm startup to explicit Petals multiaddrs via `--host_maddrs` and `--announce_maddrs`.
- Added `--announce-host` support so peers can bind on one address and advertise another.
- Changed swarm readiness checks to wait for Petals `Started`, not only `Announced that blocks`.
- Added conservative local server defaults:
  - `--num_handlers 1`
  - `--prefetch_batches 1`
  - `--sender_threads 1`
  - `--attn_cache_tokens 4096`
  - `--balance_quality 0.0`
- Added client-side progress logging and periodic traceback dumps for stalled runs.
- Added runtime environment defaults for Linux-side temp/cache directories.

## Validated Outcome

Validated in WSL and on a rented A40 pod:

- `launch_swarm.py` starts a private swarm successfully.
- Non-bootstrap peers connect correctly.
- `swarm_config.json` is generated correctly at runtime.
- `test_client.py` loads the tokenizer and distributed model successfully.
- Petals reports a full route across the configured block ranges.

Remaining issue:

- the first inference RPC fails during `model.generate(...)`

## Historical Results

The following archived result files are present in `results/`:

- `baseline_4srv.json`
- `sigterm_4srv_60s.json`
- `sigkill_4srv_60s.json`
- `sigkill_4srv_30s.json`
- `sigkill_6srv_60s.json`
- `sigkill_6srv_repl.json`

Missing from the original matrix:

- `partition_4srv_60s.json`
- `slow_4srv.json`

Use these files as prior collected experiment results, not as fully revalidated outputs from the current environment.

## Quick Start

### WSL

Run inside Ubuntu WSL:

```bash
cd /mnt/c/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference
source /home/amirali/p2p_inference/.venv/bin/activate
export HF_HUB_DISABLE_XET=1
export TMPDIR=/tmp
export HF_HOME=$HOME/.cache/huggingface
python launch_swarm.py
```

In a second terminal:

```bash
cd /mnt/c/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference
source /home/amirali/p2p_inference/.venv/bin/activate
export HF_HUB_DISABLE_XET=1
export TMPDIR=/tmp
export HF_HOME=$HOME/.cache/huggingface
python test_client.py
```

### Linux GPU Server

For a rented Linux server:

```bash
cd /workspace/p2p_inference
. .venv/bin/activate
export HF_HUB_DISABLE_XET=1
export TMPDIR=/tmp
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
python launch_swarm.py --profile compact --bind-host 0.0.0.0 --announce-host <server-local-ip>
```

Then:

```bash
python test_client.py --max-new-tokens 16 --traceback-timeout 120
```

## Useful Commands

Launch the default four-peer topology:

```bash
python launch_swarm.py
```

Launch the lower-overhead two-peer topology:

```bash
python launch_swarm.py --profile compact
```

Try a different advertised address:

```bash
python launch_swarm.py --profile compact --bind-host 0.0.0.0 --announce-host 172.16.144.2
```

Run baseline experiments against an already running swarm:

```bash
python run_baseline.py --num_runs 10 --max_tokens 20
```

## Recommended Presentation Framing

- We fixed private swarm startup and peer discovery.
- We verified route construction across multiple Petals peers on one machine.
- We recovered historical experiment outputs from earlier runs.
- Live end-to-end inference is still blocked by a Petals runtime failure on the first remote span.

For a slide-ready summary, see [PRESENTATION_README.md](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/PRESENTATION_README.md).

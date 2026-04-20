# P2P Petals Test

Private Petals swarm experiments for presentation demos and small fault-tolerance runs.

## Current Status

This repo now has a real end-to-end Petals demo path that was validated on a rented Linux GPU server:

- environment: Ubuntu on `1x A40 48 GB`
- install path: current Petals from GitHub
- known-good model: `bigscience/bloom-7b1`
- successful profiles:
  - `single`: one server serving all 30 blocks
  - `compact`: two local peers serving `0:15` and `15:30`

Fresh successful artifacts are in [results/FRESH_RUNS.md](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/FRESH_RUNS.md), [results/baseline_compact_bigscience_bloom_7b1.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/baseline_compact_bigscience_bloom_7b1.json), and [results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json).

## What Was Fixed

The original repo issues on the private-swarm/orchestration side are fixed:

- removed the invalid `--num_blocks` + `--block_indices` combination
- fixed bootstrap peer extraction
- switched to explicit `--host_maddrs` and `--announce_maddrs`
- added separate bind vs announce host support
- wait for Petals `Started` before declaring a peer ready
- added conservative server defaults for single-GPU demos
- added `single` profile and model presets
- added exact server PID capture in `swarm_config.json` for churn demos

## Known-Good Demo Path

### Runtime

- Python `3.11`
- `torch 2.5.1+cu121`
- `petals 2.3.0.dev2`
- `hivemind 1.2.0.dev0`
- `transformers 4.43.1`

### Working model/profile combination

- default preset: `bloom-7b1`
- default launch profile: `single`
- reliable multi-peer demo: `compact`

The legacy `huggyllama/llama-7b` path is still available as `--preset llama-7b-legacy`, but it is not the recommended presentation path.

## Exact Setup Commands

Run these on a clean Linux GPU server:

```bash
cd /workspace
git clone https://github.com/aakalhor/p2p_petals_test.git p2p_inference
cd /workspace/p2p_inference
python3 -m venv .venv-main
. .venv-main/bin/activate
pip install --upgrade pip "setuptools<81" wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cython pybind11 grpcio-tools protobuf pyyaml
pip install --no-build-isolation git+https://github.com/bigscience-workshop/petals
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export TMPDIR=/tmp
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
```

## Reproduce The Demo

### 1. Single-peer smoke test

Terminal 1:

```bash
cd /workspace/p2p_inference
. .venv-main/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TMPDIR=/tmp
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
python launch_swarm.py --profile single --bind-host 0.0.0.0 --announce-host <server-ip>
```

Terminal 2:

```bash
cd /workspace/p2p_inference
. .venv-main/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TMPDIR=/tmp
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
python -u test_client.py --max-new-tokens 16 --traceback-timeout 120
```

### 2. Two-peer smoke test

Terminal 1:

```bash
cd /workspace/p2p_inference
. .venv-main/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TMPDIR=/tmp
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
python launch_swarm.py --profile compact --bind-host 0.0.0.0 --announce-host <server-ip>
```

Terminal 2:

```bash
cd /workspace/p2p_inference
. .venv-main/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TMPDIR=/tmp
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
python -u test_client.py --max-new-tokens 16 --traceback-timeout 120
python -u run_baseline.py --num_runs 5 --max_tokens 12
python -u run_fault_smoke.py --max-new-tokens 8 --timeout 90 --fault-method SIGTERM
```

## Fresh Results In This Repo

- [results/FRESH_RUNS.md](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/FRESH_RUNS.md): exact commands and log excerpts
- [results/baseline_compact_bigscience_bloom_7b1.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/baseline_compact_bigscience_bloom_7b1.json): fresh compact baseline
- [results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json): small churn run

Historical archived outputs from the original repo are still present in `results/`, but the presentation should prioritize the fresh BLOOM/A40 artifacts above.

## Presentation-Safe Claim

We repaired the private-swarm orchestration, switched to a current Petals environment and a more reproducible model, and validated a real end-to-end Petals demo:

- successful single-server generation
- successful two-peer generation with a real multi-hop route
- fresh baseline output
- fresh small churn output with a verified peer kill

For the slide outline and framing, see [PRESENTATION_README.md](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/PRESENTATION_README.md).

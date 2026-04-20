# Fresh Demo Runs

These are the fresh presentation artifacts collected from the validated A40/Linux path.

## Environment

- server: Ubuntu Linux on `1x A40 48 GB`
- venv: `.venv-main`
- model: `bigscience/bloom-7b1`
- Petals install source: GitHub `main` path

The environment variables used for every run were:

```bash
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export TMPDIR=/tmp
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
```

## Exact Commands Used

### Single-peer smoke test

```bash
python launch_swarm.py --profile single --bind-host 0.0.0.0 --announce-host 172.16.144.2
python -u test_client.py --max-new-tokens 16 --traceback-timeout 120
```

### Two-peer smoke test

```bash
python launch_swarm.py --profile compact --bind-host 0.0.0.0 --announce-host 172.16.144.2
python -u test_client.py --max-new-tokens 16 --traceback-timeout 120
```

### Compact baseline

```bash
python -u run_baseline.py --num_runs 5 --max_tokens 12
```

### Compact churn demo

```bash
python -u run_fault_smoke.py --max-new-tokens 8 --timeout 90 --fault-method SIGTERM
```

## Single-Peer Successful Log

```text
[2026-04-19 23:42:13] Model: bigscience/bloom-7b1
[2026-04-19 23:42:14] Connecting to swarm and loading distributed model...
[2026-04-19 23:42:18] Distributed model is ready.
[2026-04-19 23:42:18] Generating 16 tokens...
[2026-04-19 23:42:25] Generation finished in 7.62s
[2026-04-19 23:42:25] Output: The future of robotics is here. The future of robotics is here. The future of robotics is
[2026-04-19 23:42:25] Smoke test passed.
Apr 19 23:42:18.211 [INFO] Route found: 0:30 via ...jPJ2jq
```

## Two-Peer Successful Log

```text
[2026-04-19 23:45:46] Model: bigscience/bloom-7b1
[2026-04-19 23:45:47] Connecting to swarm and loading distributed model...
[2026-04-19 23:45:50] Distributed model is ready.
[2026-04-19 23:45:50] Generating 16 tokens...
[2026-04-19 23:45:58] Generation finished in 8.23s
[2026-04-19 23:45:58] Output: The future of robotics is here. The future of robotics is here. The future of robotics is
[2026-04-19 23:45:58] Smoke test passed.
Apr 19 23:45:50.767 [INFO] Route found: 0:15 via ...7K5oaQ => 15:30 via ...cNPBJF
```

## Baseline Result

Saved file:

- [baseline_compact_bigscience_bloom_7b1.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/baseline_compact_bigscience_bloom_7b1.json)

Fresh summary from the successful run:

```text
Successful runs: 5/5 (0 failures)
Latency:
  p50: 6.507s
  p95: 10.630s
  p99: 11.036s
  mean: 7.765s
Throughput:
  mean: 1.63 tok/s
```

## Churn Result

Saved file:

- [fault_smoke_compact_bigscience_bloom_7b1_sigterm.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json)

Observed behavior:

- before fault: request succeeded
- fault: non-bootstrap peer on port `31338` killed with `SIGTERM`
- killed PID recorded: `23128`
- after fault: request timed out after `90s`

This is a valid small failure artifact for the presentation because it shows:

- the swarm was healthy before fault
- the fault was actually injected
- the post-fault request degraded as expected without redundancy

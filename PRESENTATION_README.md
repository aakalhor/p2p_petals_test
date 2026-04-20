# Presentation Notes

## One-Slide Summary

We recovered the private-swarm orchestration, moved the demo to a current Petals stack on Linux, switched from the fragile legacy `llama-7b` path to `bigscience/bloom-7b1`, and produced fresh end-to-end results:

- single-peer generation succeeded
- two-peer generation succeeded with a real route across both peers
- a fresh 5-run baseline succeeded
- a small churn demo succeeded in killing one peer and recording the degraded outcome

## Project Goal

Show a real Petals private swarm running an LLM in a P2P layout, then present one baseline result and one small failure result.

## What Was Broken Originally

- invalid Petals CLI combination in swarm startup
- brittle bootstrap peer extraction
- localhost/private-address startup issues
- legacy `huggyllama/llama-7b` path failing on the first remote inference hop

## What We Changed

### Swarm orchestration

- fixed Petals CLI arguments
- switched to explicit multiaddrs via `--host_maddrs` and `--announce_maddrs`
- added bind-host vs announce-host support
- wait for `Started` before marking peers ready
- added `single`, `compact`, and `presentation` profiles

### Known-good demo path

- added model presets
- made `bigscience/bloom-7b1` the default demo preset
- made `single` the default launch profile
- saved exact server PIDs into `swarm_config.json` for fault demos

### Diagnostics and experiments

- `test_client.py` now logs readiness, token generation, and tracebacks
- `run_baseline.py` writes model/profile-specific outputs
- `run_fault_smoke.py` wraps the smoke client and records a real peer kill

## Validated Environment

- Ubuntu Linux on rented server
- `1x A40 48 GB`
- Python `3.11`
- `torch 2.5.1+cu121`
- `petals 2.3.0.dev2`
- `transformers 4.43.1`

## Fresh Results You Can Show

### Single-peer smoke test

- model ready
- route found across `0:30`
- successful generation of 16 new tokens

### Two-peer smoke test

- model ready
- route found across `0:15 => 15:30`
- successful generation of 16 new tokens

### Baseline

- `5/5` successful requests
- saved in [results/baseline_compact_bigscience_bloom_7b1.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/baseline_compact_bigscience_bloom_7b1.json)

### Churn demo

- `SIGTERM` sent to the non-bootstrap peer
- exact killed PID recorded
- saved in [results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json)

## Slide Outline

1. Problem
   Private Petals swarm demo for P2P LLM inference and fault behavior.

2. Initial blockers
   Bad CLI arguments, bad peer parsing, legacy model/runtime path failing.

3. Fixes
   Correct swarm startup, private addressing, profiles, diagnostics, and new demo model preset.

4. Environment
   A40 server, latest Petals from GitHub, BLOOM-7.1B.

5. Fresh evidence
   Show single-peer smoke log, two-peer route log, baseline JSON summary, and churn JSON summary.

6. Final claim
   We achieved a real end-to-end Petals private-swarm demo and captured fresh presentation-safe outputs.

## Suggested Live Demo

If you only have time for one live demo in class:

1. run `python launch_swarm.py --profile compact --bind-host 0.0.0.0 --announce-host <server-ip>`
2. run `python -u test_client.py --max-new-tokens 16 --traceback-timeout 120`
3. point to the `Route found: 0:15 ... => 15:30 ...` line
4. show the generated text
5. use the saved baseline and churn JSON files on the next slide

## Important Framing

The original `huggyllama/llama-7b` private-swarm path is still useful as a debugging story, but it is not the presentation demo path anymore. The presentation should focus on the validated current-stack BLOOM path.

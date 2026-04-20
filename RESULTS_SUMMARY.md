# Results Summary

## Final Validated Demo Path

- environment: Ubuntu Linux on `1x A40 48 GB`
- install: current Petals from GitHub in a clean venv
- model: `bigscience/bloom-7b1`
- working profiles:
  - `single`
  - `compact`

## Fresh Successful Outputs

- single-peer smoke test: success
- two-peer smoke test: success
- compact baseline: `5/5` success
- compact fault smoke: peer `31338` killed with `SIGTERM`, failure recorded after fault

## Fresh Artifact Files

- [results/FRESH_RUNS.md](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/FRESH_RUNS.md)
- [results/baseline_compact_bigscience_bloom_7b1.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/baseline_compact_bigscience_bloom_7b1.json)
- [results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json](C:/Users/amirali/Desktop/Adv_Dist/P2P/p2p_inference/results/fault_smoke_compact_bigscience_bloom_7b1_sigterm.json)

## Presentation-Safe Claim

We fixed the private-swarm orchestration and validated a real Petals P2P LLM demo on a current Linux Petals stack. The repo now contains fresh successful generation logs, a fresh baseline result, and a fresh small churn result.

# Presentation Notes

## One-Slide Summary

We adapted a Petals-based private swarm repo so it can reliably launch multiple local peers on one machine, fixed bootstrap discovery and localhost-safe addressing, and verified that the client can connect and build full routes across the swarm. Archived experiment outputs exist for baseline and churn scenarios. The remaining open issue is that live generation fails on the first remote inference span in the currently tested environments.

## Project Goal

The repo is intended to launch multiple local Petals servers for `huggyllama/llama-7b`, then evaluate baseline performance and fault behavior under server churn.

## What Was Broken Originally

- `swarm_manager.py` passed both `--num_blocks` and `--block_indices`, which Petals rejects.
- Bootstrap peer parsing could capture a trailing quote from the logged multiaddr.
- Local peer startup logic was brittle for one-machine localhost/private use.

## What Was Fixed

### Startup and addressing

- Removed the conflicting Petals CLI arguments.
- Switched to explicit Petals multiaddrs with `--host_maddrs` and `--announce_maddrs`.
- Added support for separate bind and announce hosts.
- Fixed bootstrap peer extraction.

### Readiness and debugging

- Changed startup checks to wait for Petals `Started`.
- Added conservative local server defaults to reduce process overhead.
- Added client-side logging and faulthandler traces so stalls are diagnosable.
- Added Linux-side temp/cache defaults for WSL and Linux GPU servers.

## What We Successfully Validated

### On Windows laptop with WSL

- Four-peer swarm startup works.
- Bootstrap and non-bootstrap peer connection works.
- `swarm_config.json` is generated correctly.
- The client loads the distributed model and discovers a full route.

### On rented A40 GPU server

- Clean environment build succeeded.
- Two-peer and four-peer private swarms both start successfully.
- The client loads the distributed model successfully.
- Route construction works with both localhost and real pod IP advertisement.

## Remaining Limitation

The first live inference span fails with:

- `ConnectionResetError`
- `BrokenPipeError`
- `TimeoutError`

This happens after:

- swarm formation
- client connection
- model loading
- route discovery

So the remaining issue is not peer discovery or startup orchestration. It is a Petals runtime failure during the first remote inference step.

## What Results We Have

Archived JSON result files:

- `results/baseline_4srv.json`
- `results/sigterm_4srv_60s.json`
- `results/sigkill_4srv_60s.json`
- `results/sigkill_4srv_30s.json`
- `results/sigkill_6srv_60s.json`
- `results/sigkill_6srv_repl.json`

These are presentation-usable as historical collected outputs.

## What Results We Do Not Have

- no successful fresh end-to-end generation result from the current validated environments
- missing experiment files for:
  - `partition_4srv_60s`
  - `slow_4srv`

## Honest Presentation Conclusion

The engineering recovery was successful for swarm startup, peer discovery, and route formation. The repo is now much more robust and reproducible for the orchestration part of the project. However, full end-to-end generation is still blocked by a Petals inference runtime failure in the tested environments, so the experiment matrix is only partially reproducible today.

## Suggested Slide Outline

1. Problem
   Run a private Petals swarm for `llama-7b` and test P2P inference under failures.

2. Initial blockers
   Bad Petals CLI arguments, broken bootstrap parsing, localhost swarm startup issues.

3. Fixes
   Correct Petals startup arguments, localhost-safe multiaddrs, stronger readiness checks, better diagnostics.

4. Validated achievements
   Local multi-peer swarm startup works on WSL and on a rented A40 server.

5. Evidence
   Show `launch_swarm.py` output, route-found client logs, and archived JSON results.

6. Remaining issue
   Generation fails on first remote inference span with connection resets/timeouts.

7. Takeaway
   Swarm orchestration is repaired; runtime inference remains the open systems issue.

## Suggested Demo Plan

- Show `python launch_swarm.py` starting the private swarm.
- Show the client reaching `Distributed model is ready`.
- Show the `Route found: ...` log line.
- Then explain that the remaining failure is at the first live inference RPC.

That is still a defensible systems presentation because it distinguishes orchestration success from serving-runtime failure.

# Results Summary

## Validated Fixes

- Petals servers now launch with valid CLI arguments.
- Bootstrap peer extraction works.
- Private swarm peers connect correctly on one machine.
- Local/private addressing is configurable for localhost or server IPs.
- Swarm startup waits for Petals `Started`.

## Validated Behaviors

- `launch_swarm.py` works for:
  - `presentation` profile: four peers
  - `compact` profile: two peers
- `test_client.py` loads the tokenizer and distributed model.
- Petals discovers valid routes across the configured block ranges.

## Remaining Failure

Generation fails during the first remote span with repeated:

- `ConnectionResetError`
- `BrokenPipeError`
- `TimeoutError`

Observed on:

- Windows laptop with WSL
- rented Linux A40 server
- `nf4` and `int8`
- localhost and real pod IP advertisement
- two-peer and four-peer layouts

## Archived Experiment Files

- `results/baseline_4srv.json`
- `results/sigterm_4srv_60s.json`
- `results/sigkill_4srv_60s.json`
- `results/sigkill_4srv_30s.json`
- `results/sigkill_6srv_60s.json`
- `results/sigkill_6srv_repl.json`

## Missing Files

- `results/partition_4srv_60s.json`
- `results/slow_4srv.json`

## Presentation-Safe Claim

We repaired the swarm orchestration layer and recovered historical experiment outputs, but full end-to-end inference remains blocked by a Petals runtime failure during the first live remote forward pass.

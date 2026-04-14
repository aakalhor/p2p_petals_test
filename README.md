# P2P LLM Inference Research

**Thesis**: P2P LLM inference via Petals is brittle under node churn; this paper characterizes the degradation and proposes replication and routing improvements to mitigate it.

## Setup

```bash
conda activate petals
huggingface-cli login  # Need Llama 2 access approved
```

## Usage

### 1. Launch the swarm (4 servers)
```bash
python launch_swarm.py
```

### 2. Run baseline measurements
```bash
python run_baseline.py
```

### 3. Run all experiments
```bash
python run_experiments.py
```

### 4. Generate analysis plots
```bash
python analyze_results.py
```

## Project Structure
- `launch_swarm.py` — Launches 4 Petals servers in a private swarm
- `swarm_manager.py` — SwarmManager class for managing servers + fault injection
- `run_baseline.py` — Phase 3: Baseline measurements (N=50 inference runs)
- `run_churn_experiment.py` — Phase 4: Churn experiment runner
- `run_experiments.py` — Phase 5: Full experiment matrix (Baseline + E1-E7)
- `analyze_results.py` — Phase 6: Analysis and plot generation
- `test_client.py` — Quick smoke test for client inference
- `results/` — Experiment output JSON files
- `plots/` — Generated analysis plots

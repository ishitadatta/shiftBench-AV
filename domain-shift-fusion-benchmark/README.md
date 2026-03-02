# Domain Shift / Benchmarking / Fusion Robustness

Open-source Python toolkit for **uncertainty-aware sensor-fusion benchmarking** under **domain shift** and **sensor degradation**.

This repository is designed for research and reproducible benchmarking workflows, with a practical focus on:
- uncertainty-aware fusion evaluation
- domain-shift scenario stress testing
- sensor perturbation and fault injection
- reproducible benchmark reporting

## Capabilities

- Uncertainty-aware fusion (inverse-variance weighting)
- Sensor perturbation engine:
  - `gaussian_noise`
  - `dropout`
  - `bias_drift`
  - `temporal_jitter`
  - `uncertainty_inflation`
- Scenario-driven domain shift benchmarking via JSON config
- Dataset support:
  - synthetic generated dataset
  - SQLite table dataset
- Metrics:
  - Accuracy
  - Negative Log-Likelihood (NLL)
  - Expected Calibration Error (ECE)
  - Uncertainty-Risk Correlation
- CLI tools for benchmark execution and demo DB creation

## Repository structure

```text
domain-shift-fusion-benchmark/
  configs/
    sample_benchmark.json
    sqlite_benchmark.json
  examples/
    run_sqlite_demo.sh
  src/fusionbench/
    bench/
      metrics.py
      runner.py
    cli/
      main.py
    core/
      types.py
      utils.py
    data/
      synthetic.py
      sqlite_store.py
    domain_shift/
      scenarios.py
    fusion/
      uncertainty.py
    perturbations/
      operators.py
  tests/
  pyproject.toml
  README.md
```

## Prerequisites

- Python 3.9+
- macOS/Linux/Windows shell

## Installation

### 1. Create and activate virtual environment

```bash
cd /Users/hrudhairajasekhar/Projects/ishita-datta/domain-shift-fusion-benchmark
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install package in editable mode

```bash
pip install -e .
```

### 3. Optional: install dev dependencies

```bash
pip install -e .[dev]
```

### Offline/no-network fallback

If your environment cannot download packages, you can still run this project directly:

```bash
cd /Users/hrudhairajasekhar/Projects/ishita-datta/domain-shift-fusion-benchmark
PYTHONPATH=src python3 -m fusionbench.cli.main run --config configs/sample_benchmark.json --output examples/sample_results.json
```

## Quick start (synthetic dataset)

Run the included benchmark config:

```bash
fusionbench run \
  --config configs/sample_benchmark.json \
  --output examples/sample_results.json
```

The command prints JSON to terminal and writes output to `examples/sample_results.json`.

## SQLite-backed benchmarking workflow

### 1. Create demo SQLite dataset

```bash
fusionbench make-demo-db \
  --db-path examples/demo_samples.db \
  --n-samples 1500 \
  --seed 11
```

### 2. Run benchmark using SQLite dataset

```bash
fusionbench run \
  --config configs/sqlite_benchmark.json \
  --output examples/sqlite_results.json
```

### 3. One-command demo script

```bash
bash examples/run_sqlite_demo.sh
```

## CLI reference

### `fusionbench run`

Run benchmark from config.

```bash
fusionbench run --config <path/to/config.json> --output <path/to/results.json>
```

### `fusionbench make-demo-db`

Create a synthetic SQLite dataset.

```bash
fusionbench make-demo-db --db-path <db.sqlite> [--n-samples 1500] [--seed 42]
```

## Config format

Top-level JSON keys:
- `benchmark`
- `dataset`
- `scenarios`

### `benchmark`

```json
{
  "name": "domain_shift_fusion_baseline",
  "seed": 42,
  "calibration_bins": 10
}
```

### `dataset` (synthetic)

```json
{
  "source": "synthetic",
  "n_samples": 1500,
  "seed": 123
}
```

### `dataset` (sqlite)

```json
{
  "source": "sqlite",
  "db_path": "examples/demo_samples.db",
  "table": "samples",
  "limit": 1000
}
```

### `scenarios`

Each scenario supports an `operations` list of perturbations.

```json
{
  "name": "compound_night_shift",
  "description": "Multimodal degradation under low light and weather",
  "operations": [
    { "type": "gaussian_noise", "target": "camera", "std": 0.11 },
    { "type": "dropout", "target": "lidar", "probability": 0.15 },
    { "type": "bias_drift", "target": "radar", "offset": 0.05 },
    { "type": "temporal_jitter", "target": "all", "window": 4 },
    { "type": "uncertainty_inflation", "target": "all", "factor": 1.2 }
  ]
}
```

## Metrics interpretation

- **Accuracy**: classification correctness under thresholded fused score
- **NLL**: probabilistic quality (lower is better)
- **ECE**: calibration gap between confidence and empirical accuracy (lower is better)
- **Uncertainty-Risk Correlation**: whether higher uncertainty tracks prediction failures

## Testing

Run the unit test suite:

```bash
pytest
```

## Reproducibility guidance

- Keep `seed` fixed in both benchmark and dataset configs.
- Version-control your config files and output JSON.
- Compare scenarios using identical dataset and model settings.

## Extending this project

1. Plug in real sensor datasets by implementing an additional loader under `src/fusionbench/data/`.
2. Add advanced fusion methods under `src/fusionbench/fusion/`.
3. Add perturbation types (occlusion masks, blur kernels, packet loss models) in `src/fusionbench/perturbations/operators.py`.
4. Add domain-specific metrics in `src/fusionbench/bench/metrics.py`.

## Typical troubleshooting

- `fusionbench: command not found`
  - Ensure virtual environment is active and `pip install -e .` completed successfully.
- `dataset.db_path is required`
  - Set `dataset.source` to `sqlite` only when `db_path` is provided.
- SQLite table errors
  - Confirm the target table exists and columns match expected schema.

## License

MIT (see `LICENSE`).

## Last verified

Verified on **March 2, 2026** with:

- `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q` -> `4 passed`
- `PYTHONPATH=src python3 -m fusionbench.cli.main run --config configs/sample_benchmark.json --output examples/sample_results.json`
- `PYTHONPATH=src python3 -m fusionbench.cli.main make-demo-db --db-path examples/demo_samples.db --n-samples 1500 --seed 11`
- `PYTHONPATH=src python3 -m fusionbench.cli.main run --config configs/sqlite_benchmark.json --output examples/sqlite_results.json`

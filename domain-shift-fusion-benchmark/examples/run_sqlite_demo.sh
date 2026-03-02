#!/usr/bin/env bash
set -euo pipefail

fusionbench make-demo-db --db-path examples/demo_samples.db --n-samples 1500 --seed 11
fusionbench run --config configs/sqlite_benchmark.json --output examples/sqlite_results.json

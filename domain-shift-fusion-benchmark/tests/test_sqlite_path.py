from pathlib import Path

from fusionbench.bench.runner import run_benchmark
from fusionbench.data.sqlite_store import create_demo_database


def test_sqlite_dataset_roundtrip(tmp_path: Path):
    db_file = tmp_path / "demo.db"
    create_demo_database(str(db_file), n_samples=200, seed=2)

    config = {
        "benchmark": {"name": "sqlite", "seed": 2},
        "dataset": {
            "source": "sqlite",
            "db_path": str(db_file),
            "table": "samples",
            "limit": 100,
        },
        "scenarios": [{"name": "baseline", "operations": []}],
    }

    result = run_benchmark(config)
    assert len(result.scenario_results) == 1
    assert result.scenario_results[0].n_samples == 100

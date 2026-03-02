from fusionbench.bench.runner import benchmark_result_to_dict, run_benchmark


def test_runner_executes_minimal_config():
    config = {
        "benchmark": {"name": "smoke", "seed": 4},
        "dataset": {"source": "synthetic", "n_samples": 120, "seed": 4},
        "scenarios": [
            {"name": "baseline", "operations": []},
            {
                "name": "noise",
                "operations": [{"type": "gaussian_noise", "target": "all", "std": 0.05}],
            },
        ],
    }

    result = run_benchmark(config)
    payload = benchmark_result_to_dict(result)

    assert payload["benchmark_name"] == "smoke"
    assert len(payload["scenarios"]) == 2
    assert 0.0 <= payload["scenarios"][0]["accuracy"] <= 1.0

from __future__ import annotations

from typing import Dict, List

from fusionbench.bench.metrics import aggregate_metrics
from fusionbench.core.types import BenchmarkResult, ScenarioResult, Sample
from fusionbench.data.sqlite_store import load_samples_from_database
from fusionbench.data.synthetic import generate_synthetic_samples
from fusionbench.domain_shift.scenarios import ShiftScenario, built_in_scenarios
from fusionbench.fusion.uncertainty import InverseVarianceFusion
from fusionbench.perturbations.operators import PerturbationEngine


def _load_dataset(dataset_cfg: Dict) -> List[Sample]:
    source = str(dataset_cfg.get("source", "synthetic")).strip().lower()
    if source == "synthetic":
        return generate_synthetic_samples(
            n_samples=int(dataset_cfg.get("n_samples", 1500)),
            seed=int(dataset_cfg.get("seed", 42)),
        )
    if source == "sqlite":
        db_path = dataset_cfg.get("db_path")
        if not db_path:
            raise ValueError("dataset.db_path is required when dataset.source=sqlite")
        return load_samples_from_database(
            db_path=str(db_path),
            table=str(dataset_cfg.get("table", "samples")),
            limit=int(dataset_cfg["limit"]) if dataset_cfg.get("limit") is not None else None,
        )

    raise ValueError(f"Unsupported dataset source: {source}")


def _resolve_scenarios(config_scenarios: List[Dict] | None) -> List[ShiftScenario]:
    if not config_scenarios:
        return built_in_scenarios()

    resolved: List[ShiftScenario] = []
    for item in config_scenarios:
        resolved.append(
            ShiftScenario(
                name=str(item.get("name", "unnamed_scenario")),
                description=str(item.get("description", "")),
                operations=list(item.get("operations", [])),
            )
        )
    return resolved


def run_benchmark(config: Dict) -> BenchmarkResult:
    benchmark_cfg = config.get("benchmark", {})
    benchmark_name = str(benchmark_cfg.get("name", "fusion_robustness_benchmark"))
    seed = int(benchmark_cfg.get("seed", 42))
    calibration_bins = int(benchmark_cfg.get("calibration_bins", 10))

    dataset = _load_dataset(config.get("dataset", {}))
    scenarios = _resolve_scenarios(config.get("scenarios"))

    perturb_engine = PerturbationEngine(seed=seed)
    fusion_model = InverseVarianceFusion()

    scenario_results: List[ScenarioResult] = []

    for scenario in scenarios:
        scenario_samples = perturb_engine.apply(dataset, scenario.operations)
        labels: List[int] = []
        probs: List[float] = []
        fused_uncertainties: List[float] = []

        for sample in scenario_samples:
            fused = fusion_model.fuse(sample)
            labels.append(sample.label)
            probs.append(fused.fused_score)
            fused_uncertainties.append(fused.fused_uncertainty)

        acc, nll, ece, urc = aggregate_metrics(labels, probs, fused_uncertainties, calibration_bins=calibration_bins)

        scenario_results.append(
            ScenarioResult(
                name=scenario.name,
                n_samples=len(scenario_samples),
                accuracy=acc,
                negative_log_likelihood=nll,
                expected_calibration_error=ece,
                uncertainty_risk_correlation=urc,
            )
        )

    return BenchmarkResult(benchmark_name=benchmark_name, seed=seed, scenario_results=scenario_results)


def benchmark_result_to_dict(result: BenchmarkResult) -> Dict:
    rows = []
    for item in result.scenario_results:
        rows.append(
            {
                "name": item.name,
                "n_samples": item.n_samples,
                "accuracy": round(item.accuracy, 6),
                "negative_log_likelihood": round(item.negative_log_likelihood, 6),
                "expected_calibration_error": round(item.expected_calibration_error, 6),
                "uncertainty_risk_correlation": round(item.uncertainty_risk_correlation, 6),
            }
        )

    return {
        "benchmark_name": result.benchmark_name,
        "seed": result.seed,
        "scenarios": rows,
    }



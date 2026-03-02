from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SensorReading:
    score: float
    uncertainty: float


@dataclass
class Sample:
    sample_id: int
    label: int
    sensors: Dict[str, SensorReading]
    timestamp: Optional[int] = None


@dataclass
class ScenarioResult:
    name: str
    n_samples: int
    accuracy: float
    negative_log_likelihood: float
    expected_calibration_error: float
    uncertainty_risk_correlation: float


@dataclass
class BenchmarkResult:
    benchmark_name: str
    seed: int
    scenario_results: List[ScenarioResult]


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from fusionbench.core.types import Sample
from fusionbench.core.utils import clamp


@dataclass
class FusionOutput:
    fused_score: float
    fused_uncertainty: float
    per_sensor_weights: Dict[str, float]


class InverseVarianceFusion:
    """Simple uncertainty-aware fusion baseline using inverse-variance weighting."""

    def __init__(self, minimum_uncertainty: float = 1e-3):
        self.minimum_uncertainty = minimum_uncertainty

    def fuse(self, sample: Sample) -> FusionOutput:
        weights: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

        for sensor_name, reading in sample.sensors.items():
            variance = max(self.minimum_uncertainty ** 2, reading.uncertainty ** 2)
            w = 1.0 / variance
            weights[sensor_name] = w
            weighted_sum += w * reading.score
            weight_total += w

        if weight_total <= 0.0:
            return FusionOutput(fused_score=0.5, fused_uncertainty=1.0, per_sensor_weights={})

        normalized = {k: v / weight_total for k, v in weights.items()}
        fused_score = clamp(weighted_sum / weight_total)
        fused_uncertainty = clamp((1.0 / weight_total) ** 0.5, 0.01, 1.0)
        return FusionOutput(fused_score=fused_score, fused_uncertainty=fused_uncertainty, per_sensor_weights=normalized)


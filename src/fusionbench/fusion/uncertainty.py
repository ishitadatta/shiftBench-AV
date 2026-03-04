from __future__ import annotations

from dataclasses import dataclass
import math
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


class AdaptiveReliabilityFusion:
    """
    Context-aware reliability fusion with disagreement-aware confidence tempering.

    This model extends inverse-variance weighting by:
    - applying modality priors,
    - suppressing sensors that disagree with the consensus,
    - tempering fused confidence using disagreement and aleatoric uncertainty.
    """

    def __init__(
        self,
        minimum_uncertainty: float = 1e-3,
        uncertainty_power: float = 1.7,
        disagreement_power: float = 1.3,
        disagreement_gain: float = 2.0,
        temperature_gain: float = 2.8,
        modality_priors: Dict[str, float] | None = None,
    ):
        self.minimum_uncertainty = minimum_uncertainty
        self.uncertainty_power = uncertainty_power
        self.disagreement_power = disagreement_power
        self.disagreement_gain = disagreement_gain
        self.temperature_gain = temperature_gain
        self.modality_priors = modality_priors or {"camera": 1.0, "lidar": 1.15, "radar": 0.95}

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _logit(prob: float) -> float:
        p = clamp(prob, 1e-6, 1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    def fuse(self, sample: Sample) -> FusionOutput:
        if not sample.sensors:
            return FusionOutput(fused_score=0.5, fused_uncertainty=1.0, per_sensor_weights={})

        base_weights: Dict[str, float] = {}
        for sensor_name, reading in sample.sensors.items():
            prior = max(1e-3, float(self.modality_priors.get(sensor_name, 1.0)))
            reliability = (1.0 - clamp(reading.uncertainty, 0.0, 1.0)) ** self.uncertainty_power
            base_weights[sensor_name] = max(1e-9, prior * reliability)

        base_total = sum(base_weights.values())
        if base_total <= 0.0:
            return FusionOutput(fused_score=0.5, fused_uncertainty=1.0, per_sensor_weights={})

        base_fused = sum(base_weights[name] * sample.sensors[name].score for name in base_weights) / base_total

        adjusted_weights: Dict[str, float] = {}
        for sensor_name, reading in sample.sensors.items():
            disagreement = abs(reading.score - base_fused)
            agreement_term = 1.0 / (1.0 + self.disagreement_gain * (disagreement ** self.disagreement_power))
            adjusted_weights[sensor_name] = base_weights[sensor_name] * agreement_term

        adjusted_total = sum(adjusted_weights.values())
        if adjusted_total <= 0.0:
            return FusionOutput(fused_score=0.5, fused_uncertainty=1.0, per_sensor_weights={})

        normalized = {name: weight / adjusted_total for name, weight in adjusted_weights.items()}
        fused_score_raw = sum(normalized[name] * sample.sensors[name].score for name in normalized)

        weighted_aleatoric = sum(normalized[name] * sample.sensors[name].uncertainty for name in normalized)
        weighted_disagreement = sum(normalized[name] * abs(sample.sensors[name].score - fused_score_raw) for name in normalized)

        fused_uncertainty = clamp(
            0.55 * weighted_aleatoric + 0.45 * clamp(1.5 * weighted_disagreement, 0.0, 1.0),
            0.01,
            1.0,
        )

        temperature = 1.0 + self.temperature_gain * fused_uncertainty
        fused_score = self._sigmoid(self._logit(fused_score_raw) / temperature)
        return FusionOutput(fused_score=clamp(fused_score), fused_uncertainty=fused_uncertainty, per_sensor_weights=normalized)


class CounterfactualConsensusFusion:
    """
    Counterfactual-consensus fusion.

    Core idea:
    - build base reliability from modality priors and uncertainty,
    - reward sensors that agree with peers in logit space,
    - penalize sensors whose leave-one-out removal causes unstable predictions.
    """

    def __init__(
        self,
        evidence_power: float = 1.5,
        agreement_gamma: float = 2.0,
        counterfactual_beta: float = 6.0,
        modality_priors: Dict[str, float] | None = None,
    ):
        self.evidence_power = evidence_power
        self.agreement_gamma = agreement_gamma
        self.counterfactual_beta = counterfactual_beta
        self.modality_priors = modality_priors or {"camera": 1.0, "lidar": 1.15, "radar": 1.0}

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _logit(prob: float) -> float:
        p = clamp(prob, 1e-6, 1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    @staticmethod
    def _weighted_mean(values: Dict[str, float], weights: Dict[str, float]) -> float:
        total = sum(weights.values())
        if total <= 0.0:
            return 0.0
        return sum(values[name] * weights[name] for name in weights) / total

    def fuse(self, sample: Sample) -> FusionOutput:
        if not sample.sensors:
            return FusionOutput(fused_score=0.5, fused_uncertainty=1.0, per_sensor_weights={})

        logits = {name: self._logit(reading.score) for name, reading in sample.sensors.items()}
        uncertainties = {name: clamp(reading.uncertainty, 0.0, 1.0) for name, reading in sample.sensors.items()}

        base_weights: Dict[str, float] = {}
        for name in sample.sensors:
            prior = max(1e-3, float(self.modality_priors.get(name, 1.0)))
            evidence = (1.0 - uncertainties[name]) ** self.evidence_power
            base_weights[name] = max(1e-9, prior * evidence)

        base_logit = self._weighted_mean(logits, base_weights)
        base_prob = self._sigmoid(base_logit)

        names = list(sample.sensors.keys())
        support: Dict[str, float] = {}
        for name_i in names:
            if len(names) == 1:
                support[name_i] = 1.0
                continue
            score = 0.0
            norm = 0.0
            for name_j in names:
                if name_j == name_i:
                    continue
                agreement = math.exp(-self.agreement_gamma * abs(logits[name_i] - logits[name_j]))
                score += agreement * base_weights[name_j]
                norm += base_weights[name_j]
            support[name_i] = score / max(norm, 1e-9)

        cfi: Dict[str, float] = {}
        for name in names:
            loo_weights = {k: v for k, v in base_weights.items() if k != name}
            loo_logits = {k: v for k, v in logits.items() if k != name}
            loo_prob = self._sigmoid(self._weighted_mean(loo_logits, loo_weights))
            delta = abs(base_prob - loo_prob)
            cfi[name] = math.exp(-self.counterfactual_beta * delta)

        final_weights = {
            name: base_weights[name] * (0.5 * support[name] + 0.5 * cfi[name]) for name in names
        }
        total = sum(final_weights.values())
        if total <= 0.0:
            return FusionOutput(fused_score=0.5, fused_uncertainty=1.0, per_sensor_weights={})
        normalized = {name: w / total for name, w in final_weights.items()}

        fused_logit = self._weighted_mean(logits, normalized)
        fused_score = self._sigmoid(fused_logit)

        aleatoric = sum(normalized[name] * uncertainties[name] for name in names)
        disagreement = sum(normalized[name] * abs(self._sigmoid(logits[name]) - fused_score) for name in names)
        instability = sum(normalized[name] * (1.0 - cfi[name]) for name in names)
        fused_uncertainty = clamp(0.45 * aleatoric + 0.35 * disagreement + 0.20 * instability, 0.01, 1.0)

        return FusionOutput(
            fused_score=clamp(fused_score),
            fused_uncertainty=fused_uncertainty,
            per_sensor_weights=normalized,
        )

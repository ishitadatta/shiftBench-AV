from __future__ import annotations

import copy
import random
from collections import deque
from typing import Dict, Iterable, List

from fusionbench.core.types import Sample
from fusionbench.core.utils import clamp


class PerturbationEngine:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def apply(self, samples: List[Sample], operations: Iterable[Dict]) -> List[Sample]:
        transformed = copy.deepcopy(samples)
        for operation in operations:
            transformed = self._apply_operation(transformed, operation)
        return transformed

    def _apply_operation(self, samples: List[Sample], op: Dict) -> List[Sample]:
        op_type = str(op.get("type", "")).strip().lower()
        target = op.get("target", "all")

        if op_type == "gaussian_noise":
            std = float(op.get("std", 0.05))
            return self._gaussian_noise(samples, target, std)

        if op_type == "dropout":
            probability = float(op.get("probability", 0.1))
            return self._dropout(samples, target, probability)

        if op_type == "bias_drift":
            offset = float(op.get("offset", 0.1))
            return self._bias_drift(samples, target, offset)

        if op_type == "uncertainty_inflation":
            factor = float(op.get("factor", 1.5))
            return self._uncertainty_inflation(samples, target, factor)

        if op_type == "temporal_jitter":
            window = int(op.get("window", 3))
            return self._temporal_jitter(samples, target, window)

        raise ValueError(f"Unknown perturbation type: {op_type}")

    def _target_sensors(self, sample: Sample, target: str) -> List[str]:
        if target == "all":
            return list(sample.sensors.keys())
        if target not in sample.sensors:
            return []
        return [target]

    def _gaussian_noise(self, samples: List[Sample], target: str, std: float) -> List[Sample]:
        for sample in samples:
            for sensor_name in self._target_sensors(sample, target):
                reading = sample.sensors[sensor_name]
                reading.score = clamp(reading.score + self._rng.gauss(0.0, std))
                reading.uncertainty = clamp(reading.uncertainty + abs(self._rng.gauss(0.0, std * 0.5)), 0.01, 1.0)
        return samples

    def _dropout(self, samples: List[Sample], target: str, probability: float) -> List[Sample]:
        for sample in samples:
            for sensor_name in self._target_sensors(sample, target):
                if self._rng.random() < probability:
                    reading = sample.sensors[sensor_name]
                    reading.score = 0.5
                    reading.uncertainty = min(1.0, reading.uncertainty + 0.45)
        return samples

    def _bias_drift(self, samples: List[Sample], target: str, offset: float) -> List[Sample]:
        for sample in samples:
            for sensor_name in self._target_sensors(sample, target):
                reading = sample.sensors[sensor_name]
                reading.score = clamp(reading.score + offset)
                reading.uncertainty = clamp(reading.uncertainty + abs(offset) * 0.25, 0.01, 1.0)
        return samples

    def _uncertainty_inflation(self, samples: List[Sample], target: str, factor: float) -> List[Sample]:
        for sample in samples:
            for sensor_name in self._target_sensors(sample, target):
                reading = sample.sensors[sensor_name]
                reading.uncertainty = clamp(reading.uncertainty * factor, 0.01, 1.0)
        return samples

    def _temporal_jitter(self, samples: List[Sample], target: str, window: int) -> List[Sample]:
        if window < 2:
            return samples

        history: Dict[str, deque] = {}
        for sample in samples:
            for sensor_name in self._target_sensors(sample, target):
                history.setdefault(sensor_name, deque(maxlen=window))
                queue = history[sensor_name]
                queue.append(sample.sensors[sensor_name].score)

                if len(queue) > 1 and self._rng.random() < 0.4:
                    delayed_value = queue[0]
                    reading = sample.sensors[sensor_name]
                    reading.score = clamp((reading.score + delayed_value) * 0.5)
                    reading.uncertainty = clamp(reading.uncertainty + 0.1, 0.01, 1.0)
        return samples


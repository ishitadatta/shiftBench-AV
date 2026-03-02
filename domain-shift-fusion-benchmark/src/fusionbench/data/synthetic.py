from __future__ import annotations

import random
from typing import List

from fusionbench.core.types import Sample, SensorReading
from fusionbench.core.utils import clamp


SENSOR_NAMES = ("camera", "lidar", "radar")


def _make_sensor_score(label: int, rng: random.Random, sensor_noise: float) -> float:
    base = 0.8 if label == 1 else 0.2
    return clamp(base + rng.gauss(0.0, sensor_noise))


def _make_uncertainty(score: float, label: int, rng: random.Random) -> float:
    mismatch = abs(score - float(label))
    return max(0.01, min(1.0, 0.05 + mismatch * 0.85 + rng.random() * 0.05))


def generate_synthetic_samples(n_samples: int, seed: int = 42) -> List[Sample]:
    rng = random.Random(seed)
    samples: List[Sample] = []

    for idx in range(n_samples):
        label = 1 if rng.random() >= 0.5 else 0
        camera_score = _make_sensor_score(label, rng, sensor_noise=0.12)
        lidar_score = _make_sensor_score(label, rng, sensor_noise=0.09)
        radar_score = _make_sensor_score(label, rng, sensor_noise=0.15)

        sample = Sample(
            sample_id=idx,
            label=label,
            timestamp=idx,
            sensors={
                "camera": SensorReading(camera_score, _make_uncertainty(camera_score, label, rng)),
                "lidar": SensorReading(lidar_score, _make_uncertainty(lidar_score, label, rng)),
                "radar": SensorReading(radar_score, _make_uncertainty(radar_score, label, rng)),
            },
        )
        samples.append(sample)

    return samples


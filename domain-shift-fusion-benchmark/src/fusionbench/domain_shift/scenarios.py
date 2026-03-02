from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ShiftScenario:
    name: str
    description: str
    operations: List[Dict]


def built_in_scenarios() -> List[ShiftScenario]:
    return [
        ShiftScenario(
            name="clear_weather_baseline",
            description="No perturbations; control condition.",
            operations=[],
        ),
        ShiftScenario(
            name="rain_camera_noise",
            description="Camera performance degrades due to rain blur and reflections.",
            operations=[
                {"type": "gaussian_noise", "target": "camera", "std": 0.08},
                {"type": "uncertainty_inflation", "target": "camera", "factor": 1.4},
            ],
        ),
        ShiftScenario(
            name="fog_lidar_dropout",
            description="Dense fog causes intermittent LiDAR dropout and higher uncertainty.",
            operations=[
                {"type": "dropout", "target": "lidar", "probability": 0.2},
                {"type": "uncertainty_inflation", "target": "lidar", "factor": 1.6},
            ],
        ),
        ShiftScenario(
            name="calibration_drift",
            description="Multi-sensor calibration drift induces score bias and temporal misalignment.",
            operations=[
                {"type": "bias_drift", "target": "camera", "offset": 0.06},
                {"type": "bias_drift", "target": "lidar", "offset": -0.04},
                {"type": "temporal_jitter", "target": "all", "window": 4},
            ],
        ),
        ShiftScenario(
            name="night_adverse_combo",
            description="Compound adverse scenario with camera noise, LiDAR dropout, and radar drift.",
            operations=[
                {"type": "gaussian_noise", "target": "camera", "std": 0.1},
                {"type": "dropout", "target": "lidar", "probability": 0.15},
                {"type": "bias_drift", "target": "radar", "offset": 0.05},
                {"type": "uncertainty_inflation", "target": "all", "factor": 1.25},
            ],
        ),
    ]


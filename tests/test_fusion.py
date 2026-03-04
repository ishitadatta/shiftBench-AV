from fusionbench.core.types import Sample, SensorReading
from fusionbench.fusion.uncertainty import (
    AdaptiveReliabilityFusion,
    CounterfactualConsensusFusion,
    InverseVarianceFusion,
)


def test_inverse_variance_fusion_weights_lower_uncertainty_higher():
    sample = Sample(
        sample_id=1,
        label=1,
        sensors={
            "camera": SensorReading(score=0.9, uncertainty=0.05),
            "lidar": SensorReading(score=0.2, uncertainty=0.4),
        },
    )

    model = InverseVarianceFusion()
    out = model.fuse(sample)

    assert out.per_sensor_weights["camera"] > out.per_sensor_weights["lidar"]
    assert 0.5 <= out.fused_score <= 1.0
    assert 0.0 < out.fused_uncertainty <= 1.0


def test_adaptive_reliability_downweights_disagreement_and_tempers_confidence():
    sample = Sample(
        sample_id=2,
        label=1,
        sensors={
            "camera": SensorReading(score=0.95, uncertainty=0.08),
            "lidar": SensorReading(score=0.10, uncertainty=0.08),
            "radar": SensorReading(score=0.20, uncertainty=0.12),
        },
    )

    baseline = InverseVarianceFusion().fuse(sample)
    adaptive = AdaptiveReliabilityFusion().fuse(sample)

    assert adaptive.fused_uncertainty > baseline.fused_uncertainty
    assert adaptive.per_sensor_weights["camera"] < baseline.per_sensor_weights["camera"]
    assert 0.0 < sum(adaptive.per_sensor_weights.values()) <= 1.000001
    assert 0.0 < adaptive.fused_uncertainty <= 1.0


def test_counterfactual_consensus_handles_sensor_conflict():
    sample = Sample(
        sample_id=3,
        label=0,
        sensors={
            "camera": SensorReading(score=0.92, uncertainty=0.10),
            "lidar": SensorReading(score=0.15, uncertainty=0.10),
            "radar": SensorReading(score=0.20, uncertainty=0.16),
        },
    )

    model = CounterfactualConsensusFusion()
    out = model.fuse(sample)

    assert 0.0 <= out.fused_score <= 1.0
    assert 0.0 < out.fused_uncertainty <= 1.0
    assert abs(sum(out.per_sensor_weights.values()) - 1.0) < 1e-6
    assert out.per_sensor_weights["camera"] < 0.5

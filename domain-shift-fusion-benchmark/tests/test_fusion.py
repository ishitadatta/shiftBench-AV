from fusionbench.core.types import Sample, SensorReading
from fusionbench.fusion.uncertainty import InverseVarianceFusion


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

from fusionbench.data.synthetic import generate_synthetic_samples
from fusionbench.perturbations.operators import PerturbationEngine


def test_dropout_pushes_sensor_toward_unknown():
    samples = generate_synthetic_samples(50, seed=9)
    engine = PerturbationEngine(seed=9)

    shifted = engine.apply(
        samples,
        operations=[{"type": "dropout", "target": "camera", "probability": 1.0}],
    )

    assert all(abs(s.sensors["camera"].score - 0.5) < 1e-9 for s in shifted)
    assert all(s.sensors["camera"].uncertainty >= 0.45 for s in shifted)

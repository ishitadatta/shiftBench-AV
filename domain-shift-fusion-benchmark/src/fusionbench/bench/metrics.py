from __future__ import annotations

from typing import Iterable, List, Tuple

from fusionbench.core.utils import safe_log_prob


def accuracy(labels: Iterable[int], probs: Iterable[float], threshold: float = 0.5) -> float:
    labels_list = list(labels)
    probs_list = list(probs)
    if not labels_list:
        return 0.0
    correct = 0
    for y, p in zip(labels_list, probs_list):
        pred = 1 if p >= threshold else 0
        if pred == y:
            correct += 1
    return correct / len(labels_list)


def negative_log_likelihood(labels: Iterable[int], probs: Iterable[float]) -> float:
    labels_list = list(labels)
    probs_list = list(probs)
    if not labels_list:
        return 0.0
    total = 0.0
    for y, p in zip(labels_list, probs_list):
        total -= safe_log_prob(p if y == 1 else (1.0 - p))
    return total / len(labels_list)


def expected_calibration_error(labels: Iterable[int], probs: Iterable[float], bins: int = 10) -> float:
    labels_list = list(labels)
    probs_list = list(probs)
    if not labels_list:
        return 0.0

    bucket_totals = [0] * bins
    bucket_conf = [0.0] * bins
    bucket_acc = [0.0] * bins

    for y, p in zip(labels_list, probs_list):
        idx = min(bins - 1, int(p * bins))
        bucket_totals[idx] += 1
        bucket_conf[idx] += p
        bucket_acc[idx] += 1.0 if ((p >= 0.5) == (y == 1)) else 0.0

    n = len(labels_list)
    ece = 0.0
    for idx in range(bins):
        if bucket_totals[idx] == 0:
            continue
        avg_conf = bucket_conf[idx] / bucket_totals[idx]
        avg_acc = bucket_acc[idx] / bucket_totals[idx]
        ece += (bucket_totals[idx] / n) * abs(avg_acc - avg_conf)

    return ece


def pearson_correlation(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    return cov / ((var_x * var_y) ** 0.5)


def uncertainty_risk_correlation(labels: Iterable[int], probs: Iterable[float], uncertainties: Iterable[float]) -> float:
    labels_list = list(labels)
    probs_list = list(probs)
    uncertainties_list = list(uncertainties)

    errors: List[float] = []
    for y, p in zip(labels_list, probs_list):
        pred = 1 if p >= 0.5 else 0
        errors.append(1.0 if pred != y else 0.0)

    return pearson_correlation(uncertainties_list, errors)


def aggregate_metrics(labels: List[int], probs: List[float], uncertainties: List[float], calibration_bins: int = 10) -> Tuple[float, float, float, float]:
    return (
        accuracy(labels, probs),
        negative_log_likelihood(labels, probs),
        expected_calibration_error(labels, probs, bins=calibration_bins),
        uncertainty_risk_correlation(labels, probs, uncertainties),
    )


from __future__ import annotations

import json
from pathlib import Path


def load_results(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"]: row for row in payload["scenarios"]}


def rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" />'


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "middle") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" font-family="Helvetica,Arial,sans-serif">{value}</text>'


def write_svg(path: Path, content: str, width: int = 980, height: int = 520) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{content}</svg>'
    path.write_text(svg, encoding="utf-8")


def draw_metric_chart(
    out_path: Path,
    title: str,
    metric_key: str,
    y_min: float,
    y_max: float,
    dataset: dict[str, dict[str, dict[str, float]]],
) -> None:
    width, height = 980, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 70, 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    scenarios = ["baseline", "weather_and_occlusion_shift"]
    methods = list(dataset.keys())
    colors = {"inverse_variance": "#5B8FF9", "adaptive_reliability": "#5AD8A6", "counterfactual_consensus": "#F6BD16"}

    def y_to_px(v: float) -> float:
        if y_max <= y_min:
            return margin_top + plot_h
        ratio = (v - y_min) / (y_max - y_min)
        return margin_top + plot_h - ratio * plot_h

    parts = [rect(0, 0, width, height, "#ffffff"), text(width / 2, 35, title, 20)]
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333" />')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333" />')

    for i in range(6):
        tick_v = y_min + (y_max - y_min) * (i / 5.0)
        y = y_to_px(tick_v)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#eee" />')
        parts.append(text(margin_left - 10, y + 4, f"{tick_v:.3f}", 11, anchor="end"))

    group_w = plot_w / len(scenarios)
    bar_w = group_w / (len(methods) + 1)
    for i, scenario in enumerate(scenarios):
        gx = margin_left + i * group_w
        parts.append(text(gx + group_w / 2, margin_top + plot_h + 25, scenario, 12))
        for j, method in enumerate(methods):
            value = dataset[method][scenario][metric_key]
            x = gx + (j + 0.5) * bar_w
            y = y_to_px(value)
            h = margin_top + plot_h - y
            parts.append(rect(x, y, bar_w * 0.75, h, colors[method]))
            parts.append(text(x + bar_w * 0.375, y - 6, f"{value:.3f}", 10))

    legend_x, legend_y = margin_left + 20, 50
    for idx, method in enumerate(methods):
        lx = legend_x + idx * 240
        parts.append(rect(lx, legend_y, 16, 16, colors[method]))
        parts.append(text(lx + 24, legend_y + 13, method, 12, anchor="start"))

    write_svg(out_path, "".join(parts), width=width, height=height)


def main() -> None:
    base_path = Path("examples/nuscenes_mini_results_baseline_again.json")
    adaptive_path = Path("examples/nuscenes_mini_results_adaptive.json")
    novel_path = Path("examples/nuscenes_mini_results_novel.json")

    dataset = {
        "inverse_variance": load_results(base_path),
        "adaptive_reliability": load_results(adaptive_path),
        "counterfactual_consensus": load_results(novel_path),
    }

    draw_metric_chart(
        out_path=Path("examples/figures/nll_comparison.svg"),
        title="NLL Comparison (Lower Is Better)",
        metric_key="negative_log_likelihood",
        y_min=0.6,
        y_max=1.8,
        dataset=dataset,
    )
    draw_metric_chart(
        out_path=Path("examples/figures/uncertainty_correlation_comparison.svg"),
        title="Uncertainty-Risk Correlation (Higher Is Better)",
        metric_key="uncertainty_risk_correlation",
        y_min=-0.1,
        y_max=0.4,
        dataset=dataset,
    )

    print("Wrote graphs to examples/figures/")


if __name__ == "__main__":
    main()

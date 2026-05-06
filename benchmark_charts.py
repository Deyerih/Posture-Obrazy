from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    details_path: Path
    summary_path: Path | None
    metadata: dict[str, object]
    summary_rows: list[dict[str, object]]
    records: list[dict[str, object]]


def create_placeholder_figure(title: str, message: str) -> Figure:
    figure = Figure(figsize=(11, 8), tight_layout=True)
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, fontweight="bold")
    axis.text(0.5, 0.45, message, ha="center", va="center", fontsize=11, wrap=True)
    return figure


def discover_benchmark_runs(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob("*_details.json"), reverse=True)


def load_benchmark_run(details_path: Path) -> BenchmarkRun:
    payload = json.loads(details_path.read_text(encoding="utf-8"))
    summary_path = details_path.with_name(
        details_path.name.replace("_details.json", "_summary.csv")
    )
    return BenchmarkRun(
        run_id=str(payload.get("run_id", details_path.stem.replace("_details", ""))),
        details_path=details_path,
        summary_path=summary_path if summary_path.exists() else None,
        metadata=dict(payload.get("metadata", {})),
        summary_rows=list(payload.get("summary", [])),
        records=list(payload.get("records", [])),
    )


def save_benchmark_figure(details_path: Path, output_path: Path | None = None) -> Path:
    run = load_benchmark_run(details_path)
    figure = create_benchmark_figure(run)
    resolved_output = output_path or details_path.with_suffix(".png")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output, dpi=160, bbox_inches="tight")
    return resolved_output


def create_benchmark_figure(run: BenchmarkRun) -> Figure:
    rows = sorted(
        run.summary_rows,
        key=lambda row: (str(row.get("engine", "")), str(row.get("model_name", ""))),
    )
    if not rows:
        return create_placeholder_figure(
            title=f"Benchmark Run {run.run_id}",
            message="This benchmark log does not contain summary rows yet.",
        )

    labels = [_row_label(row) for row in rows]
    avg_latency = [float(row.get("avg_latency_ms", 0.0)) for row in rows]
    median_latency = [float(row.get("median_latency_ms", 0.0)) for row in rows]
    detection_rate = [float(row.get("detection_rate", 0.0)) * 100.0 for row in rows]
    valid_rate = [float(row.get("valid_posture_rate", 0.0)) * 100.0 for row in rows]
    avg_posture = [_average_posture_for_row(run.records, row) for row in rows]

    figure = Figure(figsize=(11, 8), tight_layout=True)
    axes = figure.subplots(2, 2)
    figure.suptitle(f"Benchmark Run {run.run_id}", fontsize=14, fontweight="bold")

    _plot_bars(
        axes[0][0],
        labels,
        avg_latency,
        title="Average Latency (ms)",
        color="#1f77b4",
    )
    _plot_bars(
        axes[0][1],
        labels,
        detection_rate,
        title="Detection Rate (%)",
        color="#2ca02c",
        ylim=(0, 110),
    )
    _plot_bars(
        axes[1][0],
        labels,
        valid_rate,
        title="Valid Posture Rate (%)",
        color="#ff7f0e",
        ylim=(0, 110),
    )
    _plot_grouped_bars(
        axes[1][1],
        labels,
        avg_posture,
        median_latency,
        title="Avg Posture vs Median Latency",
        left_label="Avg Posture",
        right_label="Median Latency (ms)",
        left_color="#9467bd",
        right_color="#8c564b",
    )

    metadata_text = (
        f"Input: {run.metadata.get('input', '-')}\n"
        f"Engines: {run.metadata.get('engines', '-')}\n"
        f"Images: {run.metadata.get('images_count', '-')}\n"
        f"Records: {run.metadata.get('records_count', '-')}"
    )
    figure.text(0.01, 0.01, metadata_text, fontsize=9, alpha=0.8)
    return figure


def _row_label(row: dict[str, object]) -> str:
    engine = str(row.get("engine", ""))
    model_name = str(row.get("model_name", ""))
    return f"{engine}\n{model_name}"


def _average_posture_for_row(
    records: list[dict[str, object]], summary_row: dict[str, object]
) -> float:
    posture_values = [
        int(record["posture_level"])
        for record in records
        if record.get("engine") == summary_row.get("engine")
        and record.get("model_name") == summary_row.get("model_name")
        and record.get("posture_level") is not None
    ]
    if not posture_values:
        return 0.0
    return round(sum(posture_values) / len(posture_values), 2)


def _plot_bars(
    axis,
    labels: list[str],
    values: list[float],
    title: str,
    color: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    bars = axis.bar(labels, values, color=color, alpha=0.9)
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=0)
    axis.grid(axis="y", alpha=0.25)
    if ylim is not None:
        axis.set_ylim(*ylim)
    for bar, value in zip(bars, values, strict=False):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_grouped_bars(
    axis,
    labels: list[str],
    left_values: list[float],
    right_values: list[float],
    title: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
) -> None:
    positions = list(range(len(labels)))
    width = 0.36
    left_bars = axis.bar(
        [position - width / 2.0 for position in positions],
        left_values,
        width=width,
        label=left_label,
        color=left_color,
        alpha=0.9,
    )
    right_bars = axis.bar(
        [position + width / 2.0 for position in positions],
        right_values,
        width=width,
        label=right_label,
        color=right_color,
        alpha=0.9,
    )
    axis.set_title(title)
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    for bars in [left_bars, right_bars]:
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

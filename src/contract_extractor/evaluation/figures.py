"""Report figures, rendered in a dark theme.

Each figure is drawn on a rounded dark card over a transparent background. Colors come from
a validated categorical palette (one hue per system family) and a single-hue blue ramp for
magnitudes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from contract_extractor.schema import KEY_TO_FIELD  # noqa: E402

DPI = 200
_PREFERRED_FONTS = ("Ubuntu Sans", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans")
_INSTALLED_FONTS = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
FONT_FAMILY = [f for f in _PREFERRED_FONTS if f in _INSTALLED_FONTS] or ["sans-serif"]

DISPLAY_NAMES = {
    "llama3.2-3b-cuad": "Llama-3.2-3B + QLoRA",
    "qwen2.5-1.5b-cuad": "Qwen2.5-1.5B + QLoRA",
    "legal-bert-ner": "Legal-BERT NER",
    "qwen2.5-1.5b-zero-shot": "Qwen2.5-1.5B zero-shot",
    "llama3.2-3b-zero-shot": "Llama-3.2-3B zero-shot",
}
FIELD_LABEL_FIXES = {
    "Ip ": "IP ",
    "Rofr/Rofo/Rofn": "ROFR/ROFO/ROFN",
    "Unlimited/All-You-Can-Eat-License": "Unlimited License",
}
FAMILY_LABELS = {"fine-tuned": "Fine-tuned (QLoRA)", "zero-shot": "Zero-shot", "baseline": "Token classification"}


@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    baseline: str
    border: tuple[float, float, float, float]
    family: dict[str, str]
    pair: tuple[str, str]
    ramp: tuple[str, ...]


THEME = Theme(
    surface="#121735",
    ink="#f8fafc",
    ink_secondary="#b8bdd9",
    ink_muted="#8a90b4",
    grid="#232a52",
    baseline="#2f3666",
    border=(0.165, 0.192, 0.376, 1.0),
    family={"fine-tuned": "#3987e5", "zero-shot": "#d95926", "baseline": "#199e70"},
    pair=("#9ec5f4", "#3987e5"),
    ramp=("#1b2148", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)


def display_name(system: str) -> str:
    return DISPLAY_NAMES.get(system, system)


def field_label(key: str) -> str:
    label = KEY_TO_FIELD[key].cuad_name
    for raw, fixed in FIELD_LABEL_FIXES.items():
        label = label.replace(raw, fixed)
    return label


# Layout


@dataclass
class Canvas:
    fig: plt.Figure
    ax: plt.Axes
    theme: Theme
    width_in: float
    height_in: float


def _canvas(
    theme: Theme,
    width_in: float,
    plot_height_in: float,
    title: str,
    subtitle: str,
    left_in: float = 1.0,
    right_in: float = 0.45,
    header_in: float = 1.05,
    footer_in: float = 0.6,
) -> Canvas:
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 9, "axes.unicode_minus": False})
    height_in = header_in + plot_height_in + footer_in
    fig = plt.figure(figsize=(width_in, height_in), dpi=DPI)
    fig.patch.set_alpha(0.0)
    card = FancyBboxPatch(
        (0.004, 0.004),
        0.992,
        0.992,
        boxstyle="round,pad=0,rounding_size=0.02",
        transform=fig.transFigure,
        facecolor=theme.surface,
        edgecolor=theme.border,
        linewidth=1.0,
        mutation_aspect=width_in / height_in,
        zorder=-10,
    )
    fig.patches.append(card)
    pad = 0.32 / width_in
    fig.text(pad, 1 - 0.30 / height_in, title, color=theme.ink, fontsize=13.5, va="top")
    fig.text(pad, 1 - 0.60 / height_in, subtitle, color=theme.ink_secondary, fontsize=9, va="top", linespacing=1.4)
    ax = fig.add_axes(
        (
            left_in / width_in,
            footer_in / height_in,
            1 - (left_in + right_in) / width_in,
            plot_height_in / height_in,
        )
    )
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.baseline)
    ax.tick_params(colors=theme.ink_muted, labelsize=8.5, length=0, pad=6)
    ax.xaxis.label.set_color(theme.ink_secondary)
    ax.yaxis.label.set_color(theme.ink_secondary)
    return Canvas(fig, ax, theme, width_in, height_in)


def _legend(canvas: Canvas, items: list[tuple[str, str]], marker: str = "s") -> None:
    handles = [
        Line2D([], [], marker=marker, linestyle="none", markersize=8, markerfacecolor=color, markeredgewidth=0)
        for _, color in items
    ]
    legend = canvas.fig.legend(
        handles,
        [label for label, _ in items],
        loc="upper left",
        bbox_to_anchor=(0.32 / canvas.width_in - 0.006, 1 - 0.80 / canvas.height_in),
        ncol=len(items),
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        columnspacing=1.6,
        borderaxespad=0,
    )
    for text in legend.get_texts():
        text.set_color(canvas.theme.ink_secondary)


def _x_grid(ax: plt.Axes, theme: Theme) -> None:
    ax.grid(axis="x", color=theme.grid, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)


def _rounded_hbar(canvas: Canvas, y: float, value: float, height: float, color: str, radius_px: float = 4) -> None:
    """Horizontal bar with a rounded data end and a square end on the baseline."""
    if value <= 0:
        return
    ax = canvas.ax
    bbox = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px_per_x = bbox.width * canvas.width_in * DPI / (x1 - x0)
    px_per_y = bbox.height * canvas.height_in * DPI / abs(y1 - y0)
    radius_x = min(radius_px * DPI / 100 / px_per_x, value / 2)
    ax.add_patch(
        FancyBboxPatch(
            (0, y - height / 2),
            value,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius_x}",
            mutation_aspect=px_per_x / px_per_y,
            facecolor=color,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.add_patch(Rectangle((0, y - height / 2), min(value, 2 * radius_x), height, facecolor=color, zorder=2))


def _save(canvas: Canvas, path: Path) -> None:
    canvas.fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(canvas.fig)


def render(draw: Callable[[Path], None], directory: Path, stem: str) -> Path:
    path = directory / f"{stem}.png"
    draw(path)
    return path


# Figures


def ranked_bars(
    frame: pd.DataFrame,
    column: str,
    title: str,
    subtitle: str,
    value_format: Callable[[float], str],
    scale: float = 1.0,
) -> Callable[[Path], None]:
    data = frame.dropna(subset=[column]).sort_values(column, ascending=True).reset_index(drop=True)
    values = data[column].to_numpy() * scale

    def draw(path: Path) -> None:
        theme = THEME
        canvas = _canvas(theme, 8.4, 0.48 * len(data) + 0.2, title, subtitle, left_in=2.0)
        ax = canvas.ax
        ax.set_ylim(-0.6, len(data) - 0.4)
        ax.set_xlim(0, values.max() * 1.16)
        _x_grid(ax, theme)
        for y, (value, fam) in enumerate(zip(values, data["family"], strict=True)):
            _rounded_hbar(canvas, y, value, 0.46, theme.family[fam])
            ax.text(value, y, f"  {value_format(value)}", va="center", fontsize=9, color=theme.ink)
        ax.set_yticks(range(len(data)), [display_name(s) for s in data["system"]])
        ax.tick_params(axis="y", colors=theme.ink_secondary, labelsize=9)
        present = [f for f in theme.family if f in set(data["family"])]
        _legend(canvas, [(FAMILY_LABELS[f], theme.family[f]) for f in present])
        _save(canvas, path)

    return draw


def finetuning_gain(frame: pd.DataFrame) -> Callable[[Path], None] | None:
    pairs = []
    for base in ("llama3.2-3b", "qwen2.5-1.5b"):
        zero = frame[frame["system"] == f"{base}-zero-shot"]
        tuned = frame[frame["system"] == f"{base}-cuad"]
        if not zero.empty and not tuned.empty:
            pairs.append((base, 100 * zero["span_micro_f1"].iloc[0], 100 * tuned["span_micro_f1"].iloc[0]))
    bert = frame[frame["family"] == "baseline"]
    if not pairs:
        return None

    def draw(path: Path) -> None:
        theme = THEME
        canvas = _canvas(
            theme,
            8.4,
            3.6,
            "Fine-tuning turns small models into usable extractors",
            "Span F1 (micro) on the CUAD test contracts, same model and prompt before and after QLoRA.",
            left_in=0.75,
            right_in=2.3,
        )
        ax = canvas.ax
        ax.set_xlim(-0.12, 1.08)
        ax.set_ylim(0, 70)
        ax.grid(axis="y", color=theme.grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([0, 1], ["Zero-shot", "Fine-tuned (QLoRA)"])
        ax.tick_params(axis="x", colors=theme.ink_secondary, labelsize=9.5, pad=10)
        ax.set_yticks(range(0, 61, 20))
        label_x = 1.06
        if not bert.empty:
            level = 100 * bert["span_micro_f1"].iloc[0]
            ax.plot([-0.12, 1.0], [level, level], color=theme.family["baseline"], linewidth=1.2, zorder=1)
            ax.text(
                label_x,
                level,
                f"Legal-BERT NER  {level:.1f}",
                va="center",
                fontsize=8.5,
                color=theme.ink_secondary,
                clip_on=False,
            )
        colors = dict(zip(("llama3.2-3b", "qwen2.5-1.5b"), (theme.pair[1], theme.pair[0]), strict=True))
        for base, zero, tuned in pairs:
            color = colors[base]
            ax.plot([0, 1], [zero, tuned], color=color, linewidth=2.2, solid_capstyle="round", zorder=3)
            ax.scatter([0, 1], [zero, tuned], s=70, color=color, edgecolor=theme.surface, linewidth=2, zorder=4)
            name = display_name(f"{base}-cuad").removesuffix(" + QLoRA")
            ax.text(
                label_x,
                tuned,
                f"{name}  {tuned:.1f}  (+{tuned - zero:.1f})",
                va="center",
                fontsize=9,
                color=theme.ink,
                clip_on=False,
            )
            ax.text(-0.05, zero, f"{zero:.1f}", va="center", ha="right", fontsize=8.5, color=theme.ink_secondary)
        _legend(canvas, [(display_name(f"{b}-cuad").removesuffix(" + QLoRA"), colors[b]) for b, _, _ in pairs], "o")
        _save(canvas, path)

    return draw


def cost_quality(frame: pd.DataFrame) -> Callable[[Path], None]:
    data = frame.dropna(subset=["span_micro_f1", "cost_per_doc_usd"]).reset_index(drop=True)
    offsets = {
        "llama3.2-3b-cuad": (-10, 9, "right"),
        "qwen2.5-1.5b-cuad": (-10, -14, "right"),
        "legal-bert-ner": (10, 0, "left"),
        "qwen2.5-1.5b-zero-shot": (-10, 9, "right"),
        "llama3.2-3b-zero-shot": (-10, -14, "right"),
    }

    def draw(path: Path) -> None:
        theme = THEME
        canvas = _canvas(
            theme,
            8.4,
            4.0,
            "Quality versus cost per contract",
            "Cost = latency on one RTX 5080 × USD 0.40 per GPU hour. Upper left is better.",
            left_in=0.85,
        )
        ax = canvas.ax
        ax.set_xscale("log")
        ax.set_xlim(4e-6, 1.2e-2)
        ax.set_ylim(0, 72)
        ax.grid(which="major", color=theme.grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Cost per contract (USD, log scale)", labelpad=8)
        ax.set_ylabel("Span F1, micro", labelpad=8)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: "$" + f"{v:.5f}".rstrip("0").rstrip("."))
        )
        for _, row in data.iterrows():
            x, y = row["cost_per_doc_usd"], 100 * row["span_micro_f1"]
            ax.scatter(x, y, s=95, color=theme.family[row["family"]], edgecolor=theme.surface, linewidth=2, zorder=4)
            dx, dy, ha = offsets.get(row["system"], (10, 0, "left"))
            ax.annotate(
                f"{display_name(row['system'])}  {y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va="center",
                fontsize=8.5,
                color=theme.ink,
            )
        present = [f for f in theme.family if f in set(data["family"])]
        _legend(canvas, [(FAMILY_LABELS[f], theme.family[f]) for f in present], "o")
        _save(canvas, path)

    return draw


def _cell_text_color(rgb: tuple[float, float, float]) -> str:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#0b0b0b" if luminance > 0.45 else "#ffffff"


def per_field_heatmap(per_field: pd.DataFrame) -> Callable[[Path], None]:
    supported = per_field[per_field["support"] > 0].sort_values("support", ascending=False)
    scores = 100 * supported.drop(columns=["support"])
    columns = list(scores.columns)

    def draw(path: Path) -> None:
        theme = THEME
        cell_h = 0.27
        canvas = _canvas(
            theme,
            8.4,
            cell_h * len(scores),
            "Span F1 per clause type",
            "Rows sorted by number of gold spans in the test set (in brackets). Brighter cells are higher scores.",
            left_in=3.0,
            right_in=0.3,
            header_in=1.45,
            footer_in=0.3,
        )
        ax = canvas.ax
        cmap = LinearSegmentedColormap.from_list("ramp", theme.ramp)
        grid = scores.to_numpy()
        ax.pcolormesh(grid, cmap=cmap, vmin=0, vmax=100, edgecolors=theme.surface, linewidth=2.0)
        ax.set_xlim(0, len(columns))
        ax.set_ylim(len(scores), 0)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.tick_top()
        labels = [display_name(c).replace(" + ", "\n+ ").replace(" zero-shot", "\nzero-shot") for c in columns]
        ax.set_xticks(np.arange(len(columns)) + 0.5, labels)
        ax.tick_params(axis="x", colors=theme.ink_secondary, labelsize=8.5, pad=6)
        ax.set_yticks(
            np.arange(len(scores)) + 0.5,
            [f"{field_label(k)}  ({int(n)})" for k, n in zip(scores.index, supported["support"], strict=True)],
        )
        ax.tick_params(axis="y", colors=theme.ink_secondary, labelsize=8.2)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                rgb = to_rgb(cmap(grid[i, j] / 100))
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{grid[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=_cell_text_color(rgb),
                )
        _save(canvas, path)

    return draw


def validation_loss(summaries: dict[str, dict[str, Any]]) -> Callable[[Path], None]:
    names = sorted(summaries, key=lambda n: 0 if n.startswith("llama") else 1)
    eval_samples = summaries[names[0]].get("eval_samples", 0)
    last_step = max(h["step"] for s in summaries.values() for h in s["log_history"] if "eval_loss" in h)

    def draw(path: Path) -> None:
        theme = THEME
        canvas = _canvas(
            theme,
            8.4,
            3.4,
            "Validation loss during QLoRA fine-tuning",
            f"Loss on {eval_samples} held-out windows, JSON tokens only. "
            "The ring marks the checkpoint kept for evaluation.",
            left_in=0.85,
            right_in=2.2,
        )
        ax = canvas.ax
        ax.grid(color=theme.grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Optimizer step", labelpad=8)
        colors = {name: (theme.pair[1] if name.startswith("llama") else theme.pair[0]) for name in names}
        for name in names:
            history = [h for h in summaries[name]["log_history"] if "eval_loss" in h]
            points: dict[int, float] = {}
            for h in history:
                if points and h["step"] - max(points) < 10:
                    continue
                points[h["step"]] = h["eval_loss"]
            steps, losses = list(points), list(points.values())
            color = colors[name]
            ax.plot(steps, losses, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            ax.scatter(steps, losses, s=36, color=color, edgecolor=theme.surface, linewidth=1.5, zorder=4)
            best = int(np.argmin(losses))
            ax.scatter(
                steps[best], losses[best], s=230, facecolors="none", edgecolors=theme.ink, linewidths=1.2, zorder=5
            )
            ax.text(
                1.0,
                losses[-1],
                f"  {display_name(name).removesuffix(' + QLoRA')}  {min(losses):.3f} best",
                va="center",
                fontsize=8.5,
                color=theme.ink,
                transform=ax.get_yaxis_transform(),
                clip_on=False,
            )
        ax.set_xlim(0, last_step * 1.04)
        _legend(canvas, [(display_name(n).removesuffix(" + QLoRA"), colors[n]) for n in names], "o")
        _save(canvas, path)

    return draw

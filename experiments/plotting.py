"""Figures.

Palette and mark choices here are not taste. The four series use a fixed
categorical order validated for colour-vision deficiency (worst adjacent pair
CVD dE 9.1 light / 8.4 dark, both above the 8 target). Two of the light-mode
hues sit below 3:1 contrast against the surface, which obliges *relief*: every
series is direct-labelled at its line end, and every figure is backed by the
result CSV in ``results/``. Colour is never the only carrier of identity.

Two honesty conventions, because the interesting cells of these figures are the
empty ones:

* A method that returns **no feasible design** at some tau has no y-value there.
  A line that simply stops could be misread as missing data, so those points get
  an explicit cross at the axis floor and a footnote.
* A method that returns feasible designs in **some but not all** seeds gets a
  hollow marker. A filled marker means every replicate produced a number.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# -- Design tokens ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e8e7e3"

#: Colour follows the method, never its rank in the current figure. Fixed here
#: once so a figure that drops a method does not repaint the survivors.
SERIES_COLORS: dict[str, str] = {
    "lagrangian": "#2a78d6",  # slot 1, blue
    "scalarization_sweep": "#eb6834",  # slot 2, orange
    "fixed_penalty": "#1baf7a",  # slot 3, aqua
    "optimize_then_reject": "#eda100",  # slot 4, yellow
    "scalarization_w0.5": "#e87ba4",  # slot 5, magenta (tables and appendices)
}

DISPLAY_NAMES: dict[str, str] = {
    "lagrangian": "Lagrangian",
    "scalarization_sweep": "Scalarization",
    "fixed_penalty": "Fixed penalty",
    "optimize_then_reject": "Optimize-then-reject",
    "scalarization_w0.5": "Scalarization (w=0.5)",
}

#: The four constraint-handling rules, in legend order.
HEADLINE_METHODS: tuple[str, ...] = (
    "lagrangian",
    "scalarization_sweep",
    "fixed_penalty",
    "optimize_then_reject",
)

LINE_WIDTH = 2.0
MARKER_SIZE = 6.0


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def _style_axes(ax: plt.Axes, y_grid_only: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(axis="y" if y_grid_only else "both")
    if y_grid_only:
        ax.xaxis.grid(False)


def _tau_axis(ax: plt.Axes, taus, right_pad: float = 0.44, left_pad: float = 0.05) -> None:
    """Tau decreases left to right, so the constraint tightens rightward.

    Padding is asymmetric: the room is needed on the right for direct labels,
    and symmetric margins would leave dead space on the left.
    """
    taus = np.asarray(taus, dtype=float)
    low, high = float(np.min(taus)), float(np.max(taus))
    span = (high - low) or 1.0
    ax.set_xlim(low - right_pad * span, high + left_pad * span)
    ax.invert_xaxis()


def _series_frame(agg: pd.DataFrame, method: str) -> pd.DataFrame:
    return agg[agg["method"] == method].sort_values("tau", ascending=False)


def _plot_series(
    ax: plt.Axes,
    agg: pd.DataFrame,
    method: str,
    show_band: bool = True,
) -> tuple[float, float] | None:
    """Draw one method's mean curve with a +/-1 sd band. Returns the label anchor."""
    frame = _series_frame(agg, method)
    if frame.empty:
        return None
    color = SERIES_COLORS[method]
    tau = frame["tau"].to_numpy(dtype=float)
    mean = frame["mean"].to_numpy(dtype=float)
    std = np.nan_to_num(frame["std"].to_numpy(dtype=float), nan=0.0)
    n_valid = frame["n_valid"].to_numpy(dtype=int)
    n_seeds = frame["n_seeds"].to_numpy(dtype=int)

    if show_band:
        ax.fill_between(tau, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
    ax.plot(tau, mean, color=color, linewidth=LINE_WIDTH, zorder=3, solid_capstyle="round")

    # Filled marker = every seed produced a number. Hollow = only some did.
    complete = n_valid == n_seeds
    partial = (n_valid > 0) & ~complete
    ax.plot(
        tau[complete], mean[complete], linestyle="none", marker="o", markersize=MARKER_SIZE,
        color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
    )
    ax.plot(
        tau[partial], mean[partial], linestyle="none", marker="o", markersize=MARKER_SIZE,
        markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.8, zorder=4,
    )

    finite = np.isfinite(mean)
    if not finite.any():
        return None
    last = int(np.flatnonzero(finite)[-1])
    return float(tau[last]), float(mean[last])


def _mark_total_failures(ax: plt.Axes, agg: pd.DataFrame, methods) -> bool:
    """Cross at the axis floor wherever a method returned nothing feasible."""
    bottom = ax.get_ylim()[0]
    any_marked = False
    for method in methods:
        frame = _series_frame(agg, method)
        dead = frame[frame["n_valid"] == 0]
        if dead.empty:
            continue
        any_marked = True
        ax.plot(
            dead["tau"].to_numpy(dtype=float),
            np.full(len(dead), bottom),
            linestyle="none", marker="X", markersize=MARKER_SIZE + 1,
            color=SERIES_COLORS[method], markeredgecolor=SURFACE, markeredgewidth=1.2,
            clip_on=False, zorder=5,
        )
    return any_marked


def _place_labels(ax: plt.Axes, anchors: list[tuple[tuple[float, float], str, str]]) -> None:
    """Direct-label every line end, pushing labels apart so none overlap.

    Label text wears an ink token; the coloured end marker carries identity. A
    label displaced to avoid a collision gets a thin leader back to its line,
    so the pairing never has to be guessed.
    """
    if not anchors:
        return
    low, high = ax.get_ylim()
    min_gap = 0.075 * (high - low)

    placed: list[tuple[float, float, float, str, str]] = []
    for (x, y), text, color in sorted(anchors, key=lambda a: a[0][1], reverse=True):
        label_y = y if not placed else min(y, placed[-1][2] - min_gap)
        placed.append((x, y, label_y, text, color))

    for x, y, label_y, text, color in placed:
        displaced = abs(label_y - y) > 1e-9
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK_PRIMARY,
            zorder=6,
            annotation_clip=False,
        ) if not displaced else ax.annotate(
            text,
            xy=(x, y),
            xycoords="data",
            xytext=(10, label_y),
            textcoords=("offset points", "data"),
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK_PRIMARY,
            zorder=6,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=color, linewidth=0.9,
                            shrinkA=2, shrinkB=3, alpha=0.7),
        )


def _legend(fig: plt.Figure, methods) -> None:
    handles = [
        Line2D([], [], color=SERIES_COLORS[m], linewidth=LINE_WIDTH, marker="o",
               markersize=MARKER_SIZE - 1, markeredgecolor=SURFACE, markeredgewidth=1.2,
               label=DISPLAY_NAMES[m])
        for m in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        frameon=False,
        labelcolor=INK_PRIMARY,
        handlelength=1.8,
        columnspacing=1.6,
    )


# --------------------------------------------------------------------------
# Headline figure
# --------------------------------------------------------------------------


def headline_tau_sweep(
    best_agg: pd.DataFrame,
    rate_agg: pd.DataFrame,
    path: Path,
    methods=HEADLINE_METHODS,
    subtitle: str = "",
) -> Path:
    """The headline: best feasible binding, and feasibility rate, against tau."""
    apply_style()
    fig, (ax_best, ax_rate) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    anchors = []
    for method in methods:
        anchor = _plot_series(ax_best, best_agg, method)
        if anchor is not None:
            anchors.append((anchor, DISPLAY_NAMES[method], SERIES_COLORS[method]))
    _tau_axis(ax_best, best_agg["tau"])
    has_failures = _mark_total_failures(ax_best, best_agg, methods)
    _place_labels(ax_best, anchors)

    ax_best.set_xlabel("constraint tightness  tau  (kcal/mol)")
    ax_best.set_ylabel("best feasible binding  ddG  (kcal/mol)\nlower is better")
    ax_best.set_title("What you can actually use", loc="left", pad=10)
    _style_axes(ax_best)

    rate_anchors = []
    for method in methods:
        anchor = _plot_series(ax_rate, rate_agg, method)
        if anchor is not None:
            rate_anchors.append((anchor, DISPLAY_NAMES[method], SERIES_COLORS[method]))
    _tau_axis(ax_rate, rate_agg["tau"])
    ax_rate.set_ylim(-0.04, 1.04)
    _place_labels(ax_rate, rate_anchors)
    ax_rate.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_rate.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_rate.set_xlabel("constraint tightness  tau  (kcal/mol)")
    ax_rate.set_ylabel("feasibility rate of returned designs")
    ax_rate.set_title("How much of it clears the bar", loc="left", pad=10)
    _style_axes(ax_rate)

    _legend(fig, methods)

    notes = ["tau tightens to the right", "band = +/-1 sd over seeds",
             "hollow marker = feasible in only some seeds"]
    if has_failures:
        notes.append("X = no feasible design returned")
    fig.text(0.008, 0.055, "   ".join(notes), fontsize=8, color=INK_MUTED, ha="left")
    if subtitle:
        fig.text(0.008, 0.013, subtitle, fontsize=8, color=INK_MUTED, ha="left")

    fig.tight_layout(rect=(0, 0.095, 1, 0.91))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Hypervolume gap: small multiples, one panel per method
# --------------------------------------------------------------------------


def hypervolume_gap(
    all_agg: pd.DataFrame,
    feasible_agg: pd.DataFrame,
    path: Path,
    methods=HEADLINE_METHODS,
    subtitle: str = "",
) -> Path:
    """Hypervolume over all designs vs over feasible designs only.

    Small multiples rather than eight lines on one axes. Within a panel both
    curves share the method's hue and are told apart by line style, so the
    comparison that matters -- the gap -- is the only thing the eye has to do.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(methods), figsize=(12.0, 3.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, method in zip(axes, methods):
        color = SERIES_COLORS[method]
        a = _series_frame(all_agg, method)
        f = _series_frame(feasible_agg, method)
        tau = a["tau"].to_numpy(dtype=float)
        hv_all = a["mean"].to_numpy(dtype=float)
        hv_feasible = np.nan_to_num(f["mean"].to_numpy(dtype=float), nan=0.0)

        ax.fill_between(tau, hv_feasible, hv_all, color=color, alpha=0.16, linewidth=0,
                        label="overclaim")
        ax.plot(tau, hv_all, color=color, linewidth=LINE_WIDTH, linestyle="--",
                dash_capstyle="round", label="all designs")
        ax.plot(tau, hv_feasible, color=color, linewidth=LINE_WIDTH, label="feasible only")
        ax.set_title(DISPLAY_NAMES[method], loc="left", pad=8, fontsize=10)
        ax.set_xlabel("tau")
        _style_axes(ax)
        _tau_axis(ax, tau, right_pad=0.05)

    axes[0].set_ylabel("hypervolume")
    handles = [
        Line2D([], [], color=INK_SECONDARY, linewidth=LINE_WIDTH, linestyle="--",
               label="all returned designs"),
        Line2D([], [], color=INK_SECONDARY, linewidth=LINE_WIDTH, label="feasible designs only"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, labelcolor=INK_PRIMARY, handlelength=2.2)
    note = "the shaded gap is hypervolume bought with designs that break the constraint"
    if subtitle:
        note = f"{note}   |   {subtitle}"
    fig.text(0.008, 0.02, note, fontsize=8, color=INK_MUTED, ha="left")

    fig.tight_layout(rect=(0, 0.075, 1, 0.87))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Cost of choosing a hyperparameter blind
# --------------------------------------------------------------------------


def tuning_cost(frame: pd.DataFrame, path: Path, subtitle: str = "") -> Path:
    """Range plot: how much each rule depends on a setting chosen in advance.

    One row per rule, spanning its worst setting to its best, with the blind
    average marked. A range plot rather than bars because the span *is* the
    finding -- a bar of the best setting would report a number nobody can
    actually obtain without already knowing the answer.

    ``frame`` has columns ``rule``, ``parameter``, ``best``, ``worst``,
    ``blind_mean``, ``best_value``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(9.6, 1.9 + 0.62 * len(frame)))

    frame = frame.sort_values("worst")
    positions = np.arange(len(frame))
    for y, row in zip(positions, frame.itertuples(index=False)):
        color = SERIES_COLORS.get(row.rule, INK_SECONDARY)
        ax.plot([row.worst, row.best], [y, y], color=color, linewidth=5.0,
                solid_capstyle="round", alpha=0.32, zorder=2)
        ax.plot([row.worst, row.best], [y, y], linestyle="none", marker="o",
                markersize=MARKER_SIZE + 1, color=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)
        ax.plot([row.blind_mean], [y], marker="D", markersize=MARKER_SIZE + 1, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
        # The x-axis is inverted, so the worst setting sits at the left end;
        # the label goes further left, clear of the range itself.
        ax.annotate(f"spread {abs(row.best - row.worst):.2f}", xy=(row.worst, y),
                    xytext=(-12, 0), textcoords="offset points", va="center", ha="right",
                    fontsize=8.5, color=INK_PRIMARY, annotation_clip=False)

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{DISPLAY_NAMES.get(r.rule, r.rule)}\n{r.parameter}" for r in frame.itertuples(index=False)]
    )
    ax.set_xlabel("best feasible binding  ddG  (kcal/mol)      lower is better")
    ax.margins(x=0.16)
    ax.set_title("What a hyperparameter you must choose in advance costs you",
                 loc="left", pad=10)
    _style_axes(ax, y_grid_only=False)
    ax.yaxis.grid(False)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.invert_xaxis()  # better designs to the right

    handles = [
        Line2D([], [], color=INK_SECONDARY, marker="o", linestyle="none",
               markersize=MARKER_SIZE, label="best and worst setting"),
        Line2D([], [], color=INK_SECONDARY, marker="D", linestyle="none",
               markersize=MARKER_SIZE, label="average over settings (choosing blind)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, labelcolor=INK_PRIMARY)
    note = "each rule swept over its own hyperparameter; all other settings held fixed"
    if subtitle:
        note = f"{note}\n{subtitle}"
    fig.text(0.008, 0.015, note, fontsize=8, color=INK_MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.17, 1, 0.84))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Budget curves
# --------------------------------------------------------------------------


def budget_curves(
    curves: pd.DataFrame,
    path: Path,
    tau: float,
    methods=HEADLINE_METHODS,
    subtitle: str = "",
) -> Path:
    """Best feasible binding found so far, against oracle calls spent.

    ``curves`` has columns ``method``, ``evaluations``, ``mean``, ``std``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    anchors = []
    for method in methods:
        frame = curves[curves["method"] == method].sort_values("evaluations")
        if frame.empty:
            continue
        color = SERIES_COLORS[method]
        x = frame["evaluations"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        std = np.nan_to_num(frame["std"].to_numpy(dtype=float), nan=0.0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=LINE_WIDTH, solid_capstyle="round")
        finite = np.isfinite(mean)
        if finite.any():
            last = int(np.flatnonzero(finite)[-1])
            ax.plot(x[last], mean[last], marker="o", markersize=MARKER_SIZE, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
            anchors.append(
                ((float(x[last]), float(mean[last])), DISPLAY_NAMES[method], color)
            )

    # Room on the right for direct labels; no room on the left, since there is
    # no such thing as a negative number of oracle calls.
    ax.set_xlim(0, float(curves["evaluations"].max()) * 1.34)
    _place_labels(ax, anchors)
    ax.set_xlabel("oracle calls spent")
    ax.set_ylabel("best feasible binding  ddG  (kcal/mol)\nlower is better")
    ax.set_title(f"Budget curves at tau = {tau:+.2f} kcal/mol", loc="left", pad=10)
    _style_axes(ax)
    _legend(fig, [m for m in methods if (curves["method"] == m).any()])
    note = "band = +/-1 sd over seeds   a line that starts late had found nothing feasible yet"
    if subtitle:
        note = f"{note}\n{subtitle}"
    fig.text(0.008, 0.015, note, fontsize=8, color=INK_MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.1, 1, 0.89))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path

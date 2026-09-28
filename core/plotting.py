"""
core/plotting.py
Reusable matplotlib figures used by pages 5, 6, and 7.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from core.features import parse_age


def _timepoint_to_days(tp_labels):
    """['p45', 'p58', ...] → [45, 58, ...]"""
    return [parse_age(t) for t in tp_labels]


def plot_group_lines(
    df,
    value_col,
    groups,
    colors,
    show_individuals=True,
    error_type="SEM",
    error_style="bars",
    y_label="",
    title="",
    x_lim=None,
    y_lim=None,
    selected_animals=None,
    figsize=(7, 4.5),
):
    """
    Publication-style group trajectory plot.
    df must have columns: animal_id, group, timepoint_label, <value_col>
    """
    fig, ax = plt.subplots(figsize=figsize)

    if selected_animals is not None:
        df = df[df["animal_id"].isin(selected_animals)].copy()

    # Ordered timepoints by numeric age
    all_tps = sorted(df["timepoint_label"].unique(), key=parse_age)
    x_days  = _timepoint_to_days(all_tps)
    tp_to_x = dict(zip(all_tps, x_days))

    for group in groups:
        gdf   = df[df["group"] == group]
        color = colors.get(group, "#888888")

        # Individual animal trajectories
        if show_individuals:
            for aid, adf in gdf.groupby("animal_id"):
                adf_sorted = adf.sort_values("timepoint_label", key=lambda s: s.map(parse_age))
                xs = [tp_to_x[t] for t in adf_sorted["timepoint_label"]]
                ys = adf_sorted[value_col].values
                ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8,
                        linestyle="--", zorder=1)
                ax.scatter(xs, ys, color=color, alpha=0.35, s=18, zorder=2)

        # Group mean ± error
        stats = []
        for tp in all_tps:
            vals = gdf[gdf["timepoint_label"] == tp][value_col].dropna().values
            if len(vals) == 0:
                continue
            m  = np.mean(vals)
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            se = sd / np.sqrt(len(vals))
            err = sd if error_type == "SD" else se
            stats.append((tp_to_x[tp], m, err))

        if not stats:
            continue
        xs_m, ys_m, errs = zip(*stats)
        xs_m  = np.array(xs_m)
        ys_m  = np.array(ys_m)
        errs  = np.array(errs)

        ax.plot(xs_m, ys_m, color=color, linewidth=2.2, zorder=4,
                marker="o", markersize=6, label=group)

        if error_style == "shade":
            ax.fill_between(xs_m, ys_m - errs, ys_m + errs,
                            color=color, alpha=0.15, zorder=3)
        else:
            ax.errorbar(xs_m, ys_m, yerr=errs, fmt="none",
                        color=color, capsize=4, linewidth=1.5, zorder=3)

    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


def plot_gradient_dots(
    df,
    value_col,
    groups,
    group_colors,
    timepoints,
    x_label="P(SOD1)",
    title="",
    selected_animals=None,
    figsize=None,
    x_lim=(0, 1),
):
    """
    Horizontal dot plot: one row per animal, dots colored light→dark by timepoint.
    Animals grouped into blocks by group.
    """
    if selected_animals is not None:
        df = df[df["animal_id"].isin(selected_animals)].copy()

    # Sort timepoints by numeric age
    tp_sorted = sorted(timepoints, key=parse_age)
    n_tp      = len(tp_sorted)
    tp_idx    = {tp: i for i, tp in enumerate(tp_sorted)}

    # Build ordered animal list: group by group, alphabetical within
    animal_order = []
    group_spans  = {}
    for g in groups:
        animals_g = sorted(df[df["group"] == g]["animal_id"].unique())
        group_spans[g] = (len(animal_order), len(animal_order) + len(animals_g))
        animal_order.extend(animals_g)

    if not animal_order:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    n_animals = len(animal_order)
    if figsize is None:
        figsize = (6, max(3, n_animals * 0.35 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    # Colormap per group: light → dark
    for g in groups:
        gcolor = group_colors.get(g, "#888888")
        base   = matplotlib.colors.to_rgb(gcolor)
        # Build gradient from light (0.85) to full saturation
        cmap_colors = []
        for t in range(n_tp):
            alpha = 0.25 + 0.75 * (t / max(n_tp - 1, 1))
            blended = tuple(1 - alpha + alpha * c for c in base)
            cmap_colors.append(blended)

        animals_g = sorted(df[df["group"] == g]["animal_id"].unique())
        for aid in animals_g:
            y_pos = animal_order.index(aid)
            adf   = df[df["animal_id"] == aid]
            for _, row in adf.iterrows():
                tp = row["timepoint_label"]
                if tp not in tp_idx:
                    continue
                val = row[value_col]
                if not np.isfinite(val):
                    continue
                t_i   = tp_idx[tp]
                color = cmap_colors[t_i] if t_i < len(cmap_colors) else base
                ax.scatter(val, y_pos, color=color, s=40, zorder=3,
                           edgecolors="none")

    # Group dividers & labels
    for g, (start, end) in group_spans.items():
        if end > start:
            mid = (start + end - 1) / 2
            ax.text(-0.02, mid, g, ha="right", va="center", fontsize=9,
                    color=group_colors.get(g, "#333333"), fontweight="bold",
                    transform=ax.get_yaxis_transform())
        if start > 0:
            ax.axhline(start - 0.5, color="#cccccc", linewidth=0.8, zorder=1)

    # Y axis: animal IDs
    ax.set_yticks(range(n_animals))
    ax.set_yticklabels(animal_order, fontsize=7)
    ax.set_ylim(-0.8, n_animals - 0.2)
    ax.invert_yaxis()

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_xlim(x_lim)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # Timepoint legend
    if n_tp > 0:
        from matplotlib.lines import Line2D
        legend_elements = []
        for i, tp in enumerate(tp_sorted):
            alpha  = 0.25 + 0.75 * (i / max(n_tp - 1, 1))
            # Use first group color for legend
            base   = matplotlib.colors.to_rgb(group_colors.get(groups[0], "#888888"))
            blended = tuple(1 - alpha + alpha * c for c in base)
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=blended, markersize=7, label=tp))
        ax.legend(handles=legend_elements, title="Timepoint\n(darker=later)",
                  fontsize=7, title_fontsize=7, frameon=False,
                  bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.tight_layout()
    return fig

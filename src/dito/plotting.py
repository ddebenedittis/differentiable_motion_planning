"""Plotting helpers for piecewise-constant signals on non-uniform time grids."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import ConnectionPatch, Rectangle


def plot_timegrid(deltas, x=None, ax=None, ylabel=None, title=None,
                  labels=None):
    """Plot time grid lines and optionally overlay a trajectory.

    Args:
        labels: optional sequence of per-component labels for the overlaid
            trajectory. When given, each state line is labelled and a small
            legend is drawn.
    """
    times = np.cumsum(deltas.tolist())

    if ax is None:
        fig, ax = plt.subplots()
    for t in times:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.25)

    if x is not None:
        lines = ax.plot(times, x)
        if labels is not None:
            for line, lab in zip(lines, labels):
                line.set_label(lab)
            ax.legend(fontsize=7, loc='best')

    ax.set_xlabel("Time")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


def _resolve_cmap_norm(deltas, cmap, norm):
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    if isinstance(norm, str):
        if norm == "log":
            norm = LogNorm(vmin=np.min(deltas), vmax=np.max(deltas))
        elif norm == "linear":
            norm = Normalize(vmin=np.min(deltas), vmax=np.max(deltas))
        else:
            raise ValueError(f"Unknown norm {norm!r}; use 'linear' or 'log'")
    return cmap, norm


def _draw_zoh_segments(ax, times, x, deltas, cmap, norm):
    """Draw piecewise-constant ZOH segments colored by per-step deltas."""
    for i in range(len(x) - 1):
        ax.hlines(x[i], times[i], times[i + 1],
                  colors=cmap(norm(deltas[i + 1])), linewidth=2)
        ax.vlines(times[i + 1], x[i], x[i + 1],
                  colors=cmap(norm(deltas[i + 1])), linewidth=1)


def plot_colored(deltas, x, ax=None, cmap="plasma", norm="linear"):
    """Plot piecewise-constant signal with color indicating timestep duration.

    Args:
        deltas: timestep durations
        x: piecewise-constant signal values
        ax: optional matplotlib axis
        cmap: colormap name or Colormap instance (default "plasma")
        norm: "linear" / "log", or a matplotlib Normalize instance
            (default "linear")
    """
    times = np.cumsum(deltas)
    cmap, norm = _resolve_cmap_norm(deltas, cmap, norm)

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    _draw_zoh_segments(ax, times, x, deltas, cmap, norm)

    ax.set_xlabel("Time")
    ax.set_ylabel("Input")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, location="bottom")
    cbar.set_label("deltas")

    return ax


_ZOOM_ACCENT_COLOR = '#404040'  # dark grey for inset spines / source rect / connectors / title


def add_zoom_inset(parent_ax, fig, *, zoom_xlim, loc, draw_fn=None,
                    time_lines=None, x_for_ylim=None, y_for_ylim=None,
                    bg_alpha=0.85, accent_color=_ZOOM_ACCENT_COLOR,
                    label="Zoom"):
    """Draw a zoom inset on `parent_ax` of x-range `zoom_xlim`.

    `draw_fn(axins)` should replot the data into the inset axes (this lets
    callers reuse whichever rendering they used on the parent — single-color
    line, multi-line, ZOH segments — so colors are preserved).

    `x_for_ylim`/`y_for_ylim`: optional arrays. When given, the inset's
    y-limits are computed from data within `zoom_xlim` (with one neighbor on
    each side and 5% padding) so reference lines on the parent (e.g., bounds
    far from the data) don't squash the zoom.
    """
    inset_bg = Rectangle(
        (loc[0], loc[1]), loc[2], loc[3],
        transform=parent_ax.transAxes,
        facecolor='white', edgecolor='none', alpha=bg_alpha, zorder=15,
    )
    parent_ax.add_patch(inset_bg)

    axins = parent_ax.inset_axes(loc)
    axins.set_zorder(20)
    axins.set_facecolor('none')

    if draw_fn is not None:
        draw_fn(axins)

    if time_lines is not None:
        for t_val in time_lines:
            if zoom_xlim[0] <= t_val <= zoom_xlim[1]:
                axins.axvline(t_val, color='gray', linestyle='-',
                              alpha=0.08, linewidth=0.5)

    axins.set_xlim(zoom_xlim)

    if x_for_ylim is not None and y_for_ylim is not None:
        x_arr = np.asarray(x_for_ylim)
        y_arr = np.asarray(y_for_ylim)
        mask = (x_arr >= zoom_xlim[0]) & (x_arr <= zoom_xlim[1])
        if mask.any():
            idx = np.where(mask)[0]
            lo = max(0, idx.min() - 1)
            hi = min(len(x_arr), idx.max() + 2)
            y_window = y_arr[lo:hi]
            y_lo = float(np.min(y_window))
            y_hi = float(np.max(y_window))
            pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.5 * max(abs(y_lo), 1.0)
            axins.set_ylim(y_lo - pad, y_hi + pad)

    for spine in axins.spines.values():
        spine.set_edgecolor(accent_color)
        spine.set_linewidth(1.0)
    axins.tick_params(labelsize=8, colors=accent_color)

    if label:
        axins.set_title(label, fontsize=8, color=accent_color, pad=2)

    main_ylim = parent_ax.get_ylim()
    source_rect = Rectangle(
        (zoom_xlim[0], main_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        main_ylim[1] - main_ylim[0],
        edgecolor=accent_color, facecolor='none',
        linewidth=1.2, linestyle='--', zorder=5, clip_on=False,
    )
    parent_ax.add_patch(source_rect)

    con_top = ConnectionPatch(
        xyA=(zoom_xlim[1], main_ylim[1]), xyB=(0, 1),
        coordsA='data', coordsB='axes fraction',
        axesA=parent_ax, axesB=axins,
        color=accent_color, linewidth=1.0, linestyle='--', zorder=21,
    )
    con_bot = ConnectionPatch(
        xyA=(zoom_xlim[1], main_ylim[0]), xyB=(0, 0),
        coordsA='data', coordsB='axes fraction',
        axesA=parent_ax, axesB=axins,
        color=accent_color, linewidth=1.0, linestyle='--', zorder=21,
    )
    fig.add_artist(con_top)
    fig.add_artist(con_bot)

    return axins

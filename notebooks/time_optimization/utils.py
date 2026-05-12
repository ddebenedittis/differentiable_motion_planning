"""Example-glue utilities for the differentiable time optimization scripts.

Generic primitives (discretization, losses, r-adapt, ZOHParamSpec, plotting)
live in the `dito` package. This module keeps only the script-level glue
that depends on the pickle/history-dict conventions used by the per-example
training scripts in this directory.
"""

import json
import os
import pickle
import subprocess
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dito import (
    LOSS_REGISTRY,
    _draw_zoh_segments,
    _resolve_cmap_norm,
    add_zoom_inset,
    evaluate_continuous_cost,
    plot_colored,
    plot_timegrid,
)


# ============================================================================ #
# Run Mode Infrastructure
# ============================================================================ #

class RunMode(Enum):
    DISPLAY = "display"  # Load saved results, only display plots
    TEST = "test"        # Quick run with few iterations
    FULL = "full"        # Full training run


DEFAULT_EPOCHS = {
    "aux": 100,
    "rep": 200,
    "hs": 400,
    "zoh": 500,
    "L_IV": 200,
    "L_EQ": 200,
    "L_CPC": 200,
    "L_CSS": 200,
    "L_defect": 200,
    "L_dyn": 200,
    "L_equi": 200,
    "L_FI": 200,
    "L_SC": 200,
    "L_PWLH": 200,
    "L_SSD": 200,
}

TEST_EPOCHS = 5


def get_n_epochs(run_mode, method_key, n_epochs_override=None):
    """Return number of epochs based on run mode and method."""
    if run_mode == RunMode.TEST:
        return TEST_EPOCHS
    if n_epochs_override is not None:
        return n_epochs_override
    return DEFAULT_EPOCHS[method_key]


def get_method_run_mode(run_mode, method_key, run_overrides=None):
    """Return effective RunMode for a method, checking overrides first."""
    if run_overrides and method_key in run_overrides:
        return run_overrides[method_key]
    return run_mode


def pickle_name(base_name, n, n_default=None):
    """Append _nXX suffix when n differs from default (backward compatible)."""
    if n_default is not None and n != n_default:
        return f"{base_name}_n{n}"
    return base_name


# ============================================================================ #
# Schema-aware plotting (depends on sol_method 1/2 convention)
# ============================================================================ #

def _extract_dts(sol, history, n, sol_method):
    """Extract timestep array from solution/history based on method type."""
    if sol_method == 1:
        return np.concatenate(
            np.array([d.detach().numpy().tolist() for d in sol[2 * n:3 * n]])
        )
    elif sol_method == 2:
        return np.array(history[-1]['dts']).flatten()
    else:
        raise ValueError(f"Unknown sol_method {sol_method}")


def plot_training_res(sol, history, n, sol_method, cmap="plasma", norm="linear",
                       zoom_xlim=None, zoom_loc_state=None,
                       zoom_loc_input=None, ct_sol=None):
    """Plot training results (2x2 grid): loss, timesteps, state, colored input.

    Args:
        sol: solution tensors (list of torch tensors)
        history: list of dicts with 'loss' and 'dts' keys
        n: number of timesteps
        sol_method: 1 for aux (dts in sol), 2 for rep/zoh (dts in history)
        cmap: colormap for the colored-input subplot (default "plasma")
        norm: "linear" / "log" or a Normalize instance (default "linear")
        zoom_xlim: optional (t_lo, t_hi). When given, adds zoom insets on the
            state and input subplots (e.g., (0.0, 0.5) for stiff systems).
        zoom_loc_state: inset [x, y, w, h] for the state subplot (default
            bottom-right).
        zoom_loc_input: inset [x, y, w, h] for the input subplot (default
            top-right).
        ct_sol: optional dict with keys 'u_arr' and 'dts' giving a
            uniformly-sampled CT reference input trajectory to overlay on the
            colored-input subplot as a dashed grey line. When provided, a
            legend is added.
    """
    s_arr = np.array([s.detach().numpy().tolist() for s in sol[0:n]])
    u_arr = np.array([u.detach().numpy().tolist() for u in sol[n:2 * n]])
    d_arr = _extract_dts(sol, history, n, sol_method)

    fig, ax = plt.subplots(2, 2, figsize=(6.4, 6.4))
    ax[0, 0].plot([h['loss'] for h in history])
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].set_title("Loss Evolution")

    ax[0, 1].step(np.cumsum(d_arr), d_arr, where='pre')
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Timestep duration")
    ax[0, 1].set_title("Timesteps Evolution")

    plot_timegrid(d_arr, s_arr, ax[1, 0], ylabel="State", title="State Evolution")
    plot_colored(d_arr, u_arr, ax[1, 1], cmap=cmap, norm=norm)

    if ct_sol is not None:
        u_ct = np.asarray(ct_sol['u_arr']).flatten()
        dts_ct = np.asarray(ct_sol['dts']).flatten()
        n_ct = len(u_ct)
        times_ct = np.concatenate(([0.0], np.cumsum(dts_ct)))
        u_ct_step = np.concatenate((u_ct, u_ct[-1:]))
        ax[1, 1].step(
            times_ct, u_ct_step, where='post',
            linestyle='--', color='grey', linewidth=1.0, zorder=3,
        )
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        legend_elements = [
            Line2D([0], [0], color=cmap_obj(0.5), lw=2,
                   label=f"Non uniform - {n} steps"),
            Line2D([0], [0], color='grey', linestyle='--', lw=1.0,
                   label=f"Uniform - {n_ct} steps"),
        ]
        ax[1, 1].legend(handles=legend_elements, fontsize=7, loc='best')

    if zoom_xlim is not None:
        times_state_input = np.cumsum(d_arr)
        if zoom_loc_state is None:
            zoom_loc_state = [0.40, 0.08, 0.55, 0.55]  # bottom-right
        if zoom_loc_input is None:
            zoom_loc_input = [0.40, 0.40, 0.55, 0.55]  # top-right

        # State inset: replot multi-line state with default color cycle
        # so colors match the parent.
        add_zoom_inset(
            ax[1, 0], fig,
            zoom_xlim=zoom_xlim, loc=zoom_loc_state,
            draw_fn=lambda a: a.plot(times_state_input, s_arr),
            time_lines=times_state_input,
            x_for_ylim=times_state_input, y_for_ylim=s_arr,
        )

        # Input inset: replot ZOH segments with the same cmap/norm.
        cmap_r, norm_r = _resolve_cmap_norm(d_arr, cmap, norm)
        add_zoom_inset(
            ax[1, 1], fig,
            zoom_xlim=zoom_xlim, loc=zoom_loc_input,
            draw_fn=lambda a: _draw_zoh_segments(
                a, times_state_input, u_arr, d_arr, cmap_r, norm_r,
            ),
            time_lines=times_state_input,
            x_for_ylim=times_state_input, y_for_ylim=u_arr,
        )

    fig.set_constrained_layout(True)


# ============================================================================ #
# I/O
# ============================================================================ #

def save_training_res(out_dir, exp_name, sol, history, n, sol_method,
                      cmap="plasma", norm="linear",
                      zoom_xlim=None, zoom_loc_state=None,
                      zoom_loc_input=None):
    """Save training result plots to out_dir/exp_name/.

    Args:
        out_dir: base output directory
        exp_name: experiment name (used as subdirectory)
        sol: solution tensors
        history: training history
        n: number of timesteps
        sol_method: 1 for aux, 2 for rep/zoh
        cmap: colormap for the colored-input plot
        norm: "linear" / "log" or a Normalize instance
        zoom_xlim: optional (t_lo, t_hi). When given, adds a zoom inset to
            the saved state.pdf and input.pdf figures.
        zoom_loc_state: inset [x, y, w, h] for state.pdf (default bottom-right).
        zoom_loc_input: inset [x, y, w, h] for input.pdf (default top-right).
    """
    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    s_arr = np.array([s.detach().numpy().tolist() for s in sol[0:n]])
    u_arr = np.array([u.detach().numpy().tolist() for u in sol[n:2 * n]])
    d_arr = _extract_dts(sol, history, n, sol_method)

    if zoom_xlim is not None:
        if zoom_loc_state is None:
            zoom_loc_state = [0.40, 0.08, 0.55, 0.55]
        if zoom_loc_input is None:
            zoom_loc_input = [0.40, 0.40, 0.55, 0.55]
        times_si = np.cumsum(d_arr)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    plot_colored(d_arr, u_arr, ax, cmap=cmap, norm=norm)
    if zoom_xlim is not None:
        cmap_r, norm_r = _resolve_cmap_norm(d_arr, cmap, norm)
        add_zoom_inset(
            ax, fig,
            zoom_xlim=zoom_xlim, loc=zoom_loc_input,
            draw_fn=lambda a: _draw_zoh_segments(
                a, times_si, u_arr, d_arr, cmap_r, norm_r,
            ),
            time_lines=times_si,
            x_for_ylim=times_si, y_for_ylim=u_arr,
        )
    fig.savefig(f"{exp_dir}/input.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.step(np.cumsum(d_arr), d_arr, where='pre')
    ax.set_xlabel("Time")
    ax.set_ylabel("Timestep duration")
    fig.savefig(f"{exp_dir}/timesteps.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.plot([h['loss'] for h in history])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(f"{exp_dir}/loss.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    plot_timegrid(d_arr, s_arr, ax, ylabel="State")
    if zoom_xlim is not None:
        add_zoom_inset(
            ax, fig,
            zoom_xlim=zoom_xlim, loc=zoom_loc_state,
            draw_fn=lambda a: a.plot(times_si, s_arr),
            time_lines=times_si,
            x_for_ylim=times_si, y_for_ylim=s_arr,
        )
    fig.savefig(f"{exp_dir}/state.pdf", bbox_inches='tight')
    plt.close(fig)


def save_radapt_diagnostics(out_dir, exp_name, history, n,
                            color_accepted="#0072B2",
                            color_rejected="#E69F00"):
    """Save r-adapt move outcomes and accepted move participation as PDFs.

    Produces move_outcomes.pdf and accepted_move_participation.pdf inside
    out_dir/exp_name/.  Returns False (and does nothing) when the history
    contains no r-adapt attempts.
    """
    attempts = [h for h in history if h.get("radapt_attempted")]
    if not attempts:
        return False

    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    eps_attempt = np.array([h["epoch"] for h in attempts])
    dLs = np.array([h["radapt_dL"] for h in attempts])
    accepted = np.array([h["radapt_accepted"] for h in attempts])

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.scatter(eps_attempt[accepted], dLs[accepted],
               color=color_accepted, marker="o", label="accepted")
    ax.scatter(eps_attempt[~accepted], dLs[~accepted],
               color=color_rejected, marker="x", label="rejected")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\Delta L$")
    ax.legend(fontsize=7)
    fig.savefig(f"{exp_dir}/move_outcomes.pdf", bbox_inches='tight')
    plt.close(fig)

    js = np.array([h["radapt_j"] for h in attempts if h["radapt_accepted"]])
    is_ = np.array([h["radapt_i"] for h in attempts if h["radapt_accepted"]])
    bins = np.arange(-0.5, n + 0.5, 1)
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.hist([js, is_], bins=bins, label=["merge j", "split i"], stacked=False)
    ax.set_xlabel("Interval index")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    fig.savefig(f"{exp_dir}/accepted_move_participation.pdf",
                bbox_inches='tight')
    plt.close(fig)

    return True


def save_pickle(out_dir, name, data):
    """Save data to a pickle file."""
    with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f)


def load_pickle(out_dir, name):
    """Load data from a pickle file."""
    with open(os.path.join(out_dir, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def load_losses_config(path=None):
    """Load shared losses config from JSON.

    If path is None, looks for losses_config.json next to utils.py.
    Returns the parsed dict, or None if the file does not exist.
    """
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "losses_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def resolve_loss_names(loss_arg, config=None):
    """Expand '--loss all' using the config's enabled list or LOSS_REGISTRY.

    Args:
        loss_arg: list from argparse (e.g. ["all"] or ["L_IV", "L_FI"])
        config: loaded config dict from load_losses_config(), or None

    Returns:
        list of loss names to run
    """
    if "all" in loss_arg:
        if config is not None and "enabled" in config:
            return list(config["enabled"])
        return list(LOSS_REGISTRY.keys())
    return list(loss_arg)


def save_run_config(data_dir, args):
    """Dump CLI args + git hash to run_config.json."""
    config = vars(args).copy()
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    config["git_hash"] = git_hash
    with open(os.path.join(data_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def save_dts_distribution(out_dir, name, history):
    """Save full dts distribution across all epochs as a numpy array.

    Args:
        out_dir: output directory
        name: pickle file name (without .pkl extension)
        history: list of history dicts, each containing a 'dts' key

    Returns:
        dts_all: numpy array of shape (n_epochs, n)
    """
    dts_all = np.stack([np.array(h['dts']).flatten() for h in history])
    save_pickle(out_dir, name, dts_all)
    return dts_all


def save_timesteps_video(out_dir, name, history=None, T=None, fps=15, dpi=100,
                         dts_all=None):
    """Save animation of timestep distribution evolution to mp4 (or gif fallback).

    Args:
        out_dir: output directory
        name: output file name (without extension)
        history: list of history dicts with 'dts' key (used if dts_all is None)
        T: total time horizon (inferred from dts_all if None)
        fps: frames per second
        dpi: resolution
        dts_all: numpy array of shape (n_epochs, n), alternative to history

    Returns:
        out_path: path to the saved video file
    """
    import matplotlib.animation as animation

    if dts_all is None:
        dts_all = np.stack([np.array(h['dts']).flatten() for h in history])
    if T is None:
        T = float(dts_all[0].sum())
    n_epochs, n = dts_all.shape
    starts_all = np.concatenate(
        [np.zeros((n_epochs, 1)), np.cumsum(dts_all, axis=1)[:, :-1]], axis=1
    )

    dt_min = dts_all.min() * 0.9
    dt_max = dts_all.max() * 1.1
    dt_uniform = T / n

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    (line,) = ax.plot([], [], 'b-', linewidth=1.2, drawstyle='steps-post')
    ax.set_xlim(0, T)
    ax.set_ylim(dt_min, dt_max)
    ax.set_xlabel("Time")
    ax.set_ylabel("Timestep duration")
    ax.axhline(dt_uniform, color='gray', linestyle='--', alpha=0.5, label='uniform')
    ax.legend(fontsize=8)
    title = ax.set_title("Epoch 0")
    fig.set_constrained_layout(True)

    def init():
        line.set_data([], [])
        return line, title

    def update(frame):
        line.set_data(starts_all[frame], dts_all[frame])
        title.set_text(f"Epoch {frame}")
        return line, title

    anim = animation.FuncAnimation(
        fig, update, frames=np.arange(n_epochs),
        init_func=init, blit=True, interval=1000 / fps,
    )

    out_path = os.path.join(out_dir, f"{name}.mp4")
    try:
        anim.save(out_path, fps=fps, dpi=dpi, writer='ffmpeg')
    except Exception:
        out_path = os.path.join(out_dir, f"{name}.gif")
        anim.save(out_path, fps=fps, dpi=dpi, writer='pillow')

    plt.close(fig)
    return out_path


# ============================================================================ #
# Analysis
# ============================================================================ #

def extract_trajectory_data(ms, n):
    """Extract states, inputs, and timesteps from a method solution."""
    sol, history, sol_method = ms['sol'], ms['history'], ms.get('sol_method', 2)

    s_arr = np.array([sol[i].detach().numpy() for i in range(n)])
    u_arr = np.array([sol[n + i].detach().numpy() for i in range(n)]).flatten()

    if sol_method == 1:
        dts = np.concatenate([sol[2 * n + i].detach().numpy() for i in range(n)]).flatten()
    else:
        dts = np.array(history[-1]['dts']).flatten()

    times = np.cumsum(dts)
    return {'s': s_arr, 'u': u_arr, 'dts': dts, 'times': times}


def compute_trajectory_metrics(data, n, T):
    """Compute sampling density and trajectory change metrics."""
    s, u, dts = data['s'], data['u'], data['dts']
    dt_uniform = T / n

    return {
        'sampling_density': (1.0 / dts) * dt_uniform,
        'abs_u': np.abs(u),
        'delta_u': np.abs(np.diff(u)),
        'norm_s': np.linalg.norm(s, axis=1),
        'delta_s': np.linalg.norm(np.diff(s, axis=0), axis=1),
    }


def plot_density_and_changes(data, metrics, method_name, colors, axes=None):
    """Plot sampling density, |Delta u|, and ||Delta s|| on the same axes."""
    times = data['times']
    ax = axes if axes is not None else plt.subplots(figsize=(3.2, 2.4))[1]

    ax.plot(times, metrics['sampling_density'], label=r'Sampling density', color=colors[0])
    ax.plot(times[:-1], metrics['delta_u'], label=r'$|\Delta u|$', color=colors[1])
    ax.plot(times[:-1], metrics['delta_s'], label=r'$\|\Delta s\|_2$', color=colors[2])
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set(ylabel='Value', title=method_name)
    ax.legend(loc='upper right', fontsize=7)


# ============================================================================ #
# Shared Plotting Helpers (used by per-example plot scripts)
# ============================================================================ #

@dataclass(frozen=True)
class MethodConfig:
    """Configuration for loading a method's pickle results."""
    key: str
    sol_pickle: str
    history_pickle: str
    internal_methods: tuple[str, ...]
    sol_method: int = 2
    uses_custom_n: bool = False


def load_method_results(data_dir, method_configs, suffix="",
                        n_default=None, n_custom=None):
    """Load all available method results from pickle files.

    For configs with `uses_custom_n=True`, the pickle base name is suffixed
    with `_n{n_custom}` whenever `n_custom != n_default` (matching the
    `pickle_name` convention).
    """
    results = {}

    for cfg in method_configs:
        try:
            if cfg.uses_custom_n and n_default is not None and n_custom is not None:
                sol_base = pickle_name(cfg.sol_pickle, n_custom, n_default)
                hist_base = pickle_name(cfg.history_pickle, n_custom, n_default)
            else:
                sol_base = cfg.sol_pickle
                hist_base = cfg.history_pickle

            sol = load_pickle(data_dir, sol_base + suffix)
            history = load_pickle(data_dir, hist_base + suffix)
            n_inferred = len(np.array(history[-1]['dts']).flatten())
            results[cfg.key] = {
                "sol": sol,
                "history": history,
                "n": n_inferred,
                "internal_methods": list(cfg.internal_methods),
                "sol_method": cfg.sol_method,
            }
        except (FileNotFoundError, OSError):
            pass

    return results


def load_loss_results(data_dir, loss_names=None, suffix=""):
    """Load all available loss results from pickle files.

    Args:
        loss_names: None (no losses), "all" (all available), or list of names.
        suffix: optional pickle name suffix

    Returns:
        results: dict of {loss_name: {"sol": ..., "history": ..., "n": ...}}
    """
    if loss_names is None:
        return {}
    if loss_names == "all":
        candidates = list(LOSS_REGISTRY.keys())
    else:
        candidates = loss_names
    results = {}

    for loss_name in candidates:
        try:
            sol = load_pickle(data_dir, f"sol_{loss_name}{suffix}")
            history = load_pickle(data_dir, f"history_{loss_name}{suffix}")
            n_inferred = len(np.array(history[-1]['dts']).flatten())
            results[loss_name] = {"sol": sol, "history": history, "n": n_inferred}
        except (FileNotFoundError, OSError):
            pass

    return results


def plot_method_results(name, result, results_dir, show=False,
                        cmap="plasma", norm="linear"):
    """Plot training results for a single method (2x2 grid per sub-method)."""
    sol_dict = result["sol"]
    history = result["history"]
    n_method = result["n"]
    sol_method = result.get("sol_method", 2)
    internal_methods = result["internal_methods"]

    for method in internal_methods:
        sol = sol_dict[method]
        hist_m = [h for h in history if h['method'] == method]
        if not hist_m:
            print(f"  No history for {name}/{method}, skipping")
            continue

        plot_training_res(sol, hist_m, n_method, sol_method=sol_method,
                          cmap=cmap, norm=norm)
        plt.suptitle(f"{name}: {method}")

        if results_dir:
            _key = method.replace(' ', '_')
            save_training_res(results_dir, f"{name}_{_key}", sol, hist_m,
                              n_method, sol_method, cmap=cmap, norm=norm)

        if not show:
            plt.close('all')


def plot_loss_results(loss_name, result, results_dir, show=False,
                      cmap="plasma", norm="linear",
                      zoom_xlim=None, zoom_loc_state=None,
                      zoom_loc_input=None, ct_sol=None):
    """Plot training results for a single loss (2x2 grid).

    When ``ct_sol`` is provided (dict with 'u_arr' and 'dts'), the input
    subplot also shows the dense uniform reference as a dashed grey line and
    a legend distinguishing the two trajectories.
    """
    sol = result["sol"]
    history = result["history"]
    n = result["n"]

    plot_training_res(
        sol, history, n, sol_method=2, cmap=cmap, norm=norm,
        zoom_xlim=zoom_xlim, zoom_loc_state=zoom_loc_state,
        zoom_loc_input=zoom_loc_input,
        ct_sol=ct_sol,
    )
    plt.suptitle(loss_name)

    if results_dir:
        save_training_res(results_dir, loss_name, sol, history, n, sol_method=2,
                          cmap=cmap, norm=norm,
                          zoom_xlim=zoom_xlim,
                          zoom_loc_state=zoom_loc_state,
                          zoom_loc_input=zoom_loc_input)

    if not show:
        plt.close('all')


def plot_loss_comparison(loss_results, results_dir, show=False,
                         filename="loss_comparison"):
    """Side-by-side timestep distributions for all losses."""
    if len(loss_results) <= 1:
        return

    n_losses = len(loss_results)
    fig, axes = plt.subplots(1, n_losses, figsize=(3.2 * n_losses, 3.2))
    if n_losses == 1:
        axes = [axes]

    for ax, (loss_name, result) in zip(axes, loss_results.items()):
        history = result["history"]
        dts_final = history[-1]['dts'].flatten()
        times = np.cumsum(dts_final)
        ax.plot(times, dts_final)
        ax.set_xlabel("Time")
        ax.set_ylabel("dt")
        ax.set_title(loss_name)

    fig.set_constrained_layout(True)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, f"{filename}.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)


def plot_density_analysis_grid(method_solutions, T, colors, results_dir,
                               show=False):
    """Plot sampling density vs trajectory changes for all methods."""
    n_methods = len(method_solutions)
    if n_methods == 0:
        return

    n_rows = int(np.ceil(n_methods / 2))
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2.5 * n_rows),
                            squeeze=False)

    for i, (key, ms) in enumerate(method_solutions.items()):
        n_m = ms['n']
        data = extract_trajectory_data(ms, n_m)
        metrics = compute_trajectory_metrics(data, n_m, T)
        plot_density_and_changes(data, metrics, key, colors,
                                axes=axs[i // 2, i % 2])

    for j in range(i + 1, n_rows * 2):
        fig.delaxes(axs[j // 2, j % 2])

    fig.set_constrained_layout(True)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "density_analysis.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)


def print_continuous_costs(method_results, loss_results, *, s0_eval, A, B, Q,
                           R, T):
    """Evaluate and print true continuous-time costs."""
    print("=== True Continuous-Time Cost Comparison ===\n")

    for name, result in method_results.items():
        sol_dict = result["sol"]
        history = result["history"]
        n_method = result["n"]

        for method in result["internal_methods"]:
            sol = sol_dict[method]
            hist_m = [h for h in history if h['method'] == method]
            if not hist_m:
                continue

            dts_final = hist_m[-1]['dts']
            inputs_qp = [sol[n_method + i].detach().float()
                         for i in range(n_method)]

            try:
                true_cost = evaluate_continuous_cost(
                    inputs_qp, dts_final, s0_eval, A, B, Q, R, T,
                )
            except Exception as exc:
                print(f"{name} ({method}): cost evaluation failed: {exc}")
                continue

            print(f"{name} ({method}):")
            print(f"  Training loss (final): {hist_m[-1]['loss']:.4f}")
            print(f"  True continuous cost:  {true_cost:.4f}")
            if isinstance(dts_final, np.ndarray):
                print(f"  dt range: [{np.min(dts_final):.5f}, "
                      f"{np.max(dts_final):.5f}]")
                print(f"  dt std:   {np.std(dts_final):.5f}")
            print()

    for loss_name, result in loss_results.items():
        sol = result["sol"]
        history = result["history"]
        n_loss = result["n"]

        dts_final = history[-1]['dts']
        inputs_qp = [sol[n_loss + k].detach().float() for k in range(n_loss)]

        try:
            true_cost = evaluate_continuous_cost(
                inputs_qp, dts_final, s0_eval, A, B, Q, R, T,
            )
        except Exception as exc:
            print(f"{loss_name}: cost evaluation failed: {exc}")
            continue

        print(f"{loss_name}:")
        print(f"  Training loss (final):  {history[-1]['loss']:.4f}")
        print(f"  Loss OCP (final):       {history[-1].get('loss_ocp', 'N/A')}")
        print(f"  Loss reg (final):       {history[-1].get('loss_reg', 'N/A')}")
        print(f"  Lambda hat (final):     {history[-1].get('lambda_hat', 'N/A')}")
        print(f"  True continuous cost:   {true_cost:.4f}")
        if isinstance(dts_final, np.ndarray):
            print(f"  dt range: [{np.min(dts_final):.5f}, "
                  f"{np.max(dts_final):.5f}]")
            print(f"  dt std:   {np.std(dts_final):.5f}")
        print()


def save_summary(method_results, loss_results, results_dir, *, s0_eval, A, B,
                 Q, R, T):
    """Compute and save metrics summary JSON."""
    summary = {}

    for name, result in method_results.items():
        for method in result["internal_methods"]:
            sol = result["sol"][method]
            hist_m = [h for h in result["history"]
                      if h['method'] == method]
            if not hist_m:
                continue
            n_m = result["n"]
            dts_final = hist_m[-1]['dts']
            inputs_qp = [sol[n_m + i].detach().float() for i in range(n_m)]
            try:
                cont_cost = evaluate_continuous_cost(
                    inputs_qp, dts_final, s0_eval, A, B, Q, R, T)
            except Exception:
                cont_cost = None
            summary[f"{name}_{method}"] = {
                "continuous_cost": cont_cost,
                "final_loss": hist_m[-1]['loss'],
            }

    for loss_name, result in loss_results.items():
        sol = result["sol"]
        history = result["history"]
        n_loss = result["n"]
        dts_final = history[-1]['dts']
        inputs_qp = [sol[n_loss + k].detach().float() for k in range(n_loss)]
        try:
            cont_cost = evaluate_continuous_cost(
                inputs_qp, dts_final, s0_eval, A, B, Q, R, T)
        except Exception:
            cont_cost = None
        entry = {
            "continuous_cost": cont_cost,
            "final_loss": history[-1]['loss'],
        }
        if 'loss_ocp' in history[-1]:
            entry["final_loss_ocp"] = history[-1]['loss_ocp']
            entry["final_loss_reg"] = history[-1]['loss_reg']
            entry["final_lambda_hat"] = history[-1]['lambda_hat']
        summary[loss_name] = entry

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, metrics in summary.items():
        cost_str = (f"{metrics['continuous_cost']:.6f}"
                    if metrics['continuous_cost'] is not None else "N/A")
        print(f"\n{label}:")
        print(f"  Continuous cost:  {cost_str}")
        print(f"  Final loss:       {metrics['final_loss']:.6f}")

    return summary


# ============================================================================ #
# Plot driver (shared across per-example plot scripts)
# ============================================================================ #

def build_method_solutions(method_results, loss_results, *,
                           label_with_n=False, n_default=None,
                           prefix_map=None):
    """Build the cross-method dict consumed by plot_density_analysis_grid."""
    method_solutions = {}

    for key, result in method_results.items():
        sol_dict = result["sol"]
        history = result["history"]
        n_method = result["n"]
        sol_method = result.get("sol_method", 2)
        prefix = (prefix_map or {}).get(key, key.upper())

        for method in result["internal_methods"]:
            if label_with_n and n_default is not None and n_method != n_default:
                label = f"{prefix} (n={n_method}): {method}"
            else:
                label = f"{prefix}: {method}"
            method_solutions[label] = {
                'sol': sol_dict[method],
                'history': [h for h in history if h['method'] == method],
                'sol_method': sol_method,
                'n': n_method,
            }

    for loss_name, result in loss_results.items():
        method_solutions[loss_name] = {
            'sol': result['sol'],
            'history': result['history'],
            'sol_method': 2,
            'n': result['n'],
        }

    return method_solutions


def run_plot_main(spec, method_configs, *, parser_extras=None,
                  trajectory_plot_fn=None,
                  baseline_factory=None,
                  baseline_n_fallback=40,
                  ct_sol=None,
                  zoom_xlim=None,
                  zoom_loc_state=None,
                  zoom_loc_input=None,
                  build_solutions_kwargs=None,
                  load_method_kwargs_fn=None,
                  extra_steps=None):
    """Driver for per-example plotting scripts.

    Args:
        spec: training.ProblemSpec (provides s0_eval, A, B, Q, R, T, ...)
        method_configs: list[MethodConfig] passed to load_method_results
        parser_extras: callable(parser) to add example-specific CLI flags
        trajectory_plot_fn: optional callable
            (dts, n, *, sol=None, s_arr=None, u_arr=None, title=None,
             results_dir=None, filename=None, show=False)
            invoked for each method/loss + uniform baseline.
        baseline_factory: optional callable(n) -> (prob, s_vars, u_vars) used
            to build the uniform-baseline plot. Each variable list mirrors
            cvxpy semantics: s_vars[i+1].value gives state at step i+1.
        baseline_n_fallback: default n when no methods/losses are loaded.
        ct_sol: optional dict for plot_loss_results overlay (see utils).
        zoom_xlim/zoom_loc_state/zoom_loc_input: forwarded to plot_loss_results.
        build_solutions_kwargs: kwargs forwarded to build_method_solutions
            (e.g., {"label_with_n": True, "n_default": 160}).
        extra_steps: list of callables(args, results_dir, method_results,
            loss_results) -- example-specific extras (videos, HS comparison).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Visualization for {spec.name} time optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--method", nargs="+", default=None)
    parser.add_argument("--loss", nargs="+", default=None)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    if parser_extras is not None:
        parser_extras(parser)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(here, "data", spec.data_dirname)
    results_dir = (args.results_dir
                   or os.path.join(here, "results", spec.data_dirname))

    try:
        from dimp.utils import init_matplotlib, get_colors
        init_matplotlib()
        colors = get_colors()
    except ImportError:
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442',
                  '#D55E00']

    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    load_kwargs = (load_method_kwargs_fn(args)
                   if load_method_kwargs_fn is not None else {})
    method_results = load_method_results(data_dir, method_configs,
                                         **load_kwargs)
    loss_arg = args.loss if args.loss is not None else "all"
    loss_results = load_loss_results(data_dir, loss_names=loss_arg)

    if args.method:
        method_results = {k: v for k, v in method_results.items()
                          if k in args.method}

    if not method_results and not loss_results:
        print("No results found.")
        return

    print(f"Loaded methods: {list(method_results.keys())}")
    print(f"Loaded losses: {list(loss_results.keys())}")

    if not args.analysis_only:
        for name, result in method_results.items():
            print(f"Plotting {name}...")
            plot_method_results(name, result, results_dir, show=args.show)

            if trajectory_plot_fn is not None:
                for method in result["internal_methods"]:
                    sol = result["sol"][method]
                    hist_m = [h for h in result["history"]
                              if h['method'] == method]
                    if not hist_m:
                        continue
                    dts_final = np.array(hist_m[-1]['dts']).flatten()
                    trajectory_plot_fn(
                        dts_final, result["n"], sol=sol,
                        title=f"{name}: {method}",
                        results_dir=results_dir,
                        filename=f"trajectory_{name}_{method}",
                        show=args.show,
                    )

        # Uniform baseline trajectory.
        if baseline_factory is not None and trajectory_plot_fn is not None:
            if method_results:
                n_uniform = next(iter(method_results.values()))["n"]
            elif loss_results:
                n_uniform = next(iter(loss_results.values()))["n"]
            else:
                n_uniform = baseline_n_fallback
            try:
                prob, s_vars, u_vars = baseline_factory(n_uniform)
                prob.solve()
                if prob.status in ("optimal", "optimal_inaccurate"):
                    print(f"--- Uniform: OCP cost ---")
                    print(f"  n={n_uniform}, "
                          f"cost={prob.objective.value:.6f}")
                    dts_uniform = np.full(n_uniform, spec.T / n_uniform)
                    s_arr = np.vstack(
                        [spec.s0]
                        + [np.asarray(s_vars[i + 1].value).flatten()
                           for i in range(n_uniform)])
                    u_arr = np.array([np.asarray(u_vars[i].value).flatten()
                                      for i in range(n_uniform)])
                    trajectory_plot_fn(
                        dts_uniform, n_uniform,
                        s_arr=s_arr, u_arr=u_arr,
                        title=(f"Uniform baseline (n={n_uniform}, "
                               f"cost={prob.objective.value:.4f})"),
                        results_dir=results_dir,
                        filename="trajectory_uniform_baseline",
                        show=args.show,
                    )
                else:
                    print(f"Uniform baseline: {prob.status}")
            except Exception as exc:
                print(f"Uniform baseline failed: {exc}")

        for loss_name, result in loss_results.items():
            print(f"Plotting {loss_name}...")
            print(f"  OCP cost: {result['history'][-1].get('loss_ocp', 0):.6f}")
            plot_loss_results(loss_name, result, results_dir, show=args.show,
                              ct_sol=ct_sol, zoom_xlim=zoom_xlim,
                              zoom_loc_state=zoom_loc_state,
                              zoom_loc_input=zoom_loc_input)

            if trajectory_plot_fn is not None:
                dts_final = np.array(result["history"][-1]['dts']).flatten()
                trajectory_plot_fn(
                    dts_final, result["n"], sol=result["sol"],
                    title=loss_name,
                    results_dir=results_dir,
                    filename=f"trajectory_{loss_name}",
                    show=args.show,
                )

        print_continuous_costs(method_results, loss_results,
                               s0_eval=spec.s0_loss, A=spec.A, B=spec.B,
                               Q=spec.Q, R=spec.R, T=spec.T)

        plot_loss_comparison(loss_results, results_dir, show=args.show)

    method_solutions = build_method_solutions(
        method_results, loss_results,
        **(build_solutions_kwargs or {}))
    if method_solutions:
        print(f"Analysis: {len(method_solutions)} variants")
        plot_density_analysis_grid(
            method_solutions, spec.T, colors, results_dir, show=args.show)

    save_summary(method_results, loss_results, results_dir,
                 s0_eval=spec.s0_loss, A=spec.A, B=spec.B, Q=spec.Q, R=spec.R,
                 T=spec.T)

    if extra_steps:
        for step in extra_steps:
            step(args, results_dir, method_results, loss_results)

    print(f"\nPlots saved to: {results_dir}")

    if args.show:
        plt.show()

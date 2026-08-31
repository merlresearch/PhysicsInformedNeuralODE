# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the Burgers PINODE example and save logs, data, and figures.

This script reads saved experiment outputs from ``examples/saved_outputs``
by default and writes a timestamped run directory under ``examples/outputs``.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import json as _json
import logging
import os
import pickle
import platform
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import altair as alt
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.figure as _mpl_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from reportlab.graphics import renderPDF  # noqa: E402
from svglib.svglib import svg2rlg  # noqa: E402
from tqdm import tqdm  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve()
EXAMPLES_DIR = SCRIPT_PATH.parent
REPO_ROOT = EXAMPLES_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pinode.burgers.data_generation import generate_data  # noqa: E402
from pinode.core.dmd import DMD  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for logs, figures, data, and optional animations.",
    )
    parser.add_argument(
        "--saved-outputs-dir",
        type=Path,
        default=EXAMPLES_DIR / "saved_outputs",
        help=(
            "Directory containing experiment_*.pickle files and saved output "
            "versions."
        ),
    )
    parser.add_argument(
        "--save-animations",
        action="store_true",
        help="Save generated animations when the script creates them.",
    )
    return parser.parse_args(argv)


RUN_TIMESTAMP: str
OUTPUT_DIR: Path
FIGURES_DIR: Path
LOGS_DIR: Path
DATA_DIR: Path
ANIMATIONS_DIR: Path
SAVED_OUTPUTS_DIR: Path
SAVE_ANIMATIONS: bool
LOG_FILE: Path
_RUN_LOG_STREAM = None


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception:\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    )
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


_SAVED_FIGURE_IDS: set[int] = set()
_ORIGINAL_PLT_SAVEFIG = plt.savefig
_ORIGINAL_FIGURE_SAVEFIG = _mpl_figure.Figure.savefig


def _redirect_figure_path(path):
    if isinstance(path, (str, os.PathLike)):
        candidate = Path(path)
        if (
            not candidate.is_absolute()
            and candidate.parts
            and candidate.parts[0] == "figures"
        ):
            candidate = FIGURES_DIR / Path(*candidate.parts[1:])
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    return path


def _logged_plt_savefig(*args, **kwargs):
    if args:
        args = (_redirect_figure_path(args[0]), *args[1:])
        logging.info("Saving figure: %s", args[0])
    _SAVED_FIGURE_IDS.add(id(plt.gcf()))
    return _ORIGINAL_PLT_SAVEFIG(*args, **kwargs)


def _logged_figure_savefig(self, fname, *args, **kwargs):
    fname = _redirect_figure_path(fname)
    logging.info("Saving figure: %s", fname)
    _SAVED_FIGURE_IDS.add(id(self))
    return _ORIGINAL_FIGURE_SAVEFIG(self, fname, *args, **kwargs)


def _logged_show(*args, **kwargs):
    for number in plt.get_fignums():
        fig = plt.figure(number)
        if id(fig) not in _SAVED_FIGURE_IDS:
            auto_path = FIGURES_DIR / f"auto_figure_{number:03d}.pdf"
            logging.info("Auto-saving unsaved figure before show(): %s", auto_path)
            _logged_figure_savefig(fig, auto_path, bbox_inches="tight")
    logging.info("Skipping interactive plt.show() under Agg backend")
    return None


def _configure_run(args: argparse.Namespace) -> None:
    """Configure output paths, logging, and non-interactive figure handling."""
    global RUN_TIMESTAMP, OUTPUT_DIR, FIGURES_DIR, LOGS_DIR, DATA_DIR
    global ANIMATIONS_DIR, SAVED_OUTPUTS_DIR, SAVE_ANIMATIONS, LOG_FILE
    global _RUN_LOG_STREAM

    RUN_TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = (
        args.output_dir
        or EXAMPLES_DIR / "outputs" / f"{SCRIPT_PATH.stem}_{RUN_TIMESTAMP}"
    ).resolve()
    FIGURES_DIR = OUTPUT_DIR / "figures"
    LOGS_DIR = OUTPUT_DIR / "logs"
    DATA_DIR = OUTPUT_DIR / "data"
    ANIMATIONS_DIR = OUTPUT_DIR / "animations"
    SAVED_OUTPUTS_DIR = args.saved_outputs_dir.resolve()
    SAVE_ANIMATIONS = args.save_animations

    for directory in (OUTPUT_DIR, FIGURES_DIR, LOGS_DIR, DATA_DIR, ANIMATIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    LOG_FILE = LOGS_DIR / "run.log"
    _RUN_LOG_STREAM = LOG_FILE.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _RUN_LOG_STREAM)
    sys.stderr = _Tee(sys.__stderr__, _RUN_LOG_STREAM)
    atexit.register(_RUN_LOG_STREAM.close)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.__stdout__),
        ],
        force=True,
    )
    sys.excepthook = _log_uncaught_exception

    run_details = {
        "script": str(SCRIPT_PATH),
        "timestamp": RUN_TIMESTAMP,
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "repo_root": str(REPO_ROOT),
        "examples_dir": str(EXAMPLES_DIR),
        "saved_outputs_dir": str(SAVED_OUTPUTS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "figures_dir": str(FIGURES_DIR),
        "logs_dir": str(LOGS_DIR),
        "data_dir": str(DATA_DIR),
        "animations_dir": str(ANIMATIONS_DIR),
    }
    (LOGS_DIR / "run_details.json").write_text(
        _json.dumps(run_details, indent=2), encoding="utf-8"
    )

    plt.savefig = _logged_plt_savefig
    _mpl_figure.Figure.savefig = _logged_figure_savefig
    plt.show = _logged_show
    os.chdir(EXAMPLES_DIR)

    logging.info("Starting %s", SCRIPT_PATH.name)
    logging.info("Output directory: %s", OUTPUT_DIR)
    logging.info("Saved outputs directory: %s", SAVED_OUTPUTS_DIR)


def load_version(version_name: str, path: Path | None = None) -> dict:
    """Load every serialized experiment from a saved output version."""
    path = SAVED_OUTPUTS_DIR if path is None else path
    path = Path(path) / version_name
    result = {}
    for experiment in path.iterdir():
        model_id = experiment.stem.split("_")[1]
        with open(experiment, "rb") as f:
            result[model_id] = pickle.load(f)
    return result


def load_experiment(model_id: int, path: Path | None = None) -> tuple:
    """Load a serialized model, configuration, metrics, and run record."""
    path = SAVED_OUTPUTS_DIR if path is None else path
    with open(Path(path) / f"experiment_{model_id}.pickle", "rb") as f:
        model, config, metrics, run = pickle.load(f)
    return model, config, metrics, run


def _run_analysis() -> None:
    """Run the Burgers analyses and generate their figures in order."""

    def altair2pdf(plot, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        svg_path = path.with_suffix(".svg")
        plot.save(str(svg_path), engine="vl-convert")
        drawing = svg2rlg(str(svg_path))
        renderPDF.drawToFile(drawing, str(path))
        svg_path.unlink()
        logging.info("Saved Altair figure: %s", path)

    models = load_version("v14_compressibility")

    # ## Compression Experiment

    target_metric = {
        (0, 0): ("test_sine_short", "Test on Harmonic ICs (interpolation)"),
        (0, 1): ("test_sine", "Test on Harmonic ICs (extrapolation)"),
        # (1, 0): ("test_normal_short",  "Test on Gaussian ICs (interpolation)"),
        # (1, 1): ("test_normal", "Test on Gaussian ICs (extrapolation)"),
        # (2, 0): ("test_bumps_short",  "Test on Bump ICs (interpolation)"),
        # (2, 1): ("test_bumps", "Test on Bump ICs (extrapolation)"),
        # (3, 0): ("test_prediction_short", "Test on All ICs (interpolation)"),
        # (3, 1): ("test_prediction", "Test on All ICs (extrapolation)"),
    }
    nrows = len(target_metric) // 2
    fig = plt.figure(figsize=(13, 5 * nrows))
    grid = plt.GridSpec(nrows=nrows, ncols=3, top=0.7)
    models = load_version("v14_compressibility")
    pikn_spectra = {}
    for s in range(len(target_metric) // 2):
        for j in range(2):
            ax = fig.add_subplot(grid[s, j])
            metric, title = target_metric[(s, j)]
            # loss_function = torch.nn.MSELoss()
            style = {
                "PINODE Hybrid": "-o",
                "PINODE Data-Driven": "-o",
                "PIKN Hybrid": "--D",
                "PIKN Data-Driven": "--D",
                "DMD": "-X",
            }
            color = {
                "PINODE Hybrid": "blue",
                "PINODE Data-Driven": "green",
                "PIKN Hybrid": "blue",
                "PIKN Data-Driven": "green",
                "DMD": "orange",
            }

            def default_entry():
                return {"n_latent": [], "values": []}

            losses = defaultdict(default_entry)
            with torch.no_grad():
                for model_id, (model, config, metrics, run) in models.items():
                    if not config["include_physics_informed"]:
                        continue
                    label = config["job_name"].upper() + (
                        " Hybrid"
                        if config["include_physics_informed"]
                        else " Data-Driven"
                    )
                    if j == 0 and "pikn" in config["job_name"]:
                        latent_dynamics_matrix = (
                            model.latent_dynamics.get_parameter("0.weight")
                            .detach()
                            .numpy()
                        )
                        pikn_spectrum = np.linalg.eig(latent_dynamics_matrix)[0]
                        pikn_spectra[config["n_latent"]] = pikn_spectrum
                    losses[label]["n_latent"].append(config["n_latent"])
                    losses[label]["values"].append(metrics[metric]["values"][-1])
                    if config["n_latent"] not in losses["DMD"]["n_latent"]:
                        losses["DMD"]["n_latent"].append(int(config["n_latent"]))
                        losses["DMD"]["values"].append(
                            float(
                                run["info"]["dmd_rmse" + ("_short" if j == 0 else "")]
                            )
                        )

            for i, (label, data) in enumerate(losses.items()):
                plot_data = sorted(zip(data["n_latent"], data["values"]))
                n_latents = [a[0] for a in plot_data]
                values = [a[1] for a in plot_data]
                ax.loglog(
                    n_latents,
                    values,
                    style[label],
                    c=color[label],
                    label=label,
                    markersize=8,
                )
            if s == 0 and j == 0:
                ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.4, 1.25))
            ax.set_xlabel("Dimension of the latent space", fontsize=12)
            if j == 0:
                ax.set_ylabel("MSE on Test Data", fontsize=12)
            else:
                pass
                # ax.set_yticklabels([])
            ax.set_xticks(n_latents, n_latents, minor=False)
            ax.set_title(title)
            ax.grid(visible=True, which="major")
            ax.set_ylim((1e-5, 5e-1))

    ax = fig.add_subplot(grid[0, 2])
    ax.add_patch(
        plt.Rectangle(
            (0, -2),
            2,
            4,
            edgecolor="r",
            facecolor="pink",
            linewidth=1,
            alpha=0.2,
            hatch="x",
        )
    )
    for n_latent, eigs in pikn_spectra.items():
        ax.scatter(eigs.real, eigs.imag, label=n_latent, marker="x", alpha=0.8)
    ax.text(0.2, 1.5, "Instability Region", c="r")
    ax.legend()
    ax.set_xlim((-1.8, 1.8))
    ax.set_ylim((-1.8, 1.8))
    ax.set_title("Eigenvalues of PIKN models")
    ax.grid()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "compressibility.pdf", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.add_patch(
        plt.Rectangle(
            (0, -2),
            2,
            4,
            edgecolor="r",
            facecolor="pink",
            linewidth=1,
            alpha=0.2,
            hatch="x",
        )
    )
    for n_latent, eigs in pikn_spectra.items():
        p = ax.scatter(eigs.real, eigs.imag, label=n_latent, marker="x", alpha=0.8)
        max_real_idx = np.argmax(eigs.real)
        ax.scatter(
            eigs.real[max_real_idx],
            eigs.imag[max_real_idx],
            marker="X",
            c=p.get_facecolors()[0],
        )
    ax.grid()
    ax.text(0.5, 1.5, "Instability Region", c="r")
    ax.set_xlim((-1.8, 1.8))
    ax.set_ylim((-1.8, 1.8))
    ax.yaxis.tick_right()
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.legend()
    plt.show()

    # ## PINODE Data vs Collocations

    # ### 1) Movie for two examples

    # model_ids = {"PINODE DD": 1221, "PINODE Hybrid": 1225}
    model_ids = {"PINODE DD": 1239, "PINODE Hybrid": 1244}
    # "PIKN PI+DD": 881 , "PIKN DD": 870}
    models = {name: load_experiment(idx) for name, idx in model_ids.items()}

    config = models["PINODE Hybrid"][1]
    dt = config["dt"]
    n_steps_train = config["n_steps_train"]
    n_steps_test = config["n_steps_test"] * 3
    t_train = np.linspace(0, dt * (n_steps_train - 1), n_steps_train)
    t_test = np.linspace(0, 2 * dt * (n_steps_test - 1), 2 * n_steps_test)
    xs, u, u_x, u_xx, u_snapshots_train, u_snapshots_test = generate_data(
        **config, t_train=t_train, t_test=t_test, with_progress_bar=True
    )

    with torch.no_grad():
        predictions = [
            model.predict(u_snapshots_test[:, 0, :], torch.from_numpy(t_test))
            for _, (model, _, _, _) in models.items()
        ]

    fig, axs = plt.subplots(
        nrows=1, ncols=len(models), figsize=(2 + len(models) * 5, 5)
    )

    lines = []
    for ax, (label, _) in zip(axs, models.items()):
        ax.set_xlim((-np.pi, np.pi))
        ax.set_ylim((-2, 2))
        line_true = ax.plot([], [], c="g", lw=2, label="True")[0]
        line_model = ax.plot([], [], lw=2, c="r", label=label)[0]
        ax.legend()
        lines.append((line_true, line_model))

    def init():
        output = []
        for line_true, line_model in lines:
            line_true.set_data([], [])
            line_model.set_data([], [])
            output += [line_true, line_model]
        return output

    def animate(i, idx=-3):
        output = []
        for (line_true, line_model), prediction in zip(lines, predictions):
            line_true.set_data(xs, u_snapshots_test[idx][i])
            line_model.set_data(xs, prediction[idx][i])
            output += [line_true, line_model]
        return output

    anim = FuncAnimation(fig, animate, frames=len(t_test), interval=50)

    if SAVE_ANIMATIONS:
        animation_path = ANIMATIONS_DIR / "burgers_pinode_prediction.mp4"
        anim.save(animation_path, writer="ffmpeg", fps=10)
        logging.info("Saved animation: %s", animation_path)
    plt.close(fig)

    # snapshot_idx = -3
    # snapshot_idx = -9
    # snapshot_idx = -16
    # snapshot_idx = -19
    snapshot_idx = -30
    # snapshot_idx = 9

    separator_color = "r"

    def plot_3d(u, x, t, ax, label=None, zlim=(0, 0.8), title=None, elev=30, azim=45):
        X, T = np.meshgrid(x, t)
        surf = ax.plot_surface(
            X, T, u, cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False, label=label
        )
        ax.set_xlabel("Space", fontsize=12)
        ax.set_ylabel("Time", fontsize=12)
        ax.set_zlabel("Solution", fontsize=15)
        if zlim is not None:
            ax.set_zlim(zlim)
        if title is not None:
            ax.set_title(title, fontsize=16)
        ax.view_init(elev, azim)
        return surf

    def plot_2d(
        u,
        x,
        t,
        ax,
        label=None,
        title=None,
        zlim=(0, 3),
        fig=None,
        train_test_divider=True,
        **kwargs,
    ):
        extent = [t[0], t[-1], x[0], x[-1]]
        ax.set_ylabel("Space", fontsize=15)
        ax.set_xlabel("Time", fontsize=15)

        hmap = ax.imshow(
            u.T,
            extent=extent,
            norm=TwoSlopeNorm(vmin=zlim[0], vmax=zlim[1], vcenter=0.5),
            origin="upper",
            cmap="RdBu_r",
            aspect="auto",
        )
        if fig is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(hmap, cax=cax, orientation="vertical")
        if title is not None:
            ax.set_title(title, fontsize=16)
        if train_test_divider:
            ax.plot([t_train[-1], t_train[-1]], ax.get_ylim(), c=separator_color)

    #     ax.text(t_train[0], xs[0]-1, s="Interpolation", c='b')
    #     ax.text(t_test[len(t_test)//2+1], xs[0]-1, s="Extrapolation", c='b')

    dd_preds = predictions[0]
    hybrid_preds = predictions[1]

    fig = plt.figure(figsize=(20, 4))
    grid = plt.GridSpec(nrows=1, ncols=4, wspace=0.2, hspace=0.2)
    ax_true_sol = fig.add_subplot(grid[1])
    plot_2d(
        u_snapshots_test[snapshot_idx], xs, t_test, ax_true_sol, title="True Solution"
    )
    ax_hybrid_sol = fig.add_subplot(grid[2])
    plot_2d(
        hybrid_preds[snapshot_idx],
        xs,
        t_test,
        ax_hybrid_sol,
        title="Prediction of PINODE Hybrid",
    )
    ax_dd_sol = fig.add_subplot(grid[3])
    plot_2d(
        dd_preds[snapshot_idx],
        xs,
        t_test,
        ax_dd_sol,
        title="Prediction of PINODE Data-Driven",
        fig=fig,
    )

    ax_mse = fig.add_subplot(grid[0])
    mse_hybrid = ((u_snapshots_test - hybrid_preds) ** 2).mean(axis=-1)
    mse_dd = ((u_snapshots_test - dd_preds) ** 2).mean(axis=-1)
    ax_mse.semilogy(t_test, mse_hybrid.mean(axis=0), color="b", label="PINODE Hybrid")
    ax_mse.semilogy(t_test, mse_dd.mean(axis=0), color="g", label="PINODE Data-Driven")
    ax_mse.fill_between(
        t_test,
        np.percentile(mse_hybrid, q=10, axis=0),
        np.percentile(mse_hybrid, q=90, axis=0),
        color="b",
        alpha=0.1,
    )
    ax_mse.fill_between(
        t_test,
        np.percentile(mse_dd, q=10, axis=0),
        np.percentile(mse_dd, q=90, axis=0),
        color="g",
        alpha=0.1,
    )
    ax_mse.legend()
    ax_mse.plot([t_train[-1], t_train[-1]], ax_mse.get_ylim(), c=separator_color)
    ax_mse.set_ylabel("Mean Squared Error", fontsize=14)
    ax_mse.set_xlabel("Time", fontsize=14)
    ax_mse.set_title("Prediction Error", fontsize=16)

    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "example_burgers.pdf", bbox_inches="tight")
    plt.show()

    # ## 1.1) Examples for Graphical Abstract

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    plot_2d(u_snapshots_test[snapshot_idx], xs, t_test, ax, train_test_divider=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(FIGURES_DIR / "just_burgers.pdf", bbox_inches="tight")

    plt.figure(figsize=(6, 3))
    plt.plot(xs, u[0])
    plt.plot(xs, u[101])
    plt.plot(xs, u[301])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Space", fontsize=15)
    plt.savefig(FIGURES_DIR / "just_collocations.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    plot_2d(hybrid_preds[snapshot_idx], xs, t_test, ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(FIGURES_DIR / "burgers_hybrid_abstract.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    plot_2d(dd_preds[snapshot_idx], xs, t_test, ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(FIGURES_DIR / "burgers_dd_abstract.pdf", bbox_inches="tight")

    # ### 2) Examples of collocations and trajectories for all three types of ICs

    fig = plt.figure(figsize=(3 * 5, 5))
    grid = plt.GridSpec(nrows=1, ncols=3, wspace=0.4, hspace=0.2)

    examples = {"Harmonic": 6, "Bell-Curve": 101, "Bumps": 201}
    for i, (title, snapshot_id) in enumerate(examples.items()):
        trajectory = u_snapshots_test[snapshot_id].numpy()
        ax_traj = fig.add_subplot(grid[i], projection="3d")
        plot_3d(
            trajectory,
            xs,
            t_test,
            ax_traj,
            title=f"{title}: Solution",
            label="Solution",
        )
        ax_traj.plot(
            xs, trajectory[0], zs=-1.5, c="r", zdir="y", label="Initial Condition"
        )
        # ax_traj.legend()

    plt.savefig(FIGURES_DIR / "burgers_examples_of_ics.pdf", bbox_inches="tight")
    plt.show()

    # ### 3) Data-vs-collocations performance table

    performance_table = {}
    x_plt = []
    y_plt = []
    # going to be the same data for all of the experiment within a version
    config = None
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for model_id, (model, config, metrics, run) in load_version(
            "v16_data_vs_colls_mixed"
        ).items():
            if config["n_snapshots_train_bumps"] == 8192:
                continue
            total_loss = metrics["test_prediction"]["values"][-1]
            bumps_loss = metrics["test_bumps"]["values"][-1]
            performance_table[
                (config["n_collocations_sine"], config["n_snapshots_train_bumps"])
            ] = (bumps_loss, total_loss)
            x_plt.append(config["n_collocations_sine"])
            y_plt.append(config["n_snapshots_train_bumps"])

    x_plt = sorted(set(x_plt))
    y_plt = sorted(set(y_plt))
    z_plt_short = np.zeros((len(x_plt), len(y_plt)))
    z_plt_long = np.zeros((len(x_plt), len(y_plt)))
    for i, x in enumerate(x_plt):
        for j, y in enumerate(y_plt):
            z_plt_short[i, j], z_plt_long[i, j] = performance_table.get((x, y), (0, 0))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

    for ax, z, title in zip(
        axes,
        [z_plt_short, z_plt_long],
        ["Test on Bump ICs", "Test on Bump + Gaussian + Harmonic ICs"],
    ):
        z_percentage = 100 * (z / z[0, -1])
        norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=100, vmax=z_percentage.max())
        cmap = plt.cm.get_cmap("RdYlGn_r").copy()
        cmap.set_bad("grey")
        image = ax.imshow(
            z_percentage.T, interpolation="none", aspect="auto", cmap=cmap, norm=norm
        )
        z_percentage[0, 0] = np.nan
        ax.imshow(
            z_percentage.T, interpolation="none", aspect="auto", cmap=cmap, norm=norm
        )
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(x_plt)), x_plt)
        ax.set_yticks(range(len(y_plt)), y_plt)
        ax.set_xlabel("Collocations")
        ax.set_ylabel("Bump Trajectories")
        ax.set_title(title)
        for i, x in enumerate(x_plt):
            for j, y in enumerate(y_plt):
                if i == j == 0:
                    cell_label = "N/A"
                else:
                    cell_label = f"{z_percentage[i, j]:.1f}%"
                text = ax.text(i, j, cell_label, ha="center", va="center", color="b")
    plt.savefig(FIGURES_DIR / "data_vs_collocations.pdf", bbox_inches="tight")
    plt.show()

    # ### Comparison of data vs collocaations across algorithms

    model, config, metrics, run = load_experiment(1227)

    dt = config["dt"]
    n_steps_train = config["n_steps_train"]
    n_steps_test = 2 * config["n_steps_test"]
    t_train = np.linspace(0, dt * (n_steps_train - 1), n_steps_train)
    t_test = np.linspace(0, dt * (n_steps_test - 1), n_steps_test)
    xs, u, u_x, u_xx, u_snapshots_train, u_snapshots_test = generate_data(
        **{
            **config,
            "seed": config["seed"] + 228,
            "t_train": t_train,
            "t_test": t_test,
            "n_snapshots_test_bumps": 0,
            "n_snapshots_test_normal": 0,
            "n_snapshots_test_sine": 300,
        }
    )
    # u_snapshots_test = u_snapshots_test[:100]

    batch_size, time_len, _ = u_snapshots_train.shape
    usa = (
        u_snapshots_train[:, :-1]
        .reshape(batch_size * (time_len - 1), -1)
        .cpu()
        .detach()
        .numpy()
        .tolist()
    )
    usb = (
        u_snapshots_train[:, 1:]
        .reshape(batch_size * (time_len - 1), -1)
        .cpu()
        .detach()
        .numpy()
        .tolist()
    )

    algorithms = []
    n_collocations = []
    n_steps_horizon = []
    rmse = []

    cutoffs = (2, n_steps_train, 2 * n_steps_train, n_steps_test)

    x_plt = []
    y_plt = []
    # going to be the same data for all of the experiment within a version
    config = None
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for model_name, version in {
            "PIKN Hybrid": "v16.2_pikn_data_vs_colls",
            "PINODE Hybrid": "v16_data_vs_colls_mixed",
        }.items():
            for model_id, (model, config, metrics, run) in load_version(
                version
            ).items():
                assert (
                    len(algorithms)
                    == len(n_collocations)
                    == len(rmse)
                    == len(n_steps_horizon)
                )
                if config["n_snapshots_train"] != 256:
                    continue
                u_snapshots_test_preds = model.predict(
                    u_snapshots_test[:, 0], torch.from_numpy(t_test)
                )
                for cutoff in cutoffs:
                    algorithms += [model_name]
                    n_collocations += [config["n_collocations"]]
                    n_steps_horizon += [cutoff]
                    rmse += [
                        loss_fn(
                            u_snapshots_test_preds[:, :cutoff],
                            u_snapshots_test[:, :cutoff],
                        ).item()
                    ]

                if model_name == "PINODE Hybrid":
                    _, u, u_x, u_xx, _, _ = generate_data(
                        **config, t_train=t_train, t_test=t_test
                    )

                    if len(u) > 0:
                        usa_cur = usa + u.tolist()
                        u_t = config["nu"] * u_xx - u * u_x
                        usb_cur = usb + (u + dt * u_t).tolist()
                    else:
                        usa_cur = usa
                        usb_cur = usb

                    u_snapshots_dmd = torch.from_numpy(
                        np.stack([np.array(usa_cur), np.array(usb_cur)], axis=1)
                    )
                    print(u_snapshots_dmd.shape)

                    dmd_model = DMD(r=128).fit(u_snapshots_dmd, dt)
                    dmd_preds = list()
                    for u0 in u_snapshots_test.numpy():
                        dmd_preds.append(
                            torch.from_numpy(dmd_model.predict(u0[0, :], t_test))
                        )
                    dmd_preds = torch.stack(dmd_preds, dim=0)
                    for cutoff in cutoffs:
                        algorithms += ["DMD Hybrid"]
                        n_collocations += [config["n_collocations"]]
                        n_steps_horizon += [cutoff]
                        rmse += [
                            loss_fn(
                                dmd_preds[:, :cutoff], u_snapshots_test[:, :cutoff]
                            ).item()
                        ]

    df_rmse_collocations = pd.DataFrame(
        list(zip(algorithms, n_collocations, n_steps_horizon, rmse)),
        columns=["Model", "Num Collocations", "Number of Timesteps", "RMSE"],
    )
    df_rmse_collocations.loc[
        df_rmse_collocations["Num Collocations"] == 0, "Num Collocations"
    ] = 1
    dmd_too_big = df_rmse_collocations[
        (
            (df_rmse_collocations["Number of Timesteps"] >= 20)
            & (df_rmse_collocations["Model"] == "DMD Hybrid")
        )
        | (df_rmse_collocations["Number of Timesteps"] >= 80)
        & (df_rmse_collocations["Model"] == "PIKN Hybrid")
    ].index
    df_rmse_collocations = df_rmse_collocations.drop(dmd_too_big)
    rmse_collocations_url = DATA_DIR / "rmse_collocations.json"
    df_rmse_collocations.to_json(rmse_collocations_url, orient="records")

    xticklabels = sorted(set(n_collocations))
    xticklabels = [
        1,
    ] + xticklabels[1:]

    # alt.renderers.enable('latex')

    chart = (
        alt.Chart()
        .mark_line()
        .encode(
            alt.X("Num Collocations:Q", title="Number of Collocations, log2")
            .scale(type="log", base=2)
            .axis(
                values=xticklabels,
                tickCount=len(xticklabels),
                labelExpr='datum.value == 1 ? "0" : log(datum.value)/log(2)',
            ),
            alt.Y("mean(RMSE):Q").title("RMSE for Test Prediction").axis(grid=False),
            alt.Color("Model:N")
            .scale(
                domain=["PIKN Hybrid", "PINODE Hybrid", "DMD Hybrid"],
                range=["olive", "blue", "orange"],
            )
            .legend(orient="top", titleOrient="left"),
        )
        .properties(width=200, height=200)
    )

    plot = (
        alt.layer(chart, chart.mark_circle(), data=df_rmse_collocations)
        .facet(
            column=alt.Column(
                "Number of Timesteps:O",
                title="Prediction RMSE for Test Data of Different Length",
                header=alt.Header(
                    labelFontSize=14,
                    titleFontSize=14,
                    labelExpr='datum.value + " Timesteps"',
                ),
            )
        )
        .configure_axis(
            labelFontSize=14,
            titleFontSize=14,
        )
        .configure_title(
            fontSize=14,
        )
        .configure_legend(titleFontSize=14, labelFontSize=14)
    )

    altair2pdf(plot, FIGURES_DIR / "burgers_collocations_rmse.pdf")
    plot

    eig_real = []
    eig_imag = []
    n_collocations = []

    with torch.no_grad():
        for model_name, version in {"PIKN Hybrid": "v16.2_pikn_data_vs_colls"}.items():
            for model_id, (model, config, metrics, run) in load_version(
                version
            ).items():
                # import pdb; pdb.set_trace()
                dyn = (
                    model.get_parameter("latent_dynamics.0.weight")
                    .detach()
                    .cpu()
                    .numpy()
                )
                eigvals, _ = np.linalg.eig(dyn)
                n_collocations += len(eigvals) * [config["n_collocations"]]
                eig_real += eigvals.real.tolist()
                eig_imag += eigvals.imag.tolist()

    df_eigenvalues = pd.DataFrame(
        list(zip(n_collocations, eig_real, eig_imag)),
        columns=["n_collocations", "real", "imag"],
    )

    eigenvalue_plot = (
        alt.Chart(df_eigenvalues)
        .mark_circle()
        .encode(x="real", y="imag", column="n_collocations")
    )
    altair2pdf(eigenvalue_plot, FIGURES_DIR / "burgers_eigenvalues.pdf")

    # ### 4) Performance depending on noise

    noise_levels_hybrid = []
    noise_levels_dd = []
    hybrid_predictions = []
    data_driven_predictions = []
    with torch.no_grad():
        for model_id, (model, config, metrics, run) in load_version(
            "v17_noise"
        ).items():
            if config["include_physics_informed"]:
                noise_levels_hybrid.append(config["noise"])
                hybrid_predictions.append(metrics["test_prediction"]["values"][-1])
            else:
                noise_levels_dd.append(config["noise"])
                data_driven_predictions.append(metrics["test_prediction"]["values"][-1])

    noise_levels_hybrid, hybrid_predictions = [
        list(tup) for tup in zip(*sorted(zip(noise_levels_hybrid, hybrid_predictions)))
    ]
    noise_levels_dd, data_driven_predictions = [
        list(tup) for tup in zip(*sorted(zip(noise_levels_dd, data_driven_predictions)))
    ]

    dd_data = sorted(list(zip()))
    plt.figure()
    plt.loglog(noise_levels_hybrid, hybrid_predictions, "-ob", label="Hybrid")
    plt.loglog(noise_levels_dd, data_driven_predictions, "-og", label="Data-Driven")
    plt.xlabel("Noise STD")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    noise_levels_hybrid = []
    noise_levels_dd = []
    hybrid_predictions = []
    data_driven_predictions = []
    test_metric = "test_relative_to_dmd"
    with torch.no_grad():
        for model_id, (model, config, metrics, run) in load_version(
            "v18_noise_low_data"
        ).items():
            if config["include_physics_informed"] and config["include_data_driven"]:
                noise_levels_hybrid.append(config["noise"])
                hybrid_predictions.append(metrics[test_metric]["values"][-1])
            elif config["include_data_driven"]:
                noise_levels_dd.append(config["noise"])
                data_driven_predictions.append(metrics[test_metric]["values"][-1])
    #         else:
    #             noise_level_pi.append(config['noise'])
    #             pi_predictions.append(metrics[test_metric]['values'][-1])
    #             dmd_pi.append(run['info']['dmd_rmse'])

    # Use the no-noise v16 data to stay consistent with the previous experiment.
    _, _, _, run = load_experiment(1213)
    dmd_loss_from_no_noise_data = run["info"]["dmd_rmse"]
    _, _, metrics_pi, _ = load_experiment(1208)
    pi_loss_from_no_noise_data = (
        metrics_pi[test_metric]["values"][-1] / dmd_loss_from_no_noise_data
    )

    plt.figure()

    if len(noise_levels_hybrid) > 0:
        noise_levels_hybrid, hybrid_predictions = [
            list(tup)
            for tup in zip(*sorted(zip(noise_levels_hybrid, hybrid_predictions)))
        ]
        plt.loglog(
            noise_levels_hybrid, hybrid_predictions, "-o", c="blue", label="Hybrid"
        )

    if len(noise_levels_dd) > 0:
        noise_levels_dd, data_driven_predictions = [
            list(tup)
            for tup in zip(*sorted(zip(noise_levels_dd, data_driven_predictions)))
        ]
        plt.loglog(
            noise_levels_dd,
            data_driven_predictions,
            "-o",
            c="green",
            label="Data-Driven",
        )

    xlim = plt.xlim()
    # if len(noise_level_pi) == 1:
    #     plt.plot(xlim, [pi_loss_from_no_noise_data] * 2, "--", c="red")

    if "dmd" in test_metric:
        plt.plot(xlim, [1, 1], "--", c="black", label="DMD")
    plt.yticks(
        [0.1, 0.2, 0.5, 1, 2, 5, 10],
        ["10%", "20%", "50%", "100%", "200%", "500%", "1000%"],
    )

    plt.xlabel(r"$\sigma$, Standard Deviation of mean-zero Gaussian noise in data")
    plt.ylabel("MSE for test predictions relative to DMD")
    plt.legend()
    plt.savefig(FIGURES_DIR / "burgers_noise.pdf", bbox_inches="tight")
    plt.show()

    noise_levels_hybrid = []
    noise_levels_dd = []
    hybrid_predictions = []
    data_driven_predictions = []
    test_metric = "test_relative_to_dmd"
    with torch.no_grad():
        for model_id, (model, config, metrics, run) in [
            *load_version("v19_noise_weighted").items(),
            *load_version("v19_2_noise_weighted_square").items(),
        ]:
            if config["include_physics_informed"] and config["include_data_driven"]:
                if config["version"] == "v19_noise_weighted" and config["noise"] >= 1:
                    continue
                noise_levels_hybrid.append(config["noise"])
                hybrid_predictions.append(metrics[test_metric]["values"][-1])
            elif config["include_data_driven"]:
                noise_levels_dd.append(config["noise"])
                data_driven_predictions.append(metrics[test_metric]["values"][-1])
    #         else:
    #             noise_level_pi.append(config['noise'])
    #             pi_predictions.append(metrics[test_metric]['values'][-1])
    #             dmd_pi.append(run['info']['dmd_rmse'])

    # Use the no-noise v16 data to stay consistent with the previous experiment.
    _, _, _, run = load_experiment(1213)
    dmd_loss_from_no_noise_data = run["info"]["dmd_rmse"]
    _, _, metrics_pi, _ = load_experiment(1208)
    pi_loss_from_no_noise_data = metrics_pi[test_metric]["values"][-1] / (
        dmd_loss_from_no_noise_data if "dmd" in test_metric else 1
    )

    plt.figure(figsize=(7, 4))

    if len(noise_levels_hybrid) > 0:
        noise_levels_hybrid, hybrid_predictions = [
            list(tup)
            for tup in zip(*sorted(zip(noise_levels_hybrid, hybrid_predictions)))
        ]
        plt.loglog(
            noise_levels_hybrid, hybrid_predictions, "-o", c="blue", label="Hybrid"
        )

    if len(noise_levels_dd) > 0:
        noise_levels_dd, data_driven_predictions = [
            list(tup)
            for tup in zip(*sorted(zip(noise_levels_dd, data_driven_predictions)))
        ]
        plt.loglog(
            noise_levels_dd,
            data_driven_predictions,
            "-o",
            c="green",
            label="Data-Driven",
        )

    xlim = plt.xlim()
    # if len(noise_level_pi) == 1:
    plt.plot(
        xlim, [pi_loss_from_no_noise_data] * 2, "--", c="red", label="Physics-Informed"
    )

    if "dmd" in test_metric:
        plt.plot(xlim, [1, 1], "--", c="black", linewidth="2", label="DMD")
    plt.yticks(
        [0.1, 0.2, 0.5, 1, 2, 5, 10],
        ["10%", "20%", "50%", "100%", "200%", "500%", "1000%"],
    )

    plt.xlabel(r"$\sigma$, Standard Deviation of mean-zero Gaussian noise in data")
    plt.ylabel("MSE for test predictions relative to DMD")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.grid()
    plt.savefig(FIGURES_DIR / "burgers_noise.pdf", bbox_inches="tight")
    plt.show()

    _, config, _, _ = load_experiment(1208)

    dt = config["dt"]
    n_steps_train = config["n_steps_train"]
    n_steps_test = config["n_steps_test"]
    t_train = np.linspace(0, dt * (n_steps_train - 1), n_steps_train)
    t_test = np.linspace(0, dt * (n_steps_test - 1), n_steps_test)
    xs, u, u_x, u_xx, u_snapshots_train, u_snapshots_test = generate_data(
        **config, t_train=t_train, t_test=t_test, with_progress_bar=True
    )

    all_predictions = {}
    data_plot = pd.DataFrame(columns=["Noise", "Model", "Error"])
    with torch.no_grad():
        for model_id, (model, config, metrics, run) in [
            *load_version("v19_noise_weighted").items(),
            *load_version("v19_3_noise_weighted_square").items(),
        ]:
            if (
                config["version"] == "v19_noise_weighted"
                and config["noise"] >= 1
                and config["include_physics_informed"]
            ):
                continue

            predictions = model.predict(
                u_snapshots_test[:, 0, :], torch.from_numpy(t_test)
            )
            all_predictions[(config["job_name"], config["noise"])] = predictions
            losses = torch.mean((u_snapshots_test - predictions) ** 2, axis=-1).mean(
                axis=1
            )
            data_plot = pd.concat(
                [
                    data_plot,
                    pd.DataFrame(
                        {
                            "Noise": [config["noise"]] * config["n_snapshots_test"],
                            "Model": [config["job_name"]] * config["n_snapshots_test"],
                            "Error": losses,
                        }
                    ),
                ]
            )

    for noise in tqdm(data_plot["Noise"].unique()):
        if noise == 0:
            continue
        _, _, _, _, u_snapshots_train, _ = generate_data(
            **{**config, "noise": noise},
            t_train=t_train,
            t_test=t_test,
            with_progress_bar=False,
        )
        dmd_model = DMD(r=config["n_latent"]).fit(u_snapshots_train, dt)
        dmd_preds = list()
        for u0 in u_snapshots_test.numpy():
            dmd_preds.append(torch.from_numpy(dmd_model.predict(u0[0, :], t_test)))
        dmd_preds = torch.stack(dmd_preds, dim=0)
        all_predictions[("DMD", noise)] = dmd_preds
        losses = torch.mean((u_snapshots_test - dmd_preds) ** 2, axis=-1).mean(axis=1)
        data_plot = pd.concat(
            [
                data_plot,
                pd.DataFrame(
                    {
                        "Noise": [noise] * config["n_snapshots_test"],
                        "Model": ["DMD"] * config["n_snapshots_test"],
                        "Error": losses,
                    }
                ),
            ]
        )

    # plt.figure()
    ax = sns.catplot(
        data=data_plot,
        x="Noise",
        y="Error",
        hue="Model",
        kind="box",
        estimator="median",
        height=5,
        aspect=12 / 9,
        palette=["C0", "C2", "C1", "C3"],
    )
    ax.set(yscale="log", ylabel="Error", xlabel="Noise")
    plt.show()
    # plt.savefig(FIGURES_DIR / "burgers_noise_test.pdf", bbox_inches='tight')

    mapping = {
        "physics_informed": ("Physics-Informed", "red"),
        "data_driven": ("Data-Driven", "green"),
        "hybrid": ("Hybrid", "blue"),
        "DMD": ("DMD", "orange"),
    }

    fig = plt.figure(figsize=(5 * 3, 2 * 3))
    grid = plt.GridSpec(nrows=2, ncols=5)
    ax_plot = fig.add_subplot(grid[:2, :2])
    for model in data_plot["Model"].unique():
        current_data = data_plot[data_plot["Model"] == model]
        noise = sorted(current_data["Noise"].unique())
        medians = [
            current_data[current_data["Noise"] == s]["Error"].median() for s in noise
        ]
        label, color = mapping[model]
        if model == "physics_informed":
            ax_plot.plot(plt.xlim(), medians * 2, "--", c=color, label=label)
        else:
            ax_plot.loglog(noise, medians, "-o", label=label, c=color)
    ax_plot.legend()
    ax_plot.grid(which="major", axis="x")
    ax_plot.grid(which="minor", axis="y")
    plt.xlabel(r"$\sigma$, Standard Deviation of mean-zero Gaussian noise in data")
    plt.ylabel("Median MSE for test predictions")

    def plot_2d(
        u,
        x,
        t,
        ax,
        label=None,
        title=None,
        zlim=(0, 3),
        fig=None,
        xaxis_labels=False,
        yaxis_labels=False,
        ticks=True,
        text=False,
        **kwargs,
    ):
        extent = [t[0], t[-1], x[0], x[-1]]
        if xaxis_labels:
            ax.set_xlabel("Time")
        if yaxis_labels:
            ax.set_ylabel("Space")
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        hmap = ax.imshow(
            u.T,
            extent=extent,
            norm=TwoSlopeNorm(vmin=zlim[0], vmax=zlim[1], vcenter=0.5),
            origin="upper",
            cmap="RdBu_r",
            aspect="auto",
        )
        if fig is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(hmap, cax=cax, orientation="vertical")
        if title is not None:
            ax.set_title(title)
        ax.plot([t_train[-1], t_train[-1]], ax.get_ylim(), c=separator_color)
        if text:
            ax.text(x=0.2, y=1.9, s="Interpolation", size="small", color="red")
            ax.text(x=2.2, y=1.9, s="Extrapolation", size="small", color="red")

    ax_dd_small_noise = fig.add_subplot(grid[0, 3])
    ax_dd_large_noise = fig.add_subplot(grid[0, 4])
    ax_hy_small_noise = fig.add_subplot(grid[1, 3])
    ax_hy_large_noise = fig.add_subplot(grid[1, 4])
    ax_true = fig.add_subplot(grid[0, 2])
    ax_pi = fig.add_subplot(grid[1, 2])

    # idx_example = -2 # <- great
    idx_example = 2
    plot_2d(
        all_predictions[("hybrid", 0.001)][idx_example],
        xs,
        t_test,
        ax=ax_hy_small_noise,
        fig=None,
        ticks=False,
        title=r"Hybrid, $\sigma=10^{-3}$",
        xaxis_labels=True,
    )
    plot_2d(
        all_predictions[("hybrid", 10)][idx_example],
        xs,
        t_test,
        ax=ax_hy_large_noise,
        fig=fig,
        ticks=False,
        title=r"Hybrid, $\sigma=10$",
        xaxis_labels=True,
    )
    plot_2d(
        all_predictions[("data_driven", 0.001)][idx_example],
        xs,
        t_test,
        ax=ax_dd_small_noise,
        fig=None,
        ticks=False,
        title=r"Data-Driven, $\sigma=10^{-3}$",
    )
    plot_2d(
        all_predictions[("data_driven", 10)][idx_example],
        xs,
        t_test,
        ax=ax_dd_large_noise,
        fig=fig,
        ticks=False,
        title=r"Data-Driven, $\sigma=10$",
    )
    plot_2d(
        all_predictions[("physics_informed", 0)][idx_example],
        xs,
        t_test,
        ax=ax_pi,
        fig=None,
        ticks=False,
        title=r"Physics-Informed",
        xaxis_labels=True,
        yaxis_labels=True,
    )
    plot_2d(
        u_snapshots_test[idx_example],
        xs,
        t_test,
        ax=ax_true,
        fig=None,
        ticks=False,
        title="True Test Data",
        yaxis_labels=True,
        text=True,
    )

    # plt.tight_layout()
    plt.show()
    plt.savefig(FIGURES_DIR / "burgers_noise.pdf", bbox_inches="tight")

    # ## For graphical abstract

    mapping = {
        "physics_informed": ("Physics-Informed", "red"),
        "data_driven": ("Data-Driven", "green"),
        "hybrid": ("Hybrid", "blue"),
        "DMD": ("DMD", "orange"),
    }

    fig = plt.figure(figsize=(11, 2 * 3))
    grid = plt.GridSpec(nrows=2, ncols=2, hspace=0.3)

    ax_dd_small_noise = fig.add_subplot(grid[0, 0])
    ax_dd_large_noise = fig.add_subplot(grid[0, 1])
    ax_hy_small_noise = fig.add_subplot(grid[1, 0])
    ax_hy_large_noise = fig.add_subplot(grid[1, 1])

    # idx_example = -2 # <- great
    idx_example = 2
    plot_2d(
        all_predictions[("hybrid", 0.001)][idx_example],
        xs,
        t_test,
        ax=ax_hy_small_noise,
        fig=None,
        ticks=False,
        xaxis_labels=True,
        yaxis_labels=True,
    )
    plot_2d(
        all_predictions[("hybrid", 10)][idx_example],
        xs,
        t_test,
        ax=ax_hy_large_noise,
        fig=fig,
        ticks=False,
        xaxis_labels=True,
        yaxis_labels=True,
    )
    plot_2d(
        all_predictions[("data_driven", 0.001)][idx_example],
        xs,
        t_test,
        ax=ax_dd_small_noise,
        fig=None,
        ticks=False,
        xaxis_labels=True,
        yaxis_labels=True,
    )
    plot_2d(
        all_predictions[("data_driven", 10)][idx_example],
        xs,
        t_test,
        ax=ax_dd_large_noise,
        fig=fig,
        ticks=False,
        xaxis_labels=True,
        yaxis_labels=True,
    )

    # plt.tight_layout()
    plt.show()
    plt.savefig(FIGURES_DIR / "burgers_noise_abstract.pdf", bbox_inches="tight")

    logging.info("Completed %s successfully", SCRIPT_PATH.name)
    logging.info("Run log: %s", LOG_FILE)
    logging.info("Figures directory: %s", FIGURES_DIR)


def main(argv: list[str] | None = None) -> None:
    """Run the complete Burgers PINODE example."""
    args = _parse_args(argv)
    _configure_run(args)
    _run_analysis()


if __name__ == "__main__":
    main()

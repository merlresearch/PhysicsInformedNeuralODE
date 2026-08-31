# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the Duffing PINODE example and save logs, data, and figures.

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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as _mpl_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve()
EXAMPLES_DIR = SCRIPT_PATH.parent
REPO_ROOT = EXAMPLES_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pinode.core.dmd import DMD  # noqa: E402
from pinode.duffing.data_generation import generate_data  # noqa: E402


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


def load_experiment(model_id: int, path: Path | None = None) -> tuple:
    """Load a serialized model, configuration, metrics, and run record."""
    path = SAVED_OUTPUTS_DIR if path is None else path
    with open(Path(path) / f"experiment_{model_id}.pickle", "rb") as f:
        model, config, metrics, run = pickle.load(f)
    return model, config, metrics, run


def _run_analysis() -> None:
    """Run the Duffing analyses and generate their figures in order."""

    # ## PIKN/PINODE with Physics Loss

    # model
    pidd_model_id = 826
    pidd_model, pidd_config, _, _ = load_experiment(pidd_model_id)
    config = pidd_config

    # data
    dt = config["dt"]
    n_steps_train = config["n_steps_train"]
    n_steps_test = config["n_steps_test"]
    t_train = np.linspace(0, dt * (n_steps_train - 1), n_steps_train)
    t_test = np.linspace(0, dt * (n_steps_test - 1), n_steps_test)
    snapshots, collocations, autoencoder = generate_data(
        **config,
        t_train=t_train,
        t_test=t_test,
        with_progress_bar=True,
        return_true_autoencoder=True,
    )
    u_snapshots_train, u_snapshots_test = snapshots
    u_collocations, u_rhs_collocations = collocations
    true_encoder, true_decoder = autoencoder

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    n_traj_plot = 100
    n_collocations_plot = 300
    arrow_length = 0.08
    z_snapshots_train = true_encoder(u_snapshots_train)

    for snap in z_snapshots_train[:n_traj_plot]:
        ax.plot(snap[:, 0], snap[:, 1], c="r")
    ax.scatter(
        z_snapshots_train[:n_traj_plot, -1, 0],
        z_snapshots_train[:n_traj_plot, -1, 1],
        c="r",
        marker="o",
    )

    z_collocations = true_encoder(u_collocations)
    z_rhs_collocations = (
        pidd_model.project_collocations_to_latent(
            u_collocations, u_rhs_collocations, encoder=true_encoder
        )
        .detach()
        .numpy()
    )

    for z, dz in zip(
        z_collocations[:n_collocations_plot], z_rhs_collocations[:n_collocations_plot]
    ):
        ax.arrow(
            z[0],
            z[1],
            color="g",
            dx=arrow_length * dz[0],
            dy=arrow_length * dz[1],
            width=0.003,
            head_width=0.03,
        )

    for z, dz in zip(
        z_collocations[-n_collocations_plot:], z_rhs_collocations[-n_collocations_plot:]
    ):
        ax.arrow(
            z[0],
            z[1],
            color="b",
            dx=arrow_length * dz[0],
            dy=arrow_length * dz[1],
            width=0.003,
            head_width=0.03,
        )
    # ax.set_title("Train data")
    fig.savefig(FIGURES_DIR / "duff_colloc_train.pdf")
    fig.show()

    fig = plt.figure(figsize=(6, 6))
    ax2 = fig.add_subplot()
    z_snapshots_test = true_encoder(u_snapshots_test)
    for i, snap in enumerate(z_snapshots_test):
        c = (
            "r"
            if i < config["n_snapshots_test_left"]
            else (
                "g"
                if config["n_snapshots_test_left"]
                <= i
                < config["n_snapshots_test_left"] + config["n_snapshots_test_right"]
                else "b"
            )
        )
        ax2.plot(snap[:, 0], snap[:, 1], c=c)
        ax2.scatter(snap[-1, 0], snap[-1, 1], c=c)
    # ax2.set_title("Test data")
    fig.savefig(FIGURES_DIR / "duff_full_train.pdf")
    fig.show()

    # ### 1) Performance of Hybrid vs Data-Driven in time

    models_ids = {
        "PINODE Hybrid": 1300,
        "PINODE Data-Driven": 1298,
        "PINODE Physics-Informed": 1299,
    }
    models = {title: load_experiment(idx) for title, idx in models_ids.items()}
    config = models["PINODE Hybrid"][1]
    # data
    num_periods_test = 5
    dt = config["dt"]
    n_steps_train = config["n_steps_train"]
    n_steps_test = config["n_steps_train"] * num_periods_test
    t_train = np.linspace(0, dt * (n_steps_train - 1), n_steps_train)
    t_test = np.linspace(0, dt * (n_steps_test - 1), n_steps_test)
    snapshots, collocations, autoencoder = generate_data(
        **config,
        t_train=t_train,
        t_test=t_test,
        with_progress_bar=True,
        return_true_autoencoder=True,
    )
    u_snapshots_train, u_snapshots_test = snapshots
    u_collocations, u_rhs_collocations = collocations
    true_encoder, true_decoder = autoencoder

    data_plot = pd.DataFrame(columns=["Period", "Model", "Error"])

    for j, (label, (model, config, metrics, run)) in tqdm(enumerate(models.items())):
        with torch.no_grad():
            preds = model.predict(u_snapshots_test[:, 0, :], torch.from_numpy(t_test))
            z_snaps = true_encoder(preds).numpy()
            z_true = true_encoder(u_snapshots_test).numpy()
            losses = np.sum((z_snaps - z_true) ** 2, axis=-1).mean(axis=0)

        fig, axes = plt.subplots(
            nrows=1, ncols=num_periods_test, figsize=(num_periods_test * 3 + 2, 3)
        )
        for i, ax in zip(range(num_periods_test), axes):
            for j in list(range(5)) + list(range(100, 105)) + list(range(200, 205)):
                snap = z_snaps[j]
                c = (
                    "r"
                    if j < config["n_snapshots_test_left"]
                    else (
                        "g"
                        if config["n_snapshots_test_left"]
                        <= j
                        < config["n_snapshots_test_left"]
                        + config["n_snapshots_test_right"]
                        else "b"
                    )
                )
                ax.plot(
                    snap[i * n_steps_train : (i + 1) * n_steps_train, 0],
                    snap[i * n_steps_train : (i + 1) * n_steps_train, 1],
                    c=c,
                )
                ax.scatter(
                    snap[(i + 1) * n_steps_train - 1, 0],
                    snap[(i + 1) * n_steps_train - 1, 1],
                    c=c,
                )
            ax.set_title(f"period {i}", fontsize=14)

            data_plot = pd.concat(
                [
                    data_plot,
                    pd.DataFrame(
                        {
                            "Period": [f'{i+1 if i > 0 else ""}T'] * n_steps_train,
                            "Model": [label] * n_steps_train,
                            "Error": losses[
                                i * n_steps_train : (i + 1) * n_steps_train
                            ],
                        }
                    ),
                ]
            )
        plt.savefig(FIGURES_DIR / f"duffing_periods_{label}.pdf")
        plt.show()

    ax = sns.catplot(
        data=data_plot,
        x="Period",
        y="Error",
        legend=False,
        hue="Model",
        kind="box",
        height=5,
        aspect=12 / 9,
        palette=["C0", "C2", "C1"],
    )
    ax.set(yscale="log")
    plt.xlabel("Test Time (T is train length)", fontsize=14)
    plt.ylabel("MSE for the time-period", fontsize=14)
    plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1))
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.savefig(FIGURES_DIR / "duffing_periods.pdf", bbox_inches="tight")

    # ### 2) Comparison of various ROMs

    # models_ids = {"PINODE": 883, "PINODE Data-Driven": 878, "PIKN": 873}
    # "PIKN PI+DD": 881 , "PIKN DD": 870}

    models_ids = {"PIKN": 873, "PINODE": 883}

    # last approverd
    # models_ids = {'NODE': 749, "KN": 736, "PINODE": 883}
    models = {title: load_experiment(idx) for title, idx in models_ids.items()}
    config = models["PINODE"][1]
    dt = config["dt"]
    n_steps_train = config["n_steps_train"]
    n_steps_test = config["n_steps_test"]
    t_train = np.linspace(0, dt * (n_steps_train - 1), n_steps_train)
    t_test = np.linspace(0, dt * (n_steps_test - 1), n_steps_test)
    snapshots, collocations, autoencoder = generate_data(
        **config,
        t_train=t_train,
        t_test=t_test,
        with_progress_bar=True,
        return_true_autoencoder=True,
    )
    u_snapshots_train, u_snapshots_test = snapshots
    u_collocations, u_rhs_collocations = collocations
    true_encoder, true_decoder = autoencoder

    loss_fn = torch.nn.MSELoss()

    fig2 = plt.figure(figsize=(5 * 3, 4 * 1))
    grid2 = plt.GridSpec(ncols=3, nrows=1)

    ax_dmd = fig2.add_subplot(grid2[0])
    ax_pikn = fig2.add_subplot(grid2[1])
    ax_pinode = fig2.add_subplot(grid2[2])

    axes = {"PIKN": ax_pikn, "PINODE": ax_pinode}

    z_snapshots_test = true_encoder(u_snapshots_test)

    # DMD
    # forming a DMD model
    dmd_model = DMD(r=16).fit(u_snapshots_test, config["dt"])
    dmd_preds = list()
    for u0 in u_snapshots_test.numpy():
        dmd_preds.append(torch.from_numpy(dmd_model.predict(u0[0, :], t_test)))
    dmd_preds = torch.stack(dmd_preds, dim=0)
    z_dmd_preds = true_encoder(dmd_preds)

    # plotting DMD
    for i, z_dmd in enumerate(z_dmd_preds):
        c = (
            "r"
            if i < config["n_snapshots_test_left"]
            else (
                "g"
                if config["n_snapshots_test_left"]
                <= i
                < config["n_snapshots_test_left"] + config["n_snapshots_test_right"]
                else "b"
            )
        )
        ax_dmd.plot(z_dmd[: len(t_train) + 1, 0], z_dmd[: len(t_train) + 1, 1], c=c)
        ax_dmd.scatter(z_dmd[len(t_train), 0], z_dmd[len(t_train), 1], marker="o", c=c)
        ax_dmd.plot(z_dmd[:, 0], z_dmd[:, 1], c=c)
        ax_dmd.scatter(z_dmd[-1, 0], z_dmd[-1, 1], marker="o", c=c)

    ax_dmd.set_title("DMD [Latent Dimension = 16]", fontsize=20)

    # Other models
    for j, (label, (model, config, metrics, run)) in enumerate(models.items()):
        ax = axes[label]
        with torch.no_grad():
            preds = model.predict(u_snapshots_test[:, 0, :], torch.from_numpy(t_test))
            z_snaps = true_encoder(preds).numpy()
            z_true = true_encoder(u_snapshots_test).numpy()
            losses = np.sum((z_snaps - z_true) ** 2, axis=-1)

        ax.set_title(f"{label} [Latent Dimension = {config['n_latent']}]", fontsize=20)
        for i, (z_snap, loss) in enumerate(zip(z_snaps, losses)):
            c = (
                "r"
                if i < config["n_snapshots_test_left"]
                else (
                    "g"
                    if config["n_snapshots_test_left"]
                    <= i
                    < config["n_snapshots_test_left"] + config["n_snapshots_test_right"]
                    else "b"
                )
            )
            ax.plot(z_snap[:, 0], z_snap[:, 1], c=c)
            ax.scatter(z_snap[-1, 0], z_snap[-1, 1], marker="o", c=c)

    for ax in [ax_dmd, ax_pikn, ax_pinode]:
        ax.set_xlim((-2, 2))
        ax.set_ylim((-1.2, 1.2))

    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "duffing_comparison.pdf", bbox_inches="tight")
    fig2.show()

    # ### 2) PINODE Hybrid vs Data-Driven on Red Lobe

    models_ids = {
        "PINODE Hybrid": 883,
        "PINODE Data-Driven": 878,
    }  # , "PIKN Data-Driven":871}
    # "PIKN PI+DD": 881 , "PIKN DD": 870}
    models = {title: load_experiment(idx) for title, idx in models_ids.items()}

    loss_fn = torch.nn.MSELoss()

    config = models["PINODE Hybrid"][1]
    dt = config["dt"]
    n_steps_train = config["n_steps_train"]
    n_steps_test = config["n_steps_test"]
    t_train = np.linspace(0, dt * (n_steps_train - 1), n_steps_train)
    t_test = np.linspace(0, dt * (n_steps_test - 1), n_steps_test)
    snapshots, collocations, autoencoder = generate_data(
        **config,
        t_train=t_train,
        t_test=t_test,
        with_progress_bar=True,
        return_true_autoencoder=True,
    )
    u_snapshots_train, u_snapshots_test = snapshots
    u_collocations, u_rhs_collocations = collocations
    true_encoder, true_decoder = autoencoder

    fig_partial_data = plt.figure(figsize=(6, 6))
    ax_train_dd = fig_partial_data.add_subplot()

    fig_pd_collocs = plt.figure(figsize=(6, 6))
    ax_train_pi = fig_pd_collocs.add_subplot()

    # ax_true_test = fig.add_subplot(grid[0, 1])
    # ax_pikn_dd = fig.add_subplot(grid[1, 2])

    fig_pd_result = plt.figure(figsize=(6, 6))
    ax_pinode_dd = fig_pd_result.add_subplot()

    fig_collocs_result = plt.figure(figsize=(6, 6))
    ax_pinode_pi = fig_collocs_result.add_subplot()
    # ax_rmse = fig.add_subplot(grid[0, 3])
    # ax_energy = fig.add_subplot(grid[1, 3])

    axes = {
        "PINODE Hybrid": ax_pinode_pi,
        "PINODE Data-Driven": ax_pinode_dd,
    }

    z_snapshots_test = true_encoder(u_snapshots_test)

    # Train data plot
    n_traj_plot = 100
    n_collocations_plot = 300
    arrow_length = 0.08
    z_snapshots_train = true_encoder(u_snapshots_train)
    for snap in z_snapshots_train[:n_traj_plot]:
        ax_train_dd.plot(snap[:, 0], snap[:, 1], c="r", label="Trajectories")
        ax_train_pi.plot(snap[:, 0], snap[:, 1], c="r", label="Trajectories")
    ax_train_dd.scatter(
        z_snapshots_train[:n_traj_plot, -1, 0],
        z_snapshots_train[:n_traj_plot, -1, 1],
        c="r",
        marker="o",
    )
    ax_train_pi.scatter(
        z_snapshots_train[:n_traj_plot, -1, 0],
        z_snapshots_train[:n_traj_plot, -1, 1],
        c="r",
        marker="o",
    )
    # ax_train_dd.set_visible(False)
    # ax_train_pi.set_visible(False)
    # ax_train_dd.set_title("Trajectories (Train)")

    # collocations plot
    z_collocations = true_encoder(u_collocations)
    z_rhs_collocations = (
        models["PINODE Hybrid"][0]
        .project_collocations_to_latent(
            u_collocations, u_rhs_collocations, encoder=true_encoder
        )
        .detach()
        .numpy()
    )
    for z, dz in zip(
        z_collocations[:n_collocations_plot], z_rhs_collocations[:n_collocations_plot]
    ):
        ax_train_pi.arrow(
            z[0],
            z[1],
            color="g",
            dx=arrow_length * dz[0],
            dy=arrow_length * dz[1],
            width=0.003,
            head_width=0.03,
            label="Collocations",
        )

    for z, dz in zip(
        z_collocations[-n_collocations_plot:], z_rhs_collocations[-n_collocations_plot:]
    ):
        ax_train_pi.arrow(
            z[0],
            z[1],
            color="b",
            dx=arrow_length * dz[0],
            dy=arrow_length * dz[1],
            width=0.003,
            head_width=0.03,
            label="Collocations",
        )
    # ax_train_pi.set_title("Trajectories + Collocations (Train)")

    # Other models
    for j, (label, (model, config, metrics, run)) in enumerate(models.items()):
        preds = (
            model.predict(u_snapshots_test[:, 0, :], torch.from_numpy(t_test))
            .to("cpu")
            .detach()
        )
        ax = axes[label]
        # ax.set_title(f"{label} [{config['n_latent']}]")
        for i, z_snap in enumerate(true_encoder(preds)):
            c = (
                "r"
                if i < config["n_snapshots_test_left"]
                else (
                    "g"
                    if config["n_snapshots_test_left"]
                    <= i
                    < config["n_snapshots_test_left"] + config["n_snapshots_test_right"]
                    else "b"
                )
            )
            ax.plot(z_snap[:, 0], z_snap[:, 1], c=c)
            ax.scatter(z_snap[-1, 0], z_snap[-1, 1], marker="o", c=c)

    for ax in [ax_train_dd, ax_train_pi, ax_pinode_dd, ax_pinode_pi]:
        ax.set_xlim((-1.7, 1.7))
        ax.set_ylim((-1.2, 1.2))
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    fig_partial_data.savefig(FIGURES_DIR / "duff_red_lobe.pdf", bbox_inches="tight")
    fig_pd_collocs.savefig(
        FIGURES_DIR / "duff_red_lobe_with_collocs.pdf", bbox_inches="tight"
    )
    fig_pd_result.savefig(FIGURES_DIR / "duff_dd_result.pdf", bbox_inches="tight")
    fig_collocs_result.savefig(
        FIGURES_DIR / "duff_hybrid_result.pdf", bbox_inches="tight"
    )

    logging.info("Completed %s successfully", SCRIPT_PATH.name)
    logging.info("Run log: %s", LOG_FILE)
    logging.info("Figures directory: %s", FIGURES_DIR)


def main(argv: list[str] | None = None) -> None:
    """Run the complete Duffing PINODE example."""
    args = _parse_args(argv)
    _configure_run(args)
    _run_analysis()


if __name__ == "__main__":
    main()

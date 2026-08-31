# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import sacred
import torch
from sacred.observers import FileStorageObserver, MongoObserver

module_path = os.path.abspath(Path(__file__).parent.parent.parent.resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from pinode.core.dmd import DMD
from pinode.core.helpers import batch_jacobian
from pinode.core.pinode import PINODE
from pinode.duffing.data_generation import generate_data

ex = sacred.Experiment("Hi-Dim Duffing")
repo_root = Path(__file__).resolve().parents[3]
runs_dir = Path(os.environ.get("PINODE_SACRED_RUNS_DIR", repo_root / "runs"))
ex.observers.append(FileStorageObserver.create(runs_dir / "duffing"))

mongo_url = os.environ.get("PINODE_MONGO_URL")
if mongo_url:
    ex.observers.append(
        MongoObserver(
            url=mongo_url,
            db_name=os.environ.get("PINODE_MONGO_DB", "sacred"),
        )
    )


@ex.config
def config():
    # environment
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Which device is used for pytorch backend
    job_name = "duffing_testing"  # Title for this experiment
    version = f"local_debug"  # For batching multiple experiments

    # data_generation
    n_spatial = 128  # Spacial resolution for observable space
    n_collocations_left = 20000  # Number of collocations from the left lobe
    n_collocations_right = 20000  # Number of collocations from the right lobe
    n_collocations_outer = 20000  # Number of collocations from the outer lobe
    n_collocations = n_collocations_left + n_collocations_right + n_collocations_outer
    n_snapshots_train_left = 2048  # Number of train trajectories from the left lobe
    n_snapshots_train_right = 2048  # Number of train trajectories from the right lobe
    n_snapshots_train_outer = 2048  # Number of train trajectories from the outer lobe
    n_snapshots_test_left = 10  # Number of test trajectories from the left lobe
    n_snapshots_test_right = 10  # Number of test trajectories from the right lobe
    n_snapshots_test_outer = 10  # Number of test trajectories from the outer lobe
    n_snapshots_train = (
        n_snapshots_train_left + n_snapshots_train_right + n_snapshots_train_outer
    )
    n_snapshots_test = (
        n_snapshots_test_left + n_snapshots_test_right + n_snapshots_test_outer
    )
    n_steps_train = 50  # Number of time-steps for train data
    n_steps_test = 2 * n_steps_train  # Number of time-steps for test data
    dt = 0.1  # Time-step for all trajectories

    # network_configuration
    n_layers = 2  # Number of layers for encoder and decoder
    hidden_width = 512  # Width of hidden layers of encoders/decoders, if any
    n_latent = 2  # The dimension of the latent space
    n_latent_layers = 3  # Number of layers for latent space network
    latent_hidden_width = 128  # Width of hidden layers for latent space network
    skip_connections = False  # Not used. Left for compatibility with Yuying's code.
    linear_projection = False  # Not used. Left for compatibility with Yuying's code.

    # training parameters
    loss_function = "MSE"  # Type of the loss function. Can be "MSE" or "NMSE"
    n_epochs = 3  # Number of epochs for model training.
    batch_size = 64  # Number of trajectories per batch
    batch_size_collocations = (
        batch_size * n_steps_train
    )  # Number of collocations per batch.
    n_iterations_per_epoch = int(np.ceil(n_snapshots_train / batch_size))
    n_iterations = (
        n_epochs * n_iterations_per_epoch + 1
    )  # Total number of batch iterations
    lr = 1e-4  # Learning rate
    # If the learning rate scheduler sees that the training does not make any progress for lr_scheduler_patience epochs,
    # it decreases the step-size by a factor of lr_scheduler_gamma. It stops when the lr is below lr_min.
    lr_scheduler_patience = int(0.1 * n_epochs)  # Learning rate scheduler patience.
    lr_scheduler_gamma = 0.5  # Learning rate reduction factor (multiplicative)
    lr_min = 1e-6  # Minimum allowed learning rate

    include_physics_informed = 1  # Whether to include physics-informed loss
    include_data_driven = 1  # Whether to use data-driven loss
    collocations_reconstruction_weight = 1  # Weight for collocation reconstruction loss
    snapshots_reconstruction_weight = 1  # Weight for trajectories reconstruction loss
    snapshots_prediction_weight = 1  # Weight for prediction loss
    closure_weight = 1  # Weight for physics loss
    verbose_frequency = 1  # Evaluate+log progress each verbose_frequency epochs


@ex.automain
def run(
    n_spatial,
    n_collocations,
    n_collocations_left,
    n_collocations_right,
    n_collocations_outer,
    dt,
    n_steps_train,
    n_steps_test,
    n_snapshots_train_left,
    n_snapshots_train_right,
    n_snapshots_train_outer,
    n_snapshots_test_left,
    n_snapshots_test_right,
    n_snapshots_test_outer,
    n_layers,
    hidden_width,
    n_latent,
    n_latent_layers,
    latent_hidden_width,
    skip_connections,
    linear_projection,
    loss_function,
    device,
    lr,
    lr_scheduler_patience,
    lr_scheduler_gamma,
    lr_min,
    batch_size,
    batch_size_collocations,
    include_physics_informed,
    include_data_driven,
    closure_weight,
    collocations_reconstruction_weight,
    snapshots_prediction_weight,
    snapshots_reconstruction_weight,
    n_iterations,
    n_iterations_per_epoch,
    verbose_frequency,
    seed,
    _run,
):

    # Seed the random generator with the provided seed
    _rnd = np.random.default_rng(seed=seed)

    # Train time horizon
    t_train = torch.from_numpy(np.linspace(0, dt * (n_steps_train - 1), n_steps_train))
    # Test time horizon
    t_test = torch.from_numpy(np.linspace(0, dt * (n_steps_test - 1), n_steps_test))
    # Generate data set
    snapshots, collocations = generate_data(
        n_spatial=n_spatial,
        n_collocations_left=n_collocations_left,
        n_collocations_right=n_collocations_right,
        n_collocations_outer=n_collocations_outer,
        n_snapshots_train_left=n_snapshots_train_left,
        n_snapshots_train_right=n_snapshots_train_right,
        n_snapshots_train_outer=n_snapshots_train_outer,
        n_snapshots_test_left=n_snapshots_test_left,
        n_snapshots_test_right=n_snapshots_test_right,
        n_snapshots_test_outer=n_snapshots_test_outer,
        t_train=t_train,
        t_test=t_test,
        seed=seed,
    )

    u_snapshots_train, u_snapshots_test = snapshots
    u_collocations, u_rhs_collocations = collocations
    n_snapshots_train, n_snapshots_test = len(u_snapshots_train), len(u_snapshots_test)

    loss_fn = torch.nn.MSELoss()

    # Fit a DMD model, so we could judge the relative performance of our model to DMD
    dmd_model = DMD(r=n_latent).fit(u_snapshots_train, dt)
    dmd_predictions = list()
    for u0 in u_snapshots_test.numpy():
        dmd_predictions.append(torch.from_numpy(dmd_model.predict(u0[0, :], t_test)))
    dmd_predictions = torch.stack(dmd_predictions, dim=0)

    dmd_rmse = loss_fn(dmd_predictions, u_snapshots_test).item()
    dmd_rmse_short = loss_fn(
        dmd_predictions[:, : len(t_train), :], u_snapshots_test[:, : len(t_train), :]
    ).item()
    _run.info["dmd_rmse"] = dmd_rmse
    _run.info["dmd_rmse_short"] = dmd_rmse_short

    # Initialize a PINODE model
    net = PINODE(
        n_spatial=n_spatial,
        device=device,
        n_layers=n_layers,
        hidden_width=hidden_width,
        n_latent=n_latent,
        n_latent_layers=n_latent_layers,
        latent_hidden_width=latent_hidden_width,
        skip_connections=skip_connections,
        linear_projection=linear_projection,
    )

    u_snapshots_train = u_snapshots_train.to(device)
    u_snapshots_test = u_snapshots_test.to(device)
    u_collocations = u_collocations.to(device)
    u_rhs_collocations = u_rhs_collocations.to(device)

    # training
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=lr_scheduler_gamma,
        patience=lr_scheduler_patience,
        min_lr=lr_min,
    )
    total_loss_running_average = None
    eps_running_average = 1 - 1 / n_iterations_per_epoch

    for i in range(n_iterations):
        optimizer.zero_grad()
        total_loss = 0
        # physics-informed
        if include_physics_informed:
            physics_idxs = _rnd.choice(
                n_collocations,
                min(batch_size_collocations, n_collocations),
                replace=False,
            )
            u_physics_sampled = u_collocations[physics_idxs, :]
            u_rhs_physics_sampled = u_rhs_collocations[physics_idxs, :]

            v = net.encoder(u_physics_sampled)
            u_hat = net.decoder(v)
            physics_recon_loss = collocations_reconstruction_weight * loss_fn(
                u_hat, u_physics_sampled
            )
            total_loss += physics_recon_loss

            hv = net.latent_dynamics(v)
            dv_du = batch_jacobian(net.encoder, u_physics_sampled)
            u_rhs = u_rhs_physics_sampled.unsqueeze(axis=-1)
            dv_dt = torch.bmm(dv_du, u_rhs).squeeze(axis=-1)
            physics_linear_loss = closure_weight * loss_fn(hv, dv_dt)
            total_loss += physics_linear_loss

        # data-driven
        if include_data_driven:
            data_indxs = _rnd.choice(
                n_snapshots_train, min(batch_size, n_snapshots_train), replace=False
            )
            u_snapshots_sample = u_snapshots_train[data_indxs, :, :]
            u_snapshots_hat = net.decoder(net.encoder(u_snapshots_sample))
            u_snapshots_predictions = net.predict(u_snapshots_sample[:, 0, :], t_train)
            snapshots_trajectories_loss = snapshots_prediction_weight * loss_fn(
                u_snapshots_predictions, u_snapshots_sample
            )
            total_loss += snapshots_trajectories_loss
            snapshots_recon_loss = snapshots_reconstruction_weight * loss_fn(
                u_snapshots_hat, u_snapshots_sample
            )
            total_loss += snapshots_recon_loss

        total_loss.backward()

        loss = total_loss.item()
        if total_loss_running_average is None:
            total_loss_running_average = loss
        else:
            total_loss_running_average = (
                1 - eps_running_average
            ) + eps_running_average * loss
        if i % n_iterations_per_epoch == 0:
            scheduler.step(total_loss_running_average)

        optimizer.step()
        if i % (verbose_frequency * n_iterations_per_epoch) == 0:
            # Logging
            with torch.autograd.no_grad():
                epoch = i // n_iterations_per_epoch

                _run.log_scalar("iteration", i, step=epoch)
                _run.log_scalar("epoch", epoch, step=epoch)
                _run.log_scalar("total_loss", total_loss.item(), step=epoch)
                _run.log_scalar(
                    "total_loss_running_avg", total_loss_running_average, step=epoch
                )
                # _run.log_scalar("lr", scheduler.get_lr(), step=epoch)

                u_train_predictions = net.predict(u_snapshots_train[:, 0, :], t_train)
                _run.log_scalar(
                    "train_prediction",
                    loss_fn(u_train_predictions, u_snapshots_train).item(),
                    step=epoch,
                )
                u_test_predictions = net.predict(u_snapshots_test[:, 0, :], t_test)
                test_loss = loss_fn(u_test_predictions, u_snapshots_test).item()
                _run.log_scalar("test_prediction", test_loss, step=epoch)
                u_test_predictions_short = net.predict(
                    u_snapshots_test[:, 0, :], t_train
                )
                test_loss_short = loss_fn(
                    u_test_predictions_short, u_snapshots_test[:, : len(t_train), :]
                ).item()
                _run.log_scalar("test_prediction_short", test_loss_short, step=epoch)

                # full test
                train_loss_left = loss_fn(
                    u_train_predictions[:n_snapshots_train_left],
                    u_snapshots_train[:n_snapshots_train_left],
                )
                train_loss_right = loss_fn(
                    u_train_predictions[
                        n_snapshots_train_left : n_snapshots_train_left
                        + n_snapshots_train_right
                    ],
                    u_snapshots_train[
                        n_snapshots_train_left : n_snapshots_train_left
                        + n_snapshots_train_right
                    ],
                )
                train_loss_outer = loss_fn(
                    u_train_predictions[-n_snapshots_train_outer:],
                    u_snapshots_train[-n_snapshots_train_outer:],
                )
                _run.log_scalar("train_loss_left", train_loss_left.item(), step=epoch)
                _run.log_scalar("train_loss_right", train_loss_right.item(), step=epoch)
                _run.log_scalar("train_loss_outer", train_loss_outer.item(), step=epoch)

                test_loss_left = loss_fn(
                    u_test_predictions[:n_snapshots_test_left],
                    u_snapshots_test[:n_snapshots_test_left],
                )
                test_loss_right = loss_fn(
                    u_test_predictions[
                        n_snapshots_test_left : n_snapshots_test_left
                        + n_snapshots_test_right
                    ],
                    u_snapshots_test[
                        n_snapshots_test_left : n_snapshots_test_left
                        + n_snapshots_test_right
                    ],
                )
                test_loss_outer = loss_fn(
                    u_test_predictions[-n_snapshots_test_outer:],
                    u_snapshots_test[-n_snapshots_test_outer:],
                )
                _run.log_scalar("test_loss_left", test_loss_left.item(), step=epoch)
                _run.log_scalar("test_loss_right", test_loss_right.item(), step=epoch)
                _run.log_scalar("test_loss_outer", test_loss_outer.item(), step=epoch)

                # short test
                train_loss_left = loss_fn(
                    u_train_predictions[:n_snapshots_train_left, : len(t_train), :],
                    u_snapshots_train[:n_snapshots_train_left, : len(t_train), :],
                )
                train_loss_right = loss_fn(
                    u_train_predictions[
                        n_snapshots_train_left : n_snapshots_train_left
                        + n_snapshots_train_right,
                        : len(t_train),
                        :,
                    ],
                    u_snapshots_train[
                        n_snapshots_train_left : n_snapshots_train_left
                        + n_snapshots_train_right,
                        : len(t_train),
                        :,
                    ],
                )
                train_loss_outer = loss_fn(
                    u_train_predictions[-n_snapshots_train_outer:, : len(t_train), :],
                    u_snapshots_train[-n_snapshots_train_outer:, : len(t_train), :],
                )
                _run.log_scalar(
                    "train_loss_left_short", train_loss_left.item(), step=epoch
                )
                _run.log_scalar(
                    "train_loss_right_short", train_loss_right.item(), step=epoch
                )
                _run.log_scalar(
                    "train_loss_outer_short", train_loss_outer.item(), step=epoch
                )

                test_loss_left = loss_fn(
                    u_test_predictions[:n_snapshots_test_left, : len(t_train), :],
                    u_snapshots_test[:n_snapshots_test_left, : len(t_train), :],
                )
                test_loss_right = loss_fn(
                    u_test_predictions[
                        n_snapshots_test_left : n_snapshots_test_left
                        + n_snapshots_test_right,
                        : len(t_train),
                        :,
                    ],
                    u_snapshots_test[
                        n_snapshots_test_left : n_snapshots_test_left
                        + n_snapshots_test_right,
                        : len(t_train),
                        :,
                    ],
                )
                test_loss_outer = loss_fn(
                    u_test_predictions[-n_snapshots_test_outer:, : len(t_train), :],
                    u_snapshots_test[-n_snapshots_test_outer:, : len(t_train), :],
                )
                _run.log_scalar(
                    "test_loss_left_short", test_loss_left.item(), step=epoch
                )
                _run.log_scalar(
                    "test_loss_right_short", test_loss_right.item(), step=epoch
                )
                _run.log_scalar(
                    "test_loss_outer_short", test_loss_outer.item(), step=epoch
                )

                _run.log_scalar(
                    "test_relative_to_dmd", test_loss / dmd_rmse, step=epoch
                )
                _run.log_scalar(
                    "test_short_relative_to_dmd",
                    test_loss_short / dmd_rmse_short,
                    step=epoch,
                )

                if include_physics_informed:
                    _run.log_scalar(
                        "closure_loss", physics_linear_loss.item(), step=epoch
                    )
                    _run.log_scalar(
                        "collocations_reconstruction_loss",
                        physics_recon_loss.item(),
                        step=epoch,
                    )

                if include_data_driven:
                    _run.log_scalar(
                        "snapshots_prediction_loss",
                        snapshots_trajectories_loss.item(),
                        step=epoch,
                    )
                    _run.log_scalar(
                        "snapshots_reconstruction_loss",
                        snapshots_recon_loss.item(),
                        step=epoch,
                    )

    # Save the resulting model in a temporary file and add it to the database
    _, temp_file_path = tempfile.mkstemp()
    with open(temp_file_path, "wb") as f:
        torch.save(net.state_dict(), f)
    _run.add_artifact(temp_file_path, name="model")
    os.remove(temp_file_path)
    return loss_fn(u_test_predictions, u_snapshots_test).item()

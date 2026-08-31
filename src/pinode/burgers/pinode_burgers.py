# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

# import pdb; pdb.set_trace()

module_path = os.path.abspath(Path(__file__).parent.parent.parent.resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from pinode.burgers.data_generation import generate_data
from pinode.core.collocations import StaticProvider
from pinode.core.dmd import DMD
from pinode.core.pinode import PINODE

ex = Experiment("PINODE Burgers")
repo_root = Path(__file__).resolve().parents[3]
runs_dir = Path(os.environ.get("PINODE_SACRED_RUNS_DIR", repo_root / "runs"))
ex.observers.append(FileStorageObserver.create(runs_dir / "burgers"))

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
    job_name = "pinode_testing"  # Title for this experiment
    version = f"local_debug"  # For batching multiple experiments

    # data generation
    noise = 0.1  # Standard deviation of normal iid pixel-wide noise
    nu = 0.01  # Viscosity for Burgers data
    shift = 0  # Additive shift for all Burgers solutions
    n_spatial = 128  # Spacial resolution for Burgers solutions
    n_modes = 10  # Number of harmonic modes used for ICs
    n_bumps = 2  # Number of bumps per IC for bumps data
    k_bumps = 20  # Slope constant for sigmoid bumps
    n_collocations_sine = 100  # Number of harmonic collocations
    n_collocations_normal = 100  # Number of Gaussian (bell-curve) collocations
    n_collocations_bumps = 100  # Number of bumps collocations
    n_snapshots_train_sine = 128  # Number of trajectories with harmonic ICs in train
    n_snapshots_test_sine = 100  # Number of trajectories with harmonic ICs in test
    n_snapshots_train_normal = 0  # Number of trajectories with Gaussian ICs in train
    n_snapshots_test_normal = 100  # Number of trajectories with Gaussian ICs in test
    n_snapshots_train_bumps = 0  # Number of trajectories with bumps ICs in train
    n_snapshots_test_bumps = 100  # Number of trajectories with bumps ICs in test
    n_snapshots_train = (
        n_snapshots_train_sine + n_snapshots_train_normal + n_snapshots_train_bumps
    )
    n_snapshots_test = (
        n_snapshots_test_sine + n_snapshots_test_normal + n_snapshots_test_bumps
    )
    n_collocations = n_collocations_sine + n_collocations_normal + n_collocations_bumps
    n_steps_train = 50  # Number of time-steps for train data
    n_steps_test = 2 * n_steps_train  # Number of time-steps for test data
    dt = 0.1  # Time-step for all trajectories
    randomize_modes = True  # Whether to randomly cap frequencies for sine cols.

    # network_configuration
    # NB: For numbers of layers, we count the input and the hidden layers, but not the output layer.
    # e.g. if you want an autoencoder with one hidden layer, set n_layers=2.
    # The output layer is always a fully connected layer with no activation.
    n_layers = 3  # Number of layers for encoder and decoder
    hidden_width = 512  # Width of hidden layers of encoders/decoders, if any
    n_latent = 16  # The dimension of the latent space
    n_latent_layers = 3  # Number of layers for latent space network
    latent_hidden_width = 256  # Width of hidden layers for latent space network
    skip_connections = False  # Not used. Left for compatibility with Yuying's code.
    linear_projection = False  # Not used. Left for compatibility with Yuying's code.

    # training parameters
    loss_function = "MSE"  # Type of the loss function. Can be "MSE" or "NMSE"
    n_epochs = 3  # Number of epochs for model training.
    epoch_patience = (
        20  # Number of epochs to wait for improvement before early stopping
    )
    batch_size = 64  # Number of trajectories per batch
    batch_size_collocations = (
        batch_size * n_steps_train
    )  # Number of collocations per batch.
    n_iterations_per_epoch = max(
        int(np.ceil(n_collocations / batch_size_collocations)),
        int(np.ceil(n_snapshots_train / batch_size)),
    )
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
    snapshots_reconstruction_weight = (
        1 if noise < 1 else 1 / noise
    )  # Weight for trajectories reconstruction loss
    snapshots_prediction_weight = (
        1 if noise < 1 else 1 / noise
    )  # Weight for prediction loss
    closure_weight = 1  # Weight for physics loss
    verbose_frequency = 1  # Evaluate+log progress each verbose_frequency epochs


@ex.automain
def run(
    noise,
    nu,
    shift,
    n_spatial,
    n_modes,
    n_collocations_sine,
    n_collocations_normal,
    n_collocations_bumps,
    n_collocations,
    dt,
    n_steps_train,
    n_steps_test,
    n_snapshots_train,
    n_snapshots_test,
    n_snapshots_train_sine,
    n_snapshots_test_sine,
    randomize_modes,
    n_snapshots_train_normal,
    n_snapshots_test_normal,
    n_snapshots_train_bumps,
    n_snapshots_test_bumps,
    n_bumps,
    k_bumps,
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
    epoch_patience,
    n_iterations_per_epoch,
    verbose_frequency,
    seed,
    _run,
):

    # Seed the random generator with the provided seed
    _rnd = torch.Generator(device=device)
    _rnd.manual_seed(seed)

    # Ensure that the loss flags are consistent with the provided data
    if n_collocations == 0 and include_physics_informed == 1:
        include_physics_informed = 0
    if n_snapshots_train == 0 and include_data_driven == 1:
        include_data_driven = 0
    if n_snapshots_train == 0 and n_collocations == 0:
        raise ValueError("No data was provided.")

    # Train time horizon
    t_train = np.linspace(0, dt * (n_steps_train - 1), n_steps_train)
    # Test time horizon
    t_test = np.linspace(0, dt * (n_steps_test - 1), n_steps_test)
    # Generate data set
    # xs -- spacial grid for solutions
    # u -- collocations
    # u_x -- spacial derivatives for collocations
    # u_xx -- second spacial derivatives for collocations
    # u_snapshots_train -- snapshots for the train data
    # u_snapshots_test -- snapshots for the test data
    xs, u, u_x, u_xx, u_snapshots_train, u_snapshots_test = generate_data(
        noise=noise,
        nu=nu,
        n_spatial=n_spatial,
        n_modes=n_modes,
        n_collocations_sine=n_collocations_sine,
        n_collocations_normal=n_collocations_normal,
        n_collocations_bumps=n_collocations_bumps,
        n_snapshots_train_sine=n_snapshots_train_sine,
        n_snapshots_test_sine=n_snapshots_test_sine,
        n_snapshots_train_normal=n_snapshots_train_normal,
        n_snapshots_test_normal=n_snapshots_test_normal,
        n_snapshots_train_bumps=n_snapshots_train_bumps,
        n_snapshots_test_bumps=n_snapshots_test_bumps,
        n_bumps=n_bumps,
        k_bumps=k_bumps,
        t_train=t_train,
        t_test=t_test,
        seed=seed,
        shift=shift,
        with_progress_bar=False,
        randomize_modes=randomize_modes,
    )

    # Choose the loss function
    if loss_function == "MSE":
        # Traditional mean squared error
        loss_fn = torch.nn.MSELoss()
    elif loss_function == "NMSE":
        # Normalized mean squared error
        def loss_fn(predicted, true, eps=1e-5):
            return (
                torch.square((predicted - true)) / (torch.square(true) + eps)
            ).mean()

    else:
        raise ValueError(f'Unknown value for loss function: "{loss_function}"')

    # If we have data then fit DMD so we could monitor the model's performance relative to the DMD's performance.
    if n_snapshots_train > 0:
        dmd_model = DMD(r=n_latent).fit(u_snapshots_train, dt)
        dmd_preds = list()
        for u0 in u_snapshots_test.numpy():
            dmd_preds.append(torch.from_numpy(dmd_model.predict(u0[0, :], t_test)))
        dmd_preds = torch.stack(dmd_preds, dim=0)

        dmd_rmse = loss_fn(dmd_preds, u_snapshots_test).item()
        dmd_rmse_short = loss_fn(
            dmd_preds[:, : len(t_train), :], u_snapshots_test[:, : len(t_train), :]
        ).item()

    else:
        dmd_rmse = 1
        dmd_rmse_short = 1
    _run.info["dmd_rmse"] = dmd_rmse
    _run.info["dmd_rmse_short"] = dmd_rmse_short

    # Prepare the data for the use with pytorch
    t_train = torch.from_numpy(t_train).double().to(device)
    t_test = torch.from_numpy(t_test).double().to(device)

    if len(u) > 0:
        u = u.double().to(device)
        u_x = u_x.double().to(device)
        u_xx = u_xx.double().to(device)
        us = torch.stack([u, u_x, u_xx], dim=1)
    else:
        us = []
    if len(u_snapshots_train) > 0:
        u_snapshots_train = u_snapshots_train.double().to(device)
    if len(u_snapshots_test) > 0:
        u_snapshots_test = u_snapshots_test.double().to(device)

    # Define dynamics in the observable space
    def burgers_rhs(us, **kwargs):
        """
        u is of shape (batch_size, 1)
        u_x is of shape (batch_size, 1)
        u_xx is of shape (batch_size, 1)
        """
        u = us[:, 0]
        u_x = us[:, 1]
        u_xx = us[:, 2]

        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)
        if isinstance(u_x, np.ndarray):
            u_x = torch.from_numpy(u_x)
        if isinstance(u_xx, np.ndarray):
            u_xx = torch.from_numpy(u_xx)

        if len(u.shape) == 1:
            u = u.reshape(1, -1)
        if len(u_x.shape) == 1:
            u_x = u_x.reshape(1, -1)
        if len(u_xx.shape) == 1:
            u_xx = u_xx.reshape(1, -1)

        return nu * u_xx - u * u_x

    collocations_provider = StaticProvider(
        data=us,
        rhs=burgers_rhs,
        _rnd=_rnd,
        batch_size=batch_size_collocations,
        device=device,
    )

    # Initiate a PINODE model
    net = PINODE(
        n_spatial=n_spatial,
        device=device,
        dyn_rhs=burgers_rhs,
        n_layers=n_layers,
        hidden_width=hidden_width,
        n_latent=n_latent,
        n_latent_layers=n_latent_layers,
        latent_hidden_width=latent_hidden_width,
        skip_connections=skip_connections,
        linear_projection=linear_projection,
        autoencoder_type="fc",
    ).double()

    # training
    # loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        [*net.parameters(), *collocations_provider.parameters()], lr=lr
    )
    amp_enabled = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=lr_scheduler_gamma,
        patience=lr_scheduler_patience,
        min_lr=lr_min,
    )
    total_loss_running_average = None
    eps_running_average = 1 - 1 / (n_iterations_per_epoch / verbose_frequency)

    best_test_loss = np.inf
    previous_best_epoch = 0
    _, model_weights_path = tempfile.mkstemp()

    for i in range(n_iterations):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            total_loss = torch.zeros(size=(1,), device=device)
            # physics-informed
            if include_physics_informed:
                u_batch, f_batch = collocations_provider.get_batch()
                v, u_hat, Lv, dv_dt = net.physics_informed(u_batch, f_batch)
                physics_linear_loss = loss_fn(Lv, dv_dt)
                total_loss += closure_weight * physics_linear_loss
                physics_recon_loss = loss_fn(u_hat, u_batch)
                total_loss += collocations_reconstruction_weight * physics_recon_loss

            # data-driven
            if include_data_driven:
                bs = min(batch_size, n_snapshots_train)
                start = torch.randint(
                    low=0,
                    high=n_snapshots_train - bs,
                    size=(1,),
                    device=device,
                    generator=_rnd,
                )
                u_snapshots_sample = u_snapshots_train[start : start + bs]
                u_snapshots_hat = net.decoder(net.encoder(u_snapshots_sample))
                u_snapshots_predictions = net.predict(
                    u_snapshots_sample[:, 0, :], t_train
                )
                snapshots_trajectories_loss = loss_fn(
                    u_snapshots_predictions, u_snapshots_sample
                )
                total_loss += snapshots_prediction_weight * snapshots_trajectories_loss
                snapshots_recon_loss = loss_fn(u_snapshots_hat, u_snapshots_sample)
                total_loss += snapshots_reconstruction_weight * snapshots_recon_loss

        # backpropagation
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch = i // n_iterations_per_epoch
        if i % n_iterations_per_epoch == 0 and device.startswith("cuda"):
            _run.log_scalar(
                "gpu_memory_allocated",
                torch.cuda.memory_allocated(device=device) / 2**30,
                step=epoch,
            )
            try:
                gpu_utilization = torch.cuda.utilization(device=device)
            except (ImportError, ModuleNotFoundError):
                # Utilization telemetry requires the optional pynvml package.
                pass
            else:
                _run.log_scalar("gpu_utilization", int(gpu_utilization), step=epoch)

        if i % (verbose_frequency * n_iterations_per_epoch) == 0:
            # Performance evaluation and logging
            with torch.autograd.no_grad():
                loss = total_loss.item()
                if total_loss_running_average is None:
                    total_loss_running_average = loss
                else:
                    total_loss_running_average = (
                        1 - eps_running_average
                    ) * total_loss_running_average + eps_running_average * loss
                scheduler.step(total_loss_running_average)

                _run.log_scalar("iteration", i, step=epoch)
                _run.log_scalar("epoch", epoch, step=epoch)
                _run.log_scalar("total_loss", loss, step=epoch)
                _run.log_scalar(
                    "total_loss_running_avg", total_loss_running_average, step=epoch
                )

                # get predictions for test data
                u_test_predictions = net.predict(u_snapshots_test[:, 0, :], t_test)
                test_loss = loss_fn(u_test_predictions, u_snapshots_test).item()
                _run.log_scalar("test_prediction", test_loss, step=epoch)
                u_test_predictions_short = net.predict(
                    u_snapshots_test[:, 0, :], t_train
                )
                # everything with a suffix _short means that it's only for the train time horizon (interpolation)
                test_loss_short = loss_fn(
                    u_test_predictions_short, u_snapshots_test[:, : len(t_train), :]
                ).item()
                _run.log_scalar("test_prediction_short", test_loss_short, step=epoch)

                _run.log_scalar(
                    "test_relative_to_dmd", test_loss / dmd_rmse, step=epoch
                )
                _run.log_scalar(
                    "test_short_relative_to_dmd",
                    test_loss_short / dmd_rmse_short,
                    step=epoch,
                )

                u_test_predictions_sine = loss_fn(
                    u_test_predictions[:n_snapshots_test_sine],
                    u_snapshots_test[:n_snapshots_test_sine],
                ).item()
                _run.log_scalar("test_sine", u_test_predictions_sine, step=epoch)

                u_test_predictions_sine_short = loss_fn(
                    u_test_predictions[:n_snapshots_test_sine, : len(t_train)],
                    u_snapshots_test[:n_snapshots_test_sine, : len(t_train)],
                ).item()
                _run.log_scalar(
                    "test_sine_short", u_test_predictions_sine_short, step=epoch
                )

                u_test_predictions_normal = loss_fn(
                    u_test_predictions[
                        n_snapshots_test_sine : n_snapshots_test_sine
                        + n_snapshots_test_normal
                    ],
                    u_snapshots_test[
                        n_snapshots_test_sine : n_snapshots_test_sine
                        + n_snapshots_test_normal
                    ],
                ).item()
                _run.log_scalar("test_normal", u_test_predictions_normal, step=epoch)

                u_test_predictions_normal_short = loss_fn(
                    u_test_predictions[
                        n_snapshots_test_sine : n_snapshots_test_sine
                        + n_snapshots_test_normal,
                        : len(t_train),
                    ],
                    u_snapshots_test[
                        n_snapshots_test_sine : n_snapshots_test_sine
                        + n_snapshots_test_normal,
                        : len(t_train),
                    ],
                ).item()
                _run.log_scalar(
                    "test_normal_short", u_test_predictions_normal_short, step=epoch
                )

                u_test_predictions_bumps = loss_fn(
                    u_test_predictions[-n_snapshots_test_bumps:],
                    u_snapshots_test[-n_snapshots_test_bumps:],
                ).item()
                _run.log_scalar("test_bumps", u_test_predictions_bumps, step=epoch)

                u_test_predictions_bumps_short = loss_fn(
                    u_test_predictions[-n_snapshots_test_bumps:, : len(t_train)],
                    u_snapshots_test[-n_snapshots_test_bumps:, : len(t_train)],
                ).item()
                _run.log_scalar(
                    "test_bumps_short", u_test_predictions_bumps_short, step=epoch
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

                if test_loss_short < best_test_loss:
                    best_test_loss = test_loss_short
                    torch.save(net.state_dict(), model_weights_path)
                    previous_best_epoch = epoch
                else:
                    if epoch - previous_best_epoch > epoch_patience:
                        print("Early stopping")
                        break

    # save the model into a temporary file and add it to the database
    _run.add_artifact(model_weights_path, name="model")
    os.remove(model_weights_path)

    # save the collocation provider
    _, temp_file_path = tempfile.mkstemp()
    with open(temp_file_path, "wb") as f:
        torch.save(collocations_provider.parameters(), f)
    _run.add_artifact(temp_file_path, name="collocation_provider")
    os.remove(temp_file_path)
    return loss_fn(u_test_predictions, u_snapshots_test).item()

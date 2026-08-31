# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import integrate

module_path = os.path.abspath(Path(__file__).parent.parent.parent.resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from tqdm.notebook import tqdm

from pinode.core.helpers import batch_jacobian


def dyn_rhs(u):
    """
    Right hand-side for Duffing oscillator
    """

    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u)

    if len(u.shape) == 1:
        u = u.reshape(1, -1)

    u1 = u[..., 0:1]
    u2 = u[..., 1:2]
    return torch.cat([u2, u1 - u1**3], dim=1)


def hamiltonian(u):
    """
    Hamiltonian of Duffing oscillator
    """
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u)

    if len(u.shape) == 1:
        u = u.reshape(1, -1)

    u1 = u[..., 0:1]
    u2 = u[..., 1:2]
    return 1 / 2 * u2**2 - 1 / 2 * u1**2 + 1 / 4 * u1**4


def generate_snapshots(
    t,
    n_snapshots_left=10,
    n_snapshots_right=10,
    n_snapshots_outer=10,
    seed=42,
    with_progress_bar=False,
    **kwargs,
):
    """
    Generate trajectories for Duffing oscillator

    :param t: timeframe
    :param n_snapshots_left: number of trajectories to generate from the left lobe
    :param n_snapshots_right: number of trajectories to generate from the right lobe
    :param n_snapshots_outer: number of trajectories to generate from the outer lobe
    :param seed: random seed
    :param with_progress_bar: whether to display the progress bar (used in Jupyter notebooks)
    :param kwargs: other named arguments
    :return:
    """
    _rnd = np.random.default_rng(seed=seed)
    inits_left = []
    inits_right = []
    inits_outer = []
    # Generate points randomly until all three baskets don't fill
    while True:
        if (
            len(inits_left) == n_snapshots_left
            and len(inits_right) == n_snapshots_right
            and len(inits_outer) == n_snapshots_outer
        ):
            break
        u_init = torch.from_numpy(_rnd.uniform(size=2) * 2 - 1)
        # Decide which area the current point belongs
        if (
            u_init[0] > 0
            and hamiltonian(u_init) < 0
            and len(inits_right) < n_snapshots_right
        ):
            inits_right.append(u_init)
        elif (
            u_init[0] < 0
            and hamiltonian(u_init) < 0
            and len(inits_left) < n_snapshots_left
        ):
            inits_left.append(u_init)
        elif hamiltonian(u_init) > 0 and len(inits_outer) < n_snapshots_outer:
            inits_outer.append(u_init)
        else:
            continue
    all_inits = torch.stack(inits_left + inits_right + inits_outer, dim=0).double()
    if with_progress_bar:
        all_inits = tqdm(all_inits)
    # Generate all trajectories at once
    solutions = torch.stack(
        [
            torch.from_numpy(
                integrate.solve_ivp(
                    lambda _, x: dyn_rhs(x).squeeze(),
                    (t[0], t[-1]),
                    u_init.squeeze(),
                    t_eval=t,
                ).y.T
            )
            for u_init in all_inits
        ],
        dim=0,
    ).double()
    return solutions


def generate_collocations(
    n_collocations_left=10, n_collocations_right=10, n_collocations_outer=10, seed=42
):
    """
    Generate collocation points for Duffing oscillator

    :param n_collocations_left: number of collocations from the left lobe
    :param n_collocations_right: number of collocations from the right lobe
    :param n_collocations_outer: number of collocations from the outer lobe
    :param seed: random seed
    :return: collocation points
    """
    _rnd = np.random.default_rng(seed=seed)

    inits_left = []
    inits_right = []
    inits_outer = []
    while True:
        if (
            len(inits_left) == n_collocations_left
            and len(inits_right) == n_collocations_right
            and len(inits_outer) == n_collocations_outer
        ):
            break
        u_init = torch.from_numpy(_rnd.uniform(low=-1.5, high=1.5, size=2))
        u_init[1] /= 1.5
        if (
            u_init[0] > 0
            and hamiltonian(u_init) < 0
            and len(inits_right) < n_collocations_right
        ):
            inits_right.append(u_init)
        elif (
            u_init[0] < 0
            and hamiltonian(u_init) < 0
            and len(inits_left) < n_collocations_left
        ):
            inits_left.append(u_init)
        elif hamiltonian(u_init) > 0 and len(inits_outer) < n_collocations_outer:
            inits_outer.append(u_init)
        else:
            continue

    u = torch.stack(inits_left + inits_right + inits_outer, dim=0).double()
    u_t = dyn_rhs(u)
    return u, u_t


def generate_data(
    n_spatial=128,
    n_collocations_left=10,
    n_collocations_right=10,
    n_collocations_outer=10,
    n_snapshots_train_left=10,
    n_snapshots_train_right=10,
    n_snapshots_train_outer=10,
    n_snapshots_test_left=10,
    n_snapshots_test_right=10,
    n_snapshots_test_outer=10,
    t_train=None,
    t_test=None,
    seed=42,
    return_true_autoencoder=False,
    **kwargs,
):
    """
    Generate high-dimensional dataset (data + collocations, train + test) for Duffing experiment

    :param n_spatial: dimensions of the observable space
    :param n_collocations_left: number of collocations from the left lobe
    :param n_collocations_right: number of collocations from the right lobe
    :param n_collocations_outer: number of collocations from the outer lobe
    :param n_snapshots_train_left: number of train trajectories to generate from the left lobe
    :param n_snapshots_train_right: number of train trajectories to generate from the right lobe
    :param n_snapshots_train_outer: number of train trajectories to generate from the outer lobe
    :param n_snapshots_test_left: number of test trajectories to generate from the left lobe
    :param n_snapshots_test_right: number of test trajectories to generate from the right lobe
    :param n_snapshots_test_outer: number of test trajectories to generate from the outer lobe
    :param t_train: train timeframe
    :param t_test: test timeframe
    :param seed: random seed
    :param return_true_autoencoder: whether to return the functions that map to the true latent space and back
    :param kwargs: other keyword arguments
    :return: dataset
    """

    snapshots_train_latent = generate_snapshots(
        t_train,
        n_snapshots_train_left,
        n_snapshots_train_right,
        n_snapshots_train_outer,
        seed=seed,
        **kwargs,
    )
    snapshots_test_latent = generate_snapshots(
        t_test,
        n_snapshots_test_left,
        n_snapshots_test_right,
        n_snapshots_test_outer,
        seed=seed + 1,
        **kwargs,
    )

    _rnd = np.random.default_rng(seed=seed + 2)

    # create the matrix which defines the span of the decoder in the observable space
    encoder_span = torch.from_numpy(_rnd.normal(size=(2, n_spatial))).double()
    decoder_span = encoder_span.T @ torch.linalg.inv(encoder_span @ encoder_span.T)

    def true_decoder(z):
        return z**3 @ encoder_span

    def grad_true_decoder(zs):
        return batch_jacobian(true_decoder, zs)

    def true_encoder(x):
        z_cube = x @ decoder_span
        return z_cube.sign() * z_cube.abs().pow(1 / 3)

    def rhs_observable(x):
        # The transformed dynamics according to which the system evolves in the observable (high-dimensional) space
        return (
            grad_true_decoder(true_encoder(x))
            @ dyn_rhs(true_encoder(x)).unsqueeze(dim=-1)
        ).squeeze()

    # map data to the observable space
    snapshots_train_observable = true_decoder(snapshots_train_latent)
    snapshots_test_observable = true_decoder(snapshots_test_latent)

    # generate collocations and map them to the observable space
    collocations_latent, collocations_rhs_latent = generate_collocations(
        n_collocations_left, n_collocations_right, n_collocations_outer, seed + 3
    )
    collocations_observable = true_decoder(collocations_latent)
    collocations_rhs_observable = rhs_observable(collocations_observable).detach()

    if return_true_autoencoder:
        return (
            (snapshots_train_observable, snapshots_test_observable),
            (collocations_observable, collocations_rhs_observable),
            (true_encoder, true_decoder),
        )

    return (snapshots_train_observable, snapshots_test_observable), (
        collocations_observable,
        collocations_rhs_observable,
    )

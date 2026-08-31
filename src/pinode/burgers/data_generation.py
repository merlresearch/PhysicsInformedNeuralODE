# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import numpy as np
import torch
from scipy import integrate
from tqdm.notebook import tqdm


def normal_pdf(x, loc=0, scale=1):
    """
    Bell-curve-shaped function for Gaussian collocations

    :param x: coordinate
    :param loc: mean of the curve
    :param scale: standard deviation of the curve
    :return: value of the curve
    """
    return (
        1
        / (scale * (2 * torch.pi) ** (1 / 2))
        * torch.exp(-((x - loc) ** 2) / (2 * scale**2))
    )


def normal_pdf_x(x, loc=0, scale=1):
    """
    Spacial derivative for a bell-curve-shaped function for Gaussian collocations

    :param x: coordinate
    :param loc: mean of the curve
    :param scale: standard deviation of the curve
    :return: value of the derivative for the curve
    """
    return -(x - loc) / scale**2 * normal_pdf(x, loc=loc, scale=scale)


def normal_pdf_xx(x, loc=0, scale=1):
    """
    Second spacial derivative for a bell-curve-shaped function for Gaussian collocations

    :param x: coordinate
    :param loc: mean of the curve
    :param scale: standard deviation of the curve
    :return: value of the second derivative for the curve
    """
    return ((x - loc) ** 2 / scale**4 - 1 / scale**2) * normal_pdf(
        x, loc=loc, scale=scale
    )


def normal_pdf_xxx(x, loc=0, scale=1):
    return (
        1
        / (scale * (2 * torch.pi) ** (1 / 2))
        * (
            3 * (x - loc) * torch.exp(-((x - loc) ** 2) / (2 * scale**2)) / scale**4
            - (x - loc) ** 3 * torch.exp(-((x - loc) ** 2) / (2 * scale**2)) / scale**6
        )
    )


def normal_pdf_xxxx(x, loc=0, scale=1):
    """
    Second spacial derivative for a bell-curve-shaped function for Gaussian collocations

    :param x: coordinate
    :param loc: mean of the curve
    :param scale: standard deviation of the curve
    :return: value of the second derivative for the curve
    """
    return (
        1
        / (scale * (2 * torch.pi) ** (1 / 2))
        * (
            (x - loc) ** 4 * torch.exp(-((x - loc) ** 2) / (2 * scale**2)) / scale**8
            - 6
            * (x - loc) ** 2
            * torch.exp(-((x - loc) ** 2) / (2 * scale**2))
            / scale**6
            + 3 * torch.exp(-((x - loc) ** 2) / (2 * scale**2)) / scale**4
        )
    )


def sigmoid(x, x0=0, k=20, el=1):
    """
    Sigmoid function

    :param x: coordinate
    :param x0: coordinate of the center of the slope
    :param k: steepness of the slope
    :param el: the altitude of the peak
    :return: value of the curve
    """
    return el / (1 + np.exp(-k * (x - x0)))


def sigmoid_x(x, x0=0, k=20, el=1):
    """
    First spacial derivative of the sigmoid function

    :param x: coordinate
    :param x0: coordinate of the center of the slope
    :param k: steepness of the slope
    :param el: the altitude of the peak
    :return: value of the curve
    """
    return el * k * torch.exp(-k * (x - x0)) / (1 + torch.exp(-k * (x - x0))) ** 2


def sigmoid_xx(x, x0=0, k=20, el=1):
    """
    Second spacial derivative of the sigmoid function

    :param x: coordinate
    :param x0: coordinate of the center of the slope
    :param k: steepness of the slope
    :param el: the altitude of the peak
    :return: value of the curve
    """
    return el * (
        2 * k**2 * torch.exp(-2 * k * (x - x0)) / (torch.exp(-k * (x - x0)) + 1) ** 3
        - k**2 * torch.exp(-k * (x - x0)) / (1 + torch.exp(-k * (x - x0))) ** 2
    )


def sigmoid_xxx(x, x0=0, k=20, el=1):
    return (
        k**3
        * el
        * torch.exp(k * (x - x0))
        * (-4 * torch.exp(k * (x - x0)) + torch.exp(2 * k * (x - x0)) + 1)
        / (torch.exp(k * (x - x0)) + 1) ** 4
    )


def sigmoid_xxxx(x, x0=0, k=20, el=1):
    """
    Second spacial derivative of the sigmoid function

    :param x: coordinate
    :param x0: coordinate of the center of the slope
    :param k: steepness of the slope
    :param el: the altitude of the peak
    :return: value of the curve
    """
    return (
        -(k**4)
        * el
        * torch.exp(k * (x - x0))
        * (torch.exp(k * (x - x0)) - 1)
        * (-10 * torch.exp(k * (x - x0)) + torch.exp(2 * k * (x - x0)) + 1)
        / (torch.exp(k * (x - x0)) + 1) ** 5
    )


def burgers_fourier_rhs(t, uh, kx, nu):
    """
    Right hand-side of Burgers' equation in the fourier domain
    :param t: time
    :param uh: solution
    :param kx: frequencies
    :param nu: viscosity constant
    :return:
    """
    return -0.5 * 1j * kx * np.fft.fft(np.fft.ifft(uh) ** 2) - nu * kx**2 * uh


def burgers_solver(nu, L, x, t, u0):
    """
    Solve the 1-D Burgers equation using the Fourier Transform.
    For periodic boundary conditions
    Arguments:
        inputs:
        eps -- diffusion coeff/ viscosity
        L -- length of spatial domain, the domain will be [-L/2, L/2)
        x -- 1D numpy array with spatial discretization
        t -- 1D numpy array with time discretization
        u0 -- 1D numpy array with initial condition
        outputs:
        U -- 2D numpy array with solution to Heat equation, shape is
            (number of time steps, number of spatial points)
    """
    n = len(x)
    kx = (2 * np.pi / L) * np.fft.fftfreq(n, d=1 / n)
    uh0 = np.fft.fft(u0)

    sol = integrate.solve_ivp(
        burgers_fourier_rhs, (t[0], t[-1]), uh0, "RK45", t, args=(kx, nu)
    )
    uh = sol.y
    u = np.fft.ifft(uh, axis=0).real.T

    return u


def generate_data(
    noise=0.0,
    nu=0.01,
    n_spatial=128,
    n_modes=10,
    n_collocations_sine=10,
    n_collocations_normal=10,
    n_collocations_bumps=0,
    n_snapshots_train_sine=10,
    n_snapshots_train_normal=10,
    n_snapshots_train_bumps=0,
    n_snapshots_test_sine=10,
    n_snapshots_test_normal=10,
    n_snapshots_test_bumps=0,
    n_bumps=3,
    k_bumps=20,
    t_train=None,
    t_test=None,
    seed=42,
    shift=0,
    with_progress_bar=False,
    randomize_modes=False,
    **kwargs,
):
    """
    Generates a dataset for Burgers' experiment

    :param noise: Standard deviation of normal iid pixel-wide noise
    :param nu: Viscosity for Burgers data
    :param n_spatial: Spacial resolution for Burgers solutions
    :param n_modes: Number of harmonic modes used for ICs
    :param n_collocations_sine: Number of harmonic collocations
    :param n_collocations_normal: Number of Gaussian (bell-curve) collocations
    :param n_collocations_bumps: Number of bumps collocations
    :param n_snapshots_train_sine: Number of trajectories with harmonic ICs in train
    :param n_snapshots_train_normal: Number of trajectories with Gaussian ICs in train
    :param n_snapshots_train_bumps: Number of trajectories with bumps ICs in train
    :param n_snapshots_test_sine: Number of trajectories with harmonic ICs in test
    :param n_snapshots_test_normal: # Number of trajectories with Gaussian ICs in test
    :param n_snapshots_test_bumps: Number of trajectories with bumps ICs in test
    :param n_bumps: Number of bumps per IC for bumps data
    :param k_bumps: Slope constant for sigmoid bumps
    :param t_train:
    :param t_test:
    :param seed: Random seed
    :param shift: Additive shift for all Burgers solutions
    :param with_progress_bar: whether to display a progress bar (used in Jupyter notebooks)
    :param randomize_modes: Whether to randomly cap frequencies for sine cols.
    :param kwargs: everything else
    :return: Generated data set
    xs -- spacial grid for solutions
    u -- collocations
    u_x -- spacial derivatives for collocations
    u_xx -- second spacial derivatives for collocations
    u_snapshots_train -- snapshots for the train data
    u_snapshots_test -- snapshots for the test data
    """

    xs = torch.from_numpy(np.linspace(-np.pi, np.pi, n_spatial, dtype=np.double))

    u = []
    u_x = []
    u_xx = []
    u_snapshots_train = []
    u_snapshots_test = []

    # Create a harmonic basis for harmonic collocations
    u_basis = list()
    u_x_basis = list()
    u_xx_basis = list()
    for k in range(n_modes // 2):
        u_basis.append(torch.cos(k * xs))
        u_basis.append(torch.sin((k + 1) * xs))
        u_x_basis.append(-k * torch.sin(k * xs))
        u_x_basis.append((k + 1) * torch.cos((k + 1) * xs))
        u_xx_basis.append(-(k**2) * torch.cos(k * xs))
        u_xx_basis.append(-((k + 1) ** 2) * torch.sin((k + 1) * xs))

    u_basis = torch.stack(u_basis) + shift
    u_x_basis = torch.stack(u_x_basis)
    u_xx_basis = torch.stack(u_xx_basis)

    # sine trajectories
    if n_snapshots_test_sine + n_snapshots_train_sine > 0:
        _rnd = np.random.default_rng(seed=seed)
        ws_snapshots = (
            0.1
            * torch.from_numpy(
                _rnd.multivariate_normal(
                    mean=np.zeros(n_modes),
                    cov=np.eye(n_modes),
                    size=n_snapshots_train_sine + n_snapshots_test_sine,
                )
            ).double()
        )
        u0s = torch.matmul(ws_snapshots, u_basis)
        u_snapshots_train += [
            torch.from_numpy(burgers_solver(nu, 2 * np.pi, xs, t_train, u_0))
            for u_0 in (
                tqdm(u0s[n_snapshots_test_sine:], desc="Harmonic snapshots train")
                if with_progress_bar
                else u0s[n_snapshots_test_sine:]
            )
        ]
        u_snapshots_test += [
            torch.from_numpy(burgers_solver(nu, 2 * np.pi, xs, t_test, u_0))
            for u_0 in (
                tqdm(u0s[:n_snapshots_test_sine], desc="Harmonic snapshots test")
                if with_progress_bar
                else u0s[:n_snapshots_test_sine]
            )
        ]

    # sine collocations
    if n_collocations_sine > 0:
        _rnd = np.random.default_rng(seed=seed + 1)
        ws_collocations = (
            0.1
            * torch.from_numpy(
                _rnd.multivariate_normal(
                    mean=np.zeros(n_modes),
                    cov=np.eye(n_modes),
                    size=n_collocations_sine,
                )
            ).double()
        )

        if randomize_modes:
            cutoff_for_frequencies = _rnd.integers(
                low=1, high=n_modes, size=n_collocations_sine
            )
            for i in range(n_collocations_sine):
                ws_collocations[i, cutoff_for_frequencies[i] :] = 0

        u += [*torch.matmul(ws_collocations, u_basis)]
        u_x += [*torch.matmul(ws_collocations, u_x_basis)]
        u_xx += [*torch.matmul(ws_collocations, u_xx_basis)]

    if n_snapshots_train_normal + n_snapshots_test_normal > 0:
        # normal trajectories
        _rnd = np.random.default_rng(seed=seed + 2)
        scales = _rnd.uniform(
            low=0.1, high=1, size=n_snapshots_test_normal + n_snapshots_train_normal
        )
        u0s = torch.stack(
            [normal_pdf(xs, loc=0, scale=scale) for scale in scales], dim=0
        )

        u_snapshots_train += [
            torch.from_numpy(burgers_solver(nu, 2 * np.pi, xs, t_train, u_0))
            for u_0 in (
                tqdm(u0s[n_snapshots_test_normal:], desc="Normal snapshots train")
                if with_progress_bar
                else u0s[n_snapshots_test_normal:]
            )
        ]
        u_snapshots_test += [
            torch.from_numpy(burgers_solver(nu, 2 * np.pi, xs, t_test, u_0))
            for u_0 in (
                tqdm(u0s[:n_snapshots_test_normal], desc="Normal snapshots test")
                if with_progress_bar
                else u0s[:n_snapshots_test_normal]
            )
        ]

    # normal collocations
    if n_collocations_normal > 0:
        _rnd = np.random.default_rng(seed=seed + 3)
        scales = _rnd.uniform(low=0.1, high=1, size=n_collocations_normal)
        shifts = _rnd.integers(
            low=0, high=len(xs), size=n_collocations_normal, dtype=int
        )
        if with_progress_bar:
            normal_collocations_range = tqdm(
                list(zip(shifts, scales)), desc="Normal Collocations"
            )
        else:
            normal_collocations_range = zip(shifts, scales)
        for shift, scale in normal_collocations_range:
            u_collocation = normal_pdf(xs, loc=0, scale=scale)
            u_x_collocation = normal_pdf_x(xs, loc=0, scale=scale)
            u_xx_collocation = normal_pdf_xx(xs, loc=0, scale=scale)
            u.append(torch.hstack([u_collocation[shift:], u_collocation[:shift]]))
            u_x.append(torch.hstack([u_x_collocation[shift:], u_x_collocation[:shift]]))
            u_xx.append(
                torch.hstack([u_xx_collocation[shift:], u_xx_collocation[:shift]])
            )

    # bump trajectories
    if n_snapshots_train_bumps + n_snapshots_test_bumps > 0:
        _rnd = np.random.default_rng(seed=seed + 4)
        if with_progress_bar:
            bumps_range = tqdm(
                range(n_snapshots_test_bumps + n_snapshots_train_bumps),
                desc="Bump snapshots test + train",
            )
        else:
            bumps_range = range(n_snapshots_test_bumps + n_snapshots_train_bumps)
        for i in bumps_range:
            positions = _rnd.uniform(
                low=-np.pi * 0.8, high=np.pi * 0.8, size=2 * n_bumps
            )
            positions.sort()
            u0 = 0
            for a, b in zip(positions[::2], positions[1::2]):
                u0 += sigmoid(xs, x0=a, k=k_bumps) - sigmoid(xs, x0=b, k=k_bumps)
            if i >= n_snapshots_test_bumps:
                u_snapshots_train.append(
                    torch.from_numpy(burgers_solver(nu, 2 * np.pi, xs, t_train, u0))
                )
            else:
                u_snapshots_test.append(
                    torch.from_numpy(burgers_solver(nu, 2 * np.pi, xs, t_test, u0))
                )

    # Bump collocations
    if n_collocations_bumps > 0:
        _rnd = np.random.default_rng(seed=seed + 5)
        if with_progress_bar:
            bumps_range = tqdm(range(n_collocations_bumps), desc="Bump collocations")
        else:
            bumps_range = range(n_collocations_bumps)
        for _ in bumps_range:
            positions = _rnd.uniform(
                low=-np.pi * 0.6, high=np.pi * 0.8, size=2 * n_bumps
            )
            positions.sort()
            u_current = 0
            u_x_current = 0
            u_xx_current = 0
            k_bumps_left = _rnd.uniform(low=0.5, high=k_bumps, size=n_bumps)
            amplitudes = _rnd.uniform(low=0.1, high=1, size=n_bumps)
            for a, b, k_bump_left, amplitude in zip(
                positions[::2], positions[1::2], k_bumps_left, amplitudes
            ):
                u_current += amplitude * (
                    sigmoid(xs, x0=a, k=k_bump_left) - sigmoid(xs, x0=b, k=k_bumps)
                )
                u_x_current += amplitude * (
                    sigmoid_x(xs, x0=a, k=k_bump_left) - sigmoid_x(xs, x0=b, k=k_bumps)
                )
                u_xx_current += amplitude * (
                    sigmoid_xx(xs, x0=a, k=k_bump_left)
                    - sigmoid_xx(xs, x0=b, k=k_bumps)
                )
            u.append(u_current)
            u_x.append(u_x_current)
            u_xx.append(u_xx_current)

    if n_collocations_normal + n_collocations_sine + n_collocations_bumps > 0:
        u = torch.stack(u, dim=0)
        u_x = torch.stack(u_x, dim=0)
        u_xx = torch.stack(u_xx, dim=0)

    if len(u_snapshots_train) > 0:
        u_snapshots_train = torch.stack(u_snapshots_train, dim=0)
    if len(u_snapshots_test) > 0:
        u_snapshots_test = torch.stack(u_snapshots_test, dim=0)

    if noise > 0 and len(u_snapshots_train) > 0:
        _rnd = np.random.default_rng(seed=seed + 6)
        u_snapshots_train += torch.from_numpy(
            _rnd.normal(loc=0.0, scale=noise, size=u_snapshots_train.shape)
        ).double()

    return xs, u, u_x, u_xx, u_snapshots_train, u_snapshots_test

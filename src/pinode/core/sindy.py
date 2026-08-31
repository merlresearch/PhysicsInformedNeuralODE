# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import torch

library_3d_2nd_order_polynomials = {
    "1": lambda y, device="cpu": torch.ones(y.shape[:-1], device=device),  # 0
    "x": lambda y, _: y[..., 0],  # 1
    "y": lambda y, _: y[..., 1],  # 2
    "z": lambda y, _: y[..., 2],  # 3
    "xy": lambda y, _: y[..., 0] * y[..., 1],  # 4
    "yz": lambda y, _: y[..., 1] * y[..., 2],  # 5
    "xz": lambda y, _: y[..., 0] * y[..., 2],  # 6
    "x^2": lambda y, _: y[..., 0] ** 2,  # 7
    "y^2": lambda y, _: y[..., 1] ** 2,  # 8
    "z^2": lambda y, _: y[..., 2] ** 2,  # 9
}

library_2d_5th_order_polynomials = {
    "1": lambda y, device="cpu": torch.ones(y.shape[:-1], device=device),
    "x": lambda y, _: y[..., 0],
    "y": lambda y, _: y[..., 1],
    "xy": lambda y, _: y[..., 0] * y[..., 1],
    "x^2": lambda y, _: y[..., 0] ** 2,
    "y^2": lambda y, _: y[..., 1] ** 2,
    "x^3": lambda y, _: y[..., 0] ** 3,
    "y^3": lambda y, _: y[..., 1] ** 3,
    "xy^2": lambda y, _: y[..., 0] * y[..., 1] ** 2,
    "x^2y": lambda y, _: y[..., 0] ** 2 * y[..., 1],
    "x^4": lambda y, _: y[..., 0] ** 4,
    "y^4": lambda y, _: y[..., 1] ** 4,
    "x^3y": lambda y, _: y[..., 0] ** 3 * y[..., 1],
    "x^2y^2": lambda y, _: y[..., 0] ** 2 * y[..., 1] ** 2,
    "xy^3": lambda y, _: y[..., 0] * y[..., 1] ** 3,
    "x^3y^2": lambda y, _: y[..., 0] ** 3 * y[..., 1] ** 2,
    "x^2y^3": lambda y, _: y[..., 0] ** 2 * y[..., 1] ** 3,
    "xy^4": lambda y, _: y[..., 1] ** 4 * y[..., 0],
    "x^4y": lambda y, _: y[..., 0] ** 4 * y[..., 1],
    "x^5": lambda y, _: y[..., 0] ** 5,
    "y^5": lambda y, _: y[..., 1] ** 5,
}


class LibraryTransformLayer(torch.nn.Module):
    """
    Transforms the compartments into the space of library functions
    """

    def __init__(self, library, device="cpu"):
        super(LibraryTransformLayer, self).__init__()
        self.library = library
        self.out_shape = len(library)
        self.device = device

    def forward(self, y):
        # y = [..., num_compartments], result = [..., num_library_functions]
        # if len(y.shape) == 1:
        #     y = y.reshape(1, -1)
        return sindy_transform(y, library=self.library, device=self.device)


def sindy_transform(data, library, device="cpu"):
    # y = [num_observations, num_compartments]
    # output = [num_observations, num_library_functions]
    return torch.stack([f(data, device) for _, f in library.items()], dim=-1)


class SindyODEFunc(torch.nn.Module):

    def __init__(self, n_variables, library, init_weights=None, device="cpu"):
        super(SindyODEFunc, self).__init__()

        self.ode = torch.nn.Sequential(
            LibraryTransformLayer(library=library, device=device),
            torch.nn.Linear(
                in_features=len(library),
                out_features=n_variables,
                bias=False,
                device=device,
                dtype=torch.double,
            ),
        ).to(device)

        if init_weights is not None:
            for m in self.ode.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight = torch.nn.Parameter(
                        data=init_weights.detach().clone().double().to(device)
                    )

    def forward(self, y):
        return self.ode(y)

    def get_sindy_coefficients(self):
        return self.get_parameter("ode.1.weight")

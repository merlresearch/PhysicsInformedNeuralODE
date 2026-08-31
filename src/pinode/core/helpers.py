# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def batch_jacobian(f, x):
    """
    Quick calculation of Jacobian of f on a batch data x that also creates a graph for the Jacobian.
    Use this function whenever you have a Jacobian or gradient of something in a loss function.

    :param f: the function for which to compute the Jacobian
    :param x: Data. The first dimension is the batch.
    :return: Vector of Jacobians of f for all batches in x
    """
    f_sum = lambda z: torch.sum(f(z), axis=0)
    return torch.autograd.functional.jacobian(f_sum, x, create_graph=True).permute(
        1, 0, 2
    )

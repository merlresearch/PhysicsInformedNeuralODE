# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


class ResidualBlock(torch.nn.Module):
    """
    Implements a residual connection around a block of layers in Pytorch
    """

    def __init__(self, residual_layers):
        """
        Constructor of the block

        :param residual_layers: a set of residual layers
        """
        super().__init__()
        self.residual_layers = residual_layers

    def forward(self, x):
        return x + self.residual_layers(x)

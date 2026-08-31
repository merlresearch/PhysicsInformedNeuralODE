# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch


class CollocationsProvider:
    def __init__(self, *args, batch_size=16, **kwargs):
        self.batch_size = batch_size

    def get_batch(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")


class StaticProvider(CollocationsProvider):

    def __init__(self, data, rhs, _rnd, *args, device="cpu", batch_size=16, **kwargs):
        # data.size = n_collocations, n_compartments, n_spatial
        self.data = data
        self.rhs = rhs
        self._rnd = _rnd
        self.device = device
        self.n_collocations = len(data)
        super().__init__(
            *args, batch_size=min(batch_size, self.n_collocations), **kwargs
        )

    @staticmethod
    def parameters():
        return []

    def get_batch(self, *args, **kwargs):
        start = torch.randint(
            low=0,
            high=self.n_collocations + 1 - self.batch_size,
            size=(1,),
            device=self.device,
            generator=self._rnd,
        )
        batch = self.data[start : start + self.batch_size]
        u = batch[:, 0]
        f = self.rhs(batch).unsqueeze(axis=-1)
        return u, f

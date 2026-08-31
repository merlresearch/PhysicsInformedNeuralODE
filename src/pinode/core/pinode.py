# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import io
import os
import sys
from pathlib import Path

import gridfs
import torch
import torch.nn.functional as F
from pymongo import MongoClient
from torch.nn.utils.parametrizations import spectral_norm
from torchdiffeq import odeint

module_path = os.path.abspath(Path(__file__).parent.parent.parent.resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from pinode.core.blocks import ResidualBlock
from pinode.core.helpers import batch_jacobian
from pinode.core.sindy import SindyODEFunc, library_2d_5th_order_polynomials


class TorchDiffEqWrapper(torch.nn.Module):
    """
    A wrapper around a pytorch module that enables it to be integrated by torchdiffeq.
    """

    def __init__(self, net):
        """
        Constructor
        :param net: a pytorch model that you want to integrate over with odeint from torchdiffeq
        """
        super().__init__()
        self.net = net

    def forward(self, t, y):
        return self.net(y)


class PINODE(torch.nn.Module):
    """
    Physics-Informed Neural ODE Model
    """

    def __init__(
        self,
        n_spatial=64,
        n_latent=10,
        n_layers=2,
        autoencoder_type="fc",
        n_latent_layers=2,
        hidden_width=512,
        dyn_rhs=None,
        latent_hidden_width=128,
        linear_projection=False,
        skip_connections=False,
        device="cpu",
        use_latent_sindy=False,
        **kwargs,
    ):
        """
        Constructor for the model

        :param n_spatial: Spacial resolution for data, also an encoder's input and the decoder's output layer size.
        :param n_latent: The size of the latent (bottleneck space)
        :param n_layers: Number of layers for encoder and decoder
        :param n_latent_layers: Number of layers for latent space network
        :param hidden_width: Width of hidden layers of encoders/decoders, if any
        :param dyn_rhs: the right-hand side of the dynamics in the observable space
        :param latent_hidden_width: Width of hidden layers for latent space network
        :param linear_projection: Not used. Left for compatibility with Yuying's code
        :param skip_connections: Not used. Left for compatibility with Yuying's code.
        :param device: Pytorch backend device to use for compute
        :param kwargs: other keyword arguments
        """

        super().__init__()

        # parameters
        self.device = device
        self.dyn_rhs = dyn_rhs
        self.n_spatial = n_spatial
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_latent_layers = n_latent_layers
        self.hidden_width = hidden_width
        self.latent_hidden_width = latent_hidden_width
        self.linear_projection = linear_projection
        self.skip_connection = skip_connections
        self.autoencoder_type = autoencoder_type

        encoder = torch.nn.Sequential()
        decoder = torch.nn.Sequential()

        if autoencoder_type == "fc":
            # encoder
            for i in range(n_layers - 1):
                input_size = n_spatial if i == 0 else hidden_width
                encoder.append(
                    torch.nn.Linear(input_size, hidden_width, dtype=torch.double)
                )
                encoder.append(torch.nn.ReLU())
            input_size = n_spatial if n_layers == 1 else hidden_width
            encoder.append(
                torch.nn.Linear(input_size, n_latent, bias=False, dtype=torch.double)
            )
            self.encoder = encoder
            self.encoder.to(self.device)

            # decoder
            if linear_projection:
                decoder.append(
                    torch.nn.Linear(n_latent, n_spatial, bias=False, dtype=torch.double)
                )
            else:
                for i in range(n_layers - 1):
                    input_size = n_latent if i == 0 else hidden_width
                    decoder.append(
                        torch.nn.Linear(input_size, hidden_width, dtype=torch.double)
                    )
                    decoder.append(torch.nn.ReLU())
                input_size = n_latent if n_layers == 1 else hidden_width
                decoder.append(
                    torch.nn.Linear(
                        input_size, n_spatial, bias=False, dtype=torch.double
                    )
                )
            self.decoder = decoder
            self.decoder.to(self.device)

        elif autoencoder_type == "conv" or autoencoder_type == "conv_flat":
            # encoder
            self.conv1 = torch.nn.Conv2d(
                in_channels=2 if autoencoder_type == "conv" else 1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.conv3 = torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.conv4 = torch.nn.Conv2d(
                in_channels=64,
                out_channels=n_latent,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.enc_fc = torch.nn.Linear(
                in_features=16 * n_latent,
                out_features=n_latent,
                device=self.device,
                dtype=torch.double,
            )

            # decoder
            self.dec_fc = spectral_norm(
                torch.nn.Linear(
                    in_features=n_latent,
                    out_features=16 * n_latent,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv1 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=n_latent,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv2 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv3 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv4 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=16,
                    out_channels=2 if autoencoder_type == "conv" else 1,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )

        elif autoencoder_type == "conv_burgers":
            # decoder
            self.deconv1 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv2 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv3 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv4 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )

        elif autoencoder_type == "conv_gas":
            # encoder
            self.conv1 = torch.nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.conv3 = torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.conv4 = torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.conv5 = torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.conv6 = torch.nn.Conv2d(
                in_channels=64,
                out_channels=n_latent,
                kernel_size=3,
                stride=1,
                padding=0,
                device=self.device,
                dtype=torch.double,
            )
            self.enc_fc = torch.nn.Linear(
                in_features=15 * n_latent,
                out_features=n_latent,
                device=self.device,
                dtype=torch.double,
            )

            # decoder
            self.dec_fc = spectral_norm(
                torch.nn.Linear(
                    in_features=n_latent,
                    out_features=15 * n_latent,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv1 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=n_latent,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv2 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv3 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv4 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv5 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=16,
                    out_channels=8,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv6 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=8,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )

        elif autoencoder_type == "conv_scalarflow":
            # encoder
            self.conv1 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=8,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.conv2 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.conv3 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.conv4 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.conv5 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.conv6 = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=64,
                    out_channels=n_latent,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.enc_fc = spectral_norm(
                torch.nn.Linear(
                    in_features=15 * n_latent,
                    out_features=n_latent,
                    device=self.device,
                    dtype=torch.double,
                )
            )

            # decoder
            self.dec_fc = spectral_norm(
                torch.nn.Linear(
                    in_features=n_latent,
                    out_features=15 * n_latent,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv1 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=n_latent,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv2 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv3 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv4 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv5 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=16,
                    out_channels=8,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
            self.deconv6 = spectral_norm(
                torch.nn.ConvTranspose2d(
                    in_channels=8,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    device=self.device,
                    dtype=torch.double,
                )
            )
        else:
            raise ValueError("Unknown autoencoder type")

        # latent dynamics
        if use_latent_sindy:
            self.latent_dynamics = SindyODEFunc(
                n_variables=n_latent,
                library=library_2d_5th_order_polynomials,
                device=self.device,
            )
        else:
            latent_dynamics = torch.nn.Sequential()
            for i in range(n_latent_layers - 1):
                input_size = n_latent if i == 0 else latent_hidden_width
                latent_dynamics.append(
                    torch.nn.Linear(input_size, latent_hidden_width, dtype=torch.double)
                )
                latent_dynamics.append(torch.nn.ReLU())
            input_size = n_latent if n_latent_layers == 1 else latent_hidden_width
            latent_dynamics.append(
                torch.nn.Linear(input_size, n_latent, bias=False, dtype=torch.double)
            )
            if skip_connections:
                self.latent_dynamics = ResidualBlock(residual_layers=latent_dynamics)
            else:
                self.latent_dynamics = latent_dynamics
        self.latent_dynamics.to(self.device)

    def forward(self, u):
        return self.encoder(u)

    def encode(self, u):
        if self.autoencoder_type == "fc":
            return self.encoder(u)
        elif self.autoencoder_type == "conv" or self.autoencoder_type == "conv_flat":
            # u = (bs * T, 2, 66, 66), comes pre-padded
            u = F.silu(self.conv1(u))  # u = (:, 16, 64, 64)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 16, 32, 32)
            # block 2
            u = F.pad(
                u, (1, 1, 1, 1), mode="circular"
            )  # u = (:, 16, 34, 34), full wrap
            u = F.silu(self.conv2(u))  # u = (:, 32, 32, 32)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 32, 16, 16)
            # block 3
            u = F.pad(
                u, (1, 1, 1, 1), mode="circular"
            )  # u = (:, 32, 18, 18), full wrap
            u = F.silu(self.conv3(u))  # u = (:, 64, 16, 16)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 64, 8, 8)
            # block 4
            u = F.pad(
                u, (1, 1, 1, 1), mode="circular"
            )  # u = (:, 64, 10, 10), full wrap
            u = F.silu(self.conv4(u))  # u = (:, n_latent, 8, 8)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, n_latent, 4, 4)
            # flatten
            u = u.view(u.shape[0], -1)  # u = (:, 16*n_latent)
            u = self.enc_fc(u)  # u = (:, n_latent)
            return u
        elif self.autoencoder_type == "conv_scalarflow":
            # u = (bs * T, 240, 320), no padding
            u = u.unsqueeze(1)  # u = (:, 1, 240, 320)
            up = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, 1, 242, 322), full wrap
            u = F.silu(self.conv1(up)) + u  # u = (:, 16, 240, 320)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 16, 120, 160)
            # block 2
            uc = torch.cat([u, u], dim=1)
            up = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, 16, 122, 162), full wrap
            u = F.silu(self.conv2(up)) + uc  # u = (:, 32, 120, 160)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 32, 60, 80)
            # block 3
            uc = torch.cat([u, u], dim=1)
            up = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, 32, 62, 82), full wrap
            u = F.silu(self.conv3(up)) + uc  # u = (:, 64, 60, 80)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 64, 30, 40)
            # block 4
            uc = torch.cat([u, u], dim=1)
            up = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, 64, 32, 42), full wrap
            u = F.silu(self.conv4(up)) + uc  # u = (:, n_latent, 30, 40)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, n_latent, 15, 20)
            # block 5
            uc = u  # torch.cat([u, u], dim=1)
            up = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, n_latent, 17, 22), full wrap
            u = F.silu(self.conv5(up)) + uc  # u = (:, n_latent, 15, 20)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, n_latent, 7, 10)
            # block 6
            uc = u[:, : self.n_latent]
            up = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, n_latent, 9, 12), full wrap
            u = F.silu(self.conv6(up)) + uc  # u = (:, n_latent, 7, 10)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, n_latent, 3, 5)
            # flatten
            u = u.view(u.shape[0], -1)  # u = (:, 15*n_latent)
            u = self.enc_fc(u)  # u = (:, n_latent)
            return u.squeeze(dim=1)
        elif self.autoencoder_type == "conv_gas":
            # u = (bs * T, 240, 320), no padding
            u = u.unsqueeze(1)  # u = (:, 1, 240, 320)
            u = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, 1, 242, 322), full wrap
            u = F.silu(self.conv1(u))  # u = (:, 16, 240, 320)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 16, 120, 160)
            # block 2
            u = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, 16, 122, 162), full wrap
            u = F.silu(self.conv2(u))  # u = (:, 32, 120, 160)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 32, 60, 80)
            # block 3
            u = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, 32, 62, 82), full wrap
            u = F.silu(self.conv3(u))  # u = (:, 64, 60, 80)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, 64, 30, 40)
            # block 4
            u = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, 64, 32, 42), full wrap
            u = F.silu(self.conv4(u))  # u = (:, n_latent, 30, 40)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, n_latent, 15, 20)
            # block 5
            u = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, n_latent, 17, 22), full wrap
            u = F.silu(self.conv5(u))  # u = (:, n_latent, 15, 20)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, n_latent, 7, 10)
            # block 6
            u = F.pad(
                u, (1, 1, 1, 1), mode="replicate"
            )  # u = (:, n_latent, 9, 12), full wrap
            u = F.silu(self.conv6(u))  # u = (:, n_latent, 7, 10)
            u = F.avg_pool2d(u, 2, 2)  # u = (:, n_latent, 3, 5)
            # flatten
            u = u.view(u.shape[0], -1)  # u = (:, 15*n_latent)
            u = self.enc_fc(u)  # u = (:, n_latent)
            return u.squeeze(dim=1)
        else:
            raise ValueError("Unknown autoencoder type")

    def decode(self, z):
        if self.autoencoder_type == "fc":
            u = self.decoder(z)
        elif self.autoencoder_type == "conv" or self.autoencoder_type == "conv_flat":
            # z = (bs*T, n_latent)
            z = F.silu(self.dec_fc(z))  # z = (:, 16*n_latent)
            z = z.view(z.shape[0], self.n_latent, 4, 4)  # z = (:, n_latent, 4, 4)
            # block 1
            z = F.interpolate(
                z, scale_factor=2, mode="bilinear"
            )  # z = (:, n_latent, 8, 8)
            z = F.pad(
                z, (1, 1, 1, 1), mode="circular"
            )  # z = (:, n_latent, 10, 10), full wrap
            z = F.silu(self.deconv1(z))  # z = (:, 64, 12, 12)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 64, 8, 8)
            # block 2
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 64, 16, 16)
            z = F.pad(
                z, (1, 1, 1, 1), mode="circular"
            )  # z = (:, 64, 18, 18), full wrap
            z = F.silu(self.deconv2(z))  # z = (:, 32, 20, 20)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 32, 16, 16)
            # block 3
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 32, 32, 32)
            z = F.pad(
                z, (1, 1, 1, 1), mode="circular"
            )  # z = (:, 32, 34, 34), full wrap
            z = F.silu(self.deconv3(z))  # z = (:, 16, 36, 36)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 16, 32, 32)
            # block 4
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 16, 64, 64)
            z = F.pad(
                z, (1, 1, 1, 1), mode="circular"
            )  # z = (:, 16, 66, 66), full wrap
            z = self.deconv4(z)  # z = (:, 2, 68, 68)
            z = z[:, :, 1:-1, 1:-1]  # z = (:, 2, 66, 66)
            u = z
        elif self.autoencoder_type == "conv_scalarflow":
            # z = (bs*T, n_latent)
            z = F.silu(self.dec_fc(z))  # z = (:, 16*n_latent)
            z = z.view(
                z.shape[0],
                self.n_latent,
                *((3, 5) if self.autoencoder_type == "conv_gas" else (5, 3)),
            )  # z = (:, n_latent, 3, 5)
            # block 1
            z = F.interpolate(
                z, scale_factor=2, mode="bilinear"
            )  # z = (:, n_latent, 8, 8)
            z = F.pad(
                z,
                (1, 1, 1, 2 if self.autoencoder_type == "conv_gas" else 1),
                mode="replicate",
            )  # z = (:, n_latent, 10, 10), full wrap
            z = F.silu(self.deconv1(z))  # z = (:, 64, 12, 12)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 64, 8, 8)
            # block 2
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 64, 16, 16)
            zp = F.pad(
                z,
                (1, 1, 1, 2 if self.autoencoder_type == "conv_gas" else 1),
                mode="replicate",
            )  # z = (:, 64, 18, 18), full wrap
            zp = F.silu(self.deconv2(zp))  # z = (:, 64, 20, 20)
            z = zp[:, :, 2:-2, 2:-2] + z  # z = (:, 64, 16, 16)
            # block 3
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 64, 32, 32)
            zp = F.pad(
                z, (1, 1, 1, 1), mode="replicate"
            )  # z = (:, 64, 34, 34), full wrap
            zp = F.silu(self.deconv3(zp))  # z = (:, 32, 36, 36)
            z = zp[:, :, 2:-2, 2:-2] + z[:, : zp.shape[1]]  # z = (:, 32, 32, 32)
            # block 4
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 32, 64, 64)
            zp = F.pad(
                z, (1, 1, 1, 1), mode="replicate"
            )  # z = (:, 32, 66, 66), full wrap
            zp = F.silu(self.deconv4(zp))  # z = (:, 16, 68, 68)
            z = zp[:, :, 2:-2, 2:-2] + z[:, : zp.shape[1]]  # z = (:, 16, 66, 66)
            # block 5
            z = F.interpolate(
                z, scale_factor=2, mode="bilinear"
            )  # z = (:, 16, 128, 128)
            zp = F.pad(
                z, (1, 1, 1, 1), mode="replicate"
            )  # z = (:, 16, 130, 130), full wrap
            zp = F.silu(self.deconv5(zp))  # z = (:, 8, 130, 130)
            z = zp[:, :, 2:-2, 2:-2] + z[:, : zp.shape[1]]  # z = (:, 8, 128, 128)
            # block 6
            z = F.interpolate(
                z, scale_factor=2, mode="bilinear"
            )  # z = (:, 8, 256, 256)
            zp = F.pad(
                z, (1, 1, 1, 1), mode="replicate"
            )  # z = (:, 8, 258, 258), full wrap
            zp = self.deconv6(zp)  # z = (:, 1, 258, 258)
            z = zp[:, :, 2:-2, 2:-2] + z[:, : zp.shape[1]]  # z = (:, 1, 256, 256)
            u = z.squeeze(dim=1)  # u = (:, 256, 256)
        elif self.autoencoder_type == "conv_gas":
            # z = (bs*T, n_latent)
            z = F.silu(self.dec_fc(z))  # z = (:, 16*n_latent)
            z = z.view(z.shape[0], self.n_latent, 3, 5)  # z = (:, n_latent, 4, 4)
            # block 1
            z = F.interpolate(
                z, scale_factor=2, mode="bilinear"
            )  # z = (:, n_latent, 8, 8)
            z = F.pad(
                z, (1, 1, 1, 2), mode="replicate"
            )  # z = (:, n_latent, 10, 10), full wrap
            z = F.silu(self.deconv1(z))  # z = (:, 64, 12, 12)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 64, 8, 8)
            # block 2
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 64, 16, 16)
            z = F.pad(
                z, (1, 1, 1, 2), mode="replicate"
            )  # z = (:, 64, 18, 18), full wrap
            z = F.silu(self.deconv2(z))  # z = (:, 64, 20, 20)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 64, 16, 16)
            # block 3
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 64, 32, 32)
            z = F.pad(
                z, (1, 1, 1, 1), mode="replicate"
            )  # z = (:, 64, 34, 34), full wrap
            z = F.silu(self.deconv3(z))  # z = (:, 32, 36, 36)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 32, 32, 32)
            # block 4
            z = F.interpolate(z, scale_factor=2, mode="bilinear")  # z = (:, 32, 64, 64)
            z = F.pad(
                z, (1, 1, 1, 1), mode="replicate"
            )  # z = (:, 32, 66, 66), full wrap
            z = F.silu(self.deconv4(z))  # z = (:, 16, 68, 68)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 16, 66, 66)
            # block 5
            z = F.interpolate(
                z, scale_factor=2, mode="bilinear"
            )  # z = (:, 16, 128, 128)
            z = F.pad(
                z, (1, 1, 1, 1), mode="replicate"
            )  # z = (:, 16, 130, 130), full wrap
            z = F.silu(self.deconv5(z))  # z = (:, 8, 130, 130)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 8, 128, 128)
            # block 6
            z = F.interpolate(
                z, scale_factor=2, mode="bilinear"
            )  # z = (:, 8, 256, 256)
            z = F.pad(
                z, (1, 1, 1, 1), mode="replicate"
            )  # z = (:, 8, 258, 258), full wrap
            z = F.silu(self.deconv6(z))  # z = (:, 1, 258, 258)
            z = z[:, :, 2:-2, 2:-2]  # z = (:, 1, 256, 256)
            u = z.squeeze(dim=1)
        elif self.autoencoder_type == "conv_burgers":
            z = z.unsqueeze(1)
            # block 1
            z = F.pad(
                z, (1, 1, 0, 0), mode="circular"
            )  # z = (:, 1, T, S+2), cyclic padding
            z = F.pad(
                z, (0, 0, 1, 1), mode="replicate"
            )  # z = (:, 16, T+2, S+2), same padding
            z = F.silu(self.deconv1(z))  # z = (:, 16, T, S)
            # block 2
            z = F.pad(
                z, (1, 1, 0, 0), mode="circular"
            )  # z = (:, 16, T, S+2), cyclic padding
            z = F.pad(
                z, (0, 0, 1, 1), mode="replicate"
            )  # z = (:, 16, T+2, S+2), same padding
            z = F.silu(self.deconv2(z))  # z = (:, 32, T, S)
            # block 3
            z = F.pad(
                z, (1, 1, 0, 0), mode="circular"
            )  # z = (:, 32, T, S+2), cyclic padding
            z = F.pad(
                z, (0, 0, 1, 1), mode="replicate"
            )  # z = (:, 32, T+2, S+2), same padding
            z = F.silu(self.deconv3(z))  # z = (:, 16, T, S)
            # block 4
            z = F.pad(
                z, (1, 1, 0, 0), mode="circular"
            )  # z = (:, 16, T, S+2), cyclic padding
            z = F.pad(
                z, (0, 0, 1, 1), mode="replicate"
            )  # z = (:, 16, T+2, S+2), same padding
            z = self.deconv4(z)  # z = (:, 1, T, S)
            u = z.squeeze()
        else:
            raise ValueError("Unknown autoencoder type")
        return u

    def predict(self, u_init, ts):
        """
        Does batch prediction for given initial conditions over the given timeframe

        :param u_init: (a batch of) initial conditions
        :param ts: timeframe
        :return: predictions
        """
        v_init = self.encode(u_init)
        v_predictions = odeint(
            TorchDiffEqWrapper(self.latent_dynamics), v_init, ts
        ).transpose(1, 0)
        if self.autoencoder_type == "fc":
            u_predictions = self.decode(v_predictions)
        elif self.autoencoder_type == "conv" or self.autoencoder_type == "conv_flat":
            u_predictions = self.decode(v_predictions.reshape(-1, self.n_latent)).view(
                len(u_init),
                len(ts),
                u_init.shape[-3],
                u_init.shape[-2],
                u_init.shape[-1],
            )
        elif (
            self.autoencoder_type == "conv_gas"
            or self.autoencoder_type == "conv_scalarflow"
        ):
            n_spatial_height, n_spatial_width = u_init.shape[1:]
            u_predictions = self.decode(v_predictions.reshape(-1, self.n_latent)).view(
                len(u_init), len(ts), n_spatial_height, n_spatial_width
            )
        else:
            raise ValueError("Unknown autoencoder type")
        return u_predictions

    def predict_latent(self, z_init, ts):
        return odeint(TorchDiffEqWrapper(self.latent_dynamics), z_init, ts).transpose(
            1, 0
        )

    def freeze_weights(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_weights(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    def physics_informed(self, u, f):
        """
        Calculates components of physics-informed loss function

        :param u:
        :param u_x:
        :param u_xx:
        :return:
        """
        v = self.encoder(u)
        u_hat = self.decoder(v)
        Lv = self.latent_dynamics(v)
        dv_du = batch_jacobian(self.encoder, u)
        dv_dt = torch.bmm(dv_du, f).squeeze(axis=-1)
        return v, u_hat, Lv, dv_dt

    def project_collocations_to_latent(self, u, du_dt, encoder=None):
        if not encoder:
            encoder = self.encoder
        dv_du = batch_jacobian(encoder, u)
        dv_dt = torch.bmm(dv_du, du_dt.unsqueeze(-1)).squeeze(axis=-1)
        return dv_dt


class PINODE_GIN(torch.nn.Module):
    """
    A version of PINODE that implements the inner-outer encoder-decoder architecture from Gin et al.
    https://arxiv.org/abs/1911.02710

    Everything is the same as for the model above, just the arrangement of layers is different.
    """

    def __init__(
        self,
        n_spatial=64,
        n_latent=10,
        n_layers=2,
        n_latent_layers=2,
        dyn_rhs=None,
        latent_hidden_width=128,
        linear_projection=False,
        skip_connections=False,
        device="cpu",
        **kwargs,
    ):

        super().__init__()

        # parameters
        self.device = device
        self.dyn_rhs = dyn_rhs
        self.n_spatial = n_spatial
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_latent_layers = n_latent_layers
        self.hidden_width = n_spatial
        self.latent_hidden_width = latent_hidden_width
        self.linear_projection = linear_projection
        self.skip_connection = skip_connections

        # encoder
        encoder = torch.nn.Sequential()
        for i in range(n_layers - 1):
            encoder.append(
                torch.nn.Linear(n_spatial, n_spatial, dtype=torch.double, device=device)
            )
            encoder.append(torch.nn.ReLU())
        outer_encoder = ResidualBlock(encoder)
        self.outer_encoder = outer_encoder
        self.inner_encoder = torch.nn.Linear(
            n_spatial, n_latent, bias=False, dtype=torch.double, device=device
        )
        self.encoder = torch.nn.Sequential(self.outer_encoder, self.inner_encoder)

        # latent dynamics
        latent_dynamics = torch.nn.Sequential()
        for i in range(n_latent_layers - 1):
            input_size = n_latent if i == 0 else latent_hidden_width
            latent_dynamics.append(
                torch.nn.Linear(
                    input_size, latent_hidden_width, dtype=torch.double, device=device
                )
            )
            latent_dynamics.append(torch.nn.ReLU())
        input_size = n_latent if n_latent_layers == 1 else latent_hidden_width
        latent_dynamics.append(
            torch.nn.Linear(
                input_size, n_latent, bias=False, dtype=torch.double, device=device
            )
        )
        self.latent_dynamics = latent_dynamics

        # decoder
        self.inner_decoder = torch.nn.Linear(
            n_latent, n_spatial, bias=False, dtype=torch.double, device=device
        )
        decoder = torch.nn.Sequential()
        for i in range(n_layers - 1):
            decoder.append(
                torch.nn.Linear(n_spatial, n_spatial, dtype=torch.double, device=device)
            )
            decoder.append(torch.nn.ReLU())
        self.outer_decoder = ResidualBlock(decoder)
        self.decoder = torch.nn.Sequential(self.inner_decoder, self.outer_decoder)

    def predict(self, u_init, ts):
        v_init = self.encoder(u_init)
        v_predictions = odeint(
            TorchDiffEqWrapper(self.latent_dynamics), v_init, ts
        ).transpose(1, 0)
        return self.decoder(v_predictions)


def load_experiment(model_id, mongo_auth_str, device="cpu", autoencoder_type=None):
    """
    Fetch a model from the database by its id.

    model_id: str -- an ID of the model that you want to load. E.g. "372".

    returns: a 4-tuple of
        model -- a trained pytorch model
        config -- a dictionary that stores all input parameters that were used in this experiment
        metrics -- a json dictionary that contains evaluations of the model's performance along the training (test loss for each epoch, etc)
        run -- a lot of meta information about this run (duration, hardware, git commit header, even the code itself)
    """
    client = MongoClient(mongo_auth_str)
    db = client.sacred
    files = gridfs.GridFS(db)
    run = db.runs.find_one(model_id)
    config = run["config"]
    config["device"] = device
    if autoencoder_type:
        config["autoencoder_type"] = autoencoder_type
    metrics = {v["name"]: v for v in db.metrics.find({"run_id": run["_id"]})}
    model = PINODE(**config)
    artifacts_ids = {s["name"]: s["file_id"] for s in run["artifacts"]}
    state_dict = torch.load(
        io.BytesIO(files.get(artifacts_ids["model"]).read()),
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(state_dict)
    return model, config, metrics, run


class FG_BG_PINODE:
    def __init__(self, fg_model, bg_model, device="cpu"):
        self.fg_model = fg_model
        self.bg_model = bg_model
        self.device = device
        self.autoencoder_type = fg_model.autoencoder_type
        self.n_latent = fg_model.n_latent + bg_model.n_latent

    def predict(self, u_init, ts, separately=False):
        u_init_fg = u_init[:, 0]
        v_init_fg = self.fg_model.encoder(u_init_fg)
        v_predictions = odeint(
            TorchDiffEqWrapper(self.fg_model.latent_dynamics), v_init_fg, ts
        ).transpose(1, 0)
        u_fg_predictions = self.fg_model.decoder(v_predictions)
        u_init_bg = u_init[:, 1]
        v_init_bg = self.bg_model.encoder(u_init_bg)
        v_predictions = odeint(
            TorchDiffEqWrapper(self.bg_model.latent_dynamics), v_init_bg, ts
        ).transpose(1, 0)
        u_bg_predictions = self.bg_model.decoder(v_predictions)
        if separately:
            return (
                u_fg_predictions + u_bg_predictions,
                u_fg_predictions,
                u_bg_predictions,
            )
        return u_fg_predictions + u_bg_predictions

    def decode(self, v, separately=False):
        v_fg = v[:, : self.fg_model.n_latent]
        v_bg = v[:, self.fg_model.n_latent :]
        u_fg = self.fg_model.decode(v_fg)
        u_bg = self.bg_model.decode(v_bg)
        if separately:
            return u_fg + u_bg, u_fg, u_bg
        return u_fg + u_bg

    def predict_latent(self, v0, ts):
        v0_fg = v0[:, : self.fg_model.n_latent]
        v0_bg = v0[:, self.fg_model.n_latent :]
        v_fg_predictions = odeint(
            TorchDiffEqWrapper(self.fg_model.latent_dynamics), v0_fg, ts
        ).transpose(1, 0)
        v_bg_predictions = odeint(
            TorchDiffEqWrapper(self.bg_model.latent_dynamics), v0_bg, ts
        ).transpose(1, 0)
        return torch.cat([v_fg_predictions, v_bg_predictions], dim=2)

    def freeze_weights(self):
        self.fg_model.freeze_weights()
        self.bg_model.freeze_weights()

    def unfreeze_weights(self):
        self.fg_model.unfreeze_weights()
        self.bg_model.unfreeze_weights()


def load_bg_experiment(model_id, mongo_auth_str, device="cpu", autoencoder_type=None):
    """
    Fetch a model from the database by its id.

    model_id: str -- an ID of the model that you want to load. E.g. "372".

    returns: a 4-tuple of
        model -- a trained pytorch model
        config -- a dictionary that stores all input parameters that were used in this experiment
        metrics -- a json dictionary that contains evaluations of the model's performance along the training (test loss for each epoch, etc)
        run -- a lot of meta information about this run (duration, hardware, git commit header, even the code itself)
    """
    client = MongoClient(mongo_auth_str)
    db = client.sacred
    files = gridfs.GridFS(db)
    run = db.runs.find_one(model_id)
    config = run["config"]
    config["device"] = device
    metrics = {v["name"]: v for v in db.metrics.find({"run_id": run["_id"]})}
    model = PINODE(**config)
    background_model = PINODE(**{**config, "n_latent": 2})

    artifacts_ids = {s["name"]: s["file_id"] for s in run["artifacts"]}
    state_dict = torch.load(
        io.BytesIO(files.get(artifacts_ids["model"]).read()),
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(state_dict)
    state_dict = torch.load(
        io.BytesIO(files.get(artifacts_ids["background"]).read()),
        map_location=torch.device("cpu"),
    )
    background_model.load_state_dict(state_dict)

    return (model, background_model), config, metrics, run

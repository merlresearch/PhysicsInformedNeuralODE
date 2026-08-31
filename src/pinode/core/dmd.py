# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np


class DMD:

    def __init__(self, r, Phi=None, omega=None, A=None, Ar=None, Ur=None):
        self.r = r
        self.Phi = Phi
        self.omega = omega
        self.A = A
        self.Ar = Ar
        self.Ur = Ur

    def fit(self, u_snapshots, dt, modes="exact"):
        """
        Computes the DMD of X1,X2

        Args:
            X1,X2: 2D arrays
                Data matrices with columns representing state snapshots, and the columns
                of X2 are shifted in time with respect to those of X1
            r: int
                Target rank of SVD
            dt: double
                Time step advancing X1 to X2
            modes: string
                Whether to return exact ('exact') or projected ('projX1' on POD of
                X1, 'projX2' on POD of X2) DMD modes
                in Phi

        Returns:
            Phi: 2D array
                DMD modes
            omega: 1D array
                Continuous-time DMD eigenvalues
            A: 2D array
                Full-order (rank r) DMD discrete-time linear operator
            Ar: 2D array
                Reduced-order (rank r) DMD discrete-time linear operator
            Ur: 2D array
                POD modes
        """
        batch_size, time_len, _ = u_snapshots.shape
        X1 = (
            u_snapshots[:, :-1]
            .reshape(batch_size * (time_len - 1), -1)
            .T.cpu()
            .detach()
            .numpy()
        )
        X2 = (
            u_snapshots[:, 1:]
            .reshape(batch_size * (time_len - 1), -1)
            .T.cpu()
            .detach()
            .numpy()
        )

        # Truncated SVD of X1 data
        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
        V = Vh.conj().T
        r = min(self.r, U.shape[1])
        Ur, sr, Vr = U[:, 0:r], s[0:r], V[:, 0:r]

        # Truncated SVD of X2 data
        if modes == "projX2":
            U2, s2, V2h = np.linalg.svd(X2, full_matrices=False)
            V2 = V2h.conj().T
            U2r, s2r, V2r = U2[:, 0:r], s2[0:r], V2[:, 0:r]

        # Full linear operator
        A = X2 @ Vr @ np.diag(1 / sr) @ Ur.conj().T

        # Projected linear operator
        if modes == "exact" or modes == "projX1":
            Ar = Ur.conj().T @ X2 @ Vr @ np.diag(1 / sr)
        elif modes == "projX2":
            Ar = U2r.conj().T @ X2 @ Vr @ np.diag(1 / sr) @ Ur.conj().T @ U2r

        # DMD modes and eigenvalues
        lam, Wr = np.linalg.eig(Ar)
        if modes == "exact":
            Phi = X2 @ Vr @ np.diag(1 / sr) @ Wr
        elif modes == "projX1":
            Phi = Ur @ Wr
        elif modes == "projX2":
            Phi = U2r @ Wr
            Ur = U2r
        else:
            raise ValueError("Wrong value for 'modes' argument.")

        omega = np.log(lam) / dt

        self.Phi = Phi
        self.omega = omega
        self.A = A
        self.Ar = Ar
        self.Ur = Ur

        return self

    def state_dict(self):
        return {
            "r": self.r,
            "Phi": self.Phi,
            "omega": self.omega,
            "A": self.A,
            "Ar": self.Ar,
            "Ur": self.Ur,
        }

    def predict(self, x0, t_eval):
        return np.real(self.DMD_Prediction_From_Ar(x0, t_eval).T)

    def DMD_Prediction(self, x0, t_eval):
        """
        Computes predictions of the states based on DMD modes and eigenvalues

        Args:
            x0: 1D array
                Initial state
            t_eval: 1D array
                Times values (wrt x0) at which to predict future state
            Phi: 2D array
                DMD modes
            omega: 1D array
                Continuous-time DMD eigenvalues

        Returns:
            XDMD: 2D array
                DMD prediction for states at times in t_eval
        """

        # DMD modes amplitudes
        b0 = np.linalg.pinv(self.Phi) @ x0

        # DMD prediction
        B = np.zeros((b0.shape[0], len(t_eval)), dtype=complex)
        for i, t in enumerate(t_eval):
            B[:, i] = b0 * np.exp(self.omega * t)
        XDMD = self.Phi @ B

        return XDMD

    def DMD_Prediction_From_A(self, x0, t_eval):
        """
        Computes predictions of the states based on DMD modes and eigenvalues
        using the full-order discrete-time linear operator A

        Args:
            x0: 1D arraz
                Initial state
            t_eval: 1D array
                Times values (wrt x0) at which to predict future state, must be
                spaced with the same time step used in the data for computing A
            A: 2D array
                Full-order DMD discrete-time linear operator

        Returns:
            XDMD: 2D array
                DMD prediction for states at times in t_eval
        """

        # DMD prediction
        XDMD = np.zeros((x0.shape[0], len(t_eval)), dtype=complex)
        if t_eval[0] == 0:
            XDMD[:, 0] = x0
        else:
            XDMD[:, 0] = self.A @ x0

        for i in range(1, len(t_eval)):
            XDMD[:, i] = self.A @ XDMD[:, i - 1]

        return XDMD

    def DMD_Prediction_From_Ar(self, x0, t_eval):
        """
        Computes predictions of the states based on DMD modes and eigenvalues
        using the reduced-order discrete-time linear operator Ar

        Args:
            x0: 1D arraz
                Initial state
            t_eval: 1D array
                Times values (wrt x0) at which to predict future state, must be
                spaced with the same time step used in the data for computing A
            Ar: 2D array
                Reduced-order DMD discrete-time linear operator
            Ur: 2D array
                POD modes

        Returns:
            XDMD: 2D array
                DMD prediction for states at times in t_eval
        """

        # Initial reduced state
        x0r = self.Ur.conj().T @ x0

        # DMD prediction in reduced-order subspace
        XrDMD = np.zeros((x0r.shape[0], len(t_eval)), dtype=complex)
        if t_eval[0] == 0:
            XrDMD[:, 0] = x0r
        else:
            XrDMD[:, 0] = self.Ar @ x0r

        for i in range(1, len(t_eval)):
            XrDMD[:, i] = self.Ar @ XrDMD[:, i - 1]

        # Reconstruct DMD prediction in full-order space
        XDMD = self.Ur @ XrDMD

        return XDMD

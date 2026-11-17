"""PairDiffNet: side-symmetric spectral network on the left/right log-impedance ratio.

input  x = [log|Za/Zb|, angle(Za/Zb)]  shape [B, 2, F, M]   (F frequencies, M measurements)
1. noise whitening: x / s, s = per-feature std of lesion-free pairs under the training nuisance model (buffer)
2. spectral encoder shared by all measurements: two Conv1d(k=3) layers along frequency -> token per measurement
3. concat tokens -> Linear -> residual MLP blocks (pre-LayerNorm, GELU, dropout); plus a log-energy summary of
   each measurement (an explicit energy-detector feature)
4. side symmetrisation: logit(x) = (g(x) + g(-x)) / 2, so swapping left and right never changes the output
"""
from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, d, expand=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, expand * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(expand * d, d), nn.Dropout(dropout))

    def forward(self, h):
        return h + self.net(h)


class PairDiffNet(nn.Module):
    def __init__(self, n_freq, n_meas, conv_channels=16, hidden=64, n_blocks=2, dropout=0.1, symmetrize=True,
                 energy_features=True):
        super().__init__()
        self.n_freq, self.n_meas, self.symmetrize, self.energy_features = n_freq, n_meas, symmetrize, energy_features
        self.register_buffer("feature_scale", torch.ones(2, n_freq, n_meas))
        C = conv_channels
        self.spectral = nn.Sequential(nn.Conv1d(2, C, 3, padding=1), nn.GELU(), nn.Conv1d(C, C, 3, padding=1),
                                      nn.GELU())
        self.inp = nn.Linear(n_meas * C * n_freq, hidden)
        self.energy = nn.Linear(2 * n_meas, hidden) if energy_features else None
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def set_feature_scale(self, s):
        self.feature_scale.copy_(torch.as_tensor(s, dtype=self.feature_scale.dtype))

    def _branch(self, x):
        B, _, F, M = x.shape
        tok = self.spectral(x.permute(0, 3, 1, 2).reshape(B * M, 2, F)).reshape(B, -1)
        h = self.inp(tok)
        if self.energy is not None:
            h = h + self.energy(torch.log(x.pow(2).mean(2) + 1e-2).reshape(B, -1))
        return self.head(self.blocks(h)).squeeze(-1)

    def forward(self, x):
        x = x / self.feature_scale
        if not self.symmetrize:
            return self._branch(x)
        return 0.5 * (self._branch(x) + self._branch(-x))

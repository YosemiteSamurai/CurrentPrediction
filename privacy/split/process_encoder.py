# =============================================================================
# process_encoder.py -- Foundry-side encoder for the split-learning pipeline
#
# A small private MLP that maps raw BSIM4 process parameters (the foundry's
# IP) to a fixed-size embedding (default 16 floats per device type). Lives
# inside Foundry; the design-house side never imports this module directly.
#
# Architecture (defaults):
#   pmos_mlp:  R^15 -> Linear(64) -> ReLU -> Linear(16)
#   nmos_mlp:  R^18 -> Linear(64) -> ReLU -> Linear(16)
#
# The PMOS / NMOS field counts match the slices used by models.block_2inv
# (pFields and nFields). Embedding dim is configurable so a privacy/utility
# sweep against the GAT downstream can vary it.
# =============================================================================

import torch
import torch.nn as nn


class ProcessEncoder(nn.Module):

    def __init__(self, n_pmos_params=15, n_nmos_params=18,
                 embed_dim=16, hidden=64):

        super().__init__()
        self.n_pmos_params = n_pmos_params
        self.n_nmos_params = n_nmos_params
        self.embed_dim = embed_dim
        self.hidden = hidden

        self.pmos_mlp = nn.Sequential(
            nn.Linear(n_pmos_params, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

        self.nmos_mlp = nn.Sequential(
            nn.Linear(n_nmos_params, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, raw_pmos: torch.Tensor, raw_nmos: torch.Tensor):
        """Returns (z_pmos, z_nmos), each shaped (batch, embed_dim).

        raw_pmos:  (batch, n_pmos_params)
        raw_nmos:  (batch, n_nmos_params)
        """
        return self.pmos_mlp(raw_pmos), self.nmos_mlp(raw_nmos)

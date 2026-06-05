# =============================================================================
# foundry.py -- Foundry-side container for the split-learning cut layer
#
# In the split-learning architecture (Vepakomma 2018, vertical configuration;
# SplitGNN 2023 for the attention-based GNN equivalent), the foundry holds
# raw BSIM4 process parameters and a small private encoder. The design house
# never sees the raw parameters or the encoder weights -- only the embedding
# vectors produced at the cut.
#
# This class enforces that boundary in code:
#   - Raw BSIM4 tensors live inside `_process_table` and are never exposed.
#   - `encode_batch` returns ONLY detached embedding tensors (z_p, z_n) that
#     the design-house code can build into its node feature matrix.
#   - `backward` accepts the gradients of those embeddings and applies them
#     internally, then steps the encoder optimizer.
#   - `state_dict` returns the foundry's checkpoint payload (encoder weights
#     + scalers + raw process table) -- this whole blob is meant to be saved
#     under foundry control, not handed to the design house.
#
# DP-SGD handoff point: the single line in `__init__` that constructs `self.opt_encoder`
# is meant to be swapped for `PrivacyEngine.make_private_with_epsilon(...)`.
#
# Single-process optimization: when all rows in a batch share the same
# process (the v2.0 default), `encode_batch` runs the encoder once and
# broadcasts the embedding -- the cut-layer traffic is one (z_p, z_n) pair
# per training step rather than one per sample.
# =============================================================================

import numpy as np
import torch

from process_encoder import ProcessEncoder


class Foundry:

    def __init__(self, encoder: ProcessEncoder, process_table: dict,
                 pmos_scaler=None, nmos_scaler=None, lr: float = 1e-4,
                 device=None, dp_enabled: bool = False,
                 dp_noise_multiplier: float = 0.0,
                 dp_max_grad_norm: float = 1.0):
        """
        encoder:        ProcessEncoder instance (created on the foundry side).
        process_table:  {process_name: (raw_pmos: np.ndarray, raw_nmos: np.ndarray)}
                        where raw_* arrays are already standardised. Each
                        array is 1-D of length n_pmos_params / n_nmos_params.
        pmos_scaler,
        nmos_scaler:    sklearn StandardScalers used to produce the values in
                        process_table. Saved in the foundry's state dict so
                        that new process parameters can be scaled at
                        inference time. They never leave this class.
        lr:             learning rate for the encoder optimizer.
        device:         torch device for the encoder + cached tensors.
        """

        self.device = (device if device is not None
                       else torch.device('cuda:0' if torch.cuda.is_available()
                                         else 'cpu'))
        self.encoder = encoder.to(self.device)
        self.dp_enabled = bool(dp_enabled)
        self.dp_noise_multiplier = float(dp_noise_multiplier)
        self.dp_max_grad_norm = float(dp_max_grad_norm)

        self._pmos_scaler = pmos_scaler
        self._nmos_scaler = nmos_scaler

        # Cache process_table as torch tensors on the right device so we
        # don't pay the np.array -> torch.tensor conversion on every batch.
        self._process_table = {}

        for name, (raw_p, raw_n) in process_table.items():
            self._process_table[name] = (
                torch.as_tensor(np.asarray(raw_p, dtype=np.float32),
                                device=self.device),
                torch.as_tensor(np.asarray(raw_n, dtype=np.float32),
                                device=self.device),
            )

        # ---- DP-SGD swap point ---------------------------------------------
        # Replace this Adam with a privacy-engine-wrapped equivalent to add
        # per-sample-clipped + Gaussian-noised gradient updates without
        # touching anything outside Foundry.
        self.opt_encoder = torch.optim.Adam(
            self.encoder.parameters(), lr=lr)
        # --------------------------------------------------------------------

        # Cached non-detached encoder outputs from the most recent
        # encode_batch(). Kept so that backward() can push gradients of
        # the cut-layer activations through the encoder graph.
        self._last_z_p = None
        self._last_z_n = None

    @torch.enable_grad()
    def encode_batch(self, process_names):
        """Forward pass across the cut.

        Returns (z_p_send, z_n_send), each shaped (batch_size, embed_dim),
        detached + requires_grad=True so the design-house side can build
        them into its compute graph and produce gradients for them.
        """

        if isinstance(process_names, str):
            process_names = [process_names]

        # Stack the raw BSIM4 vectors for each sample in the batch.
        raw_p = torch.stack(
            [self._process_table[n][0] for n in process_names], dim=0)
        raw_n = torch.stack(
            [self._process_table[n][1] for n in process_names], dim=0)

        # Run the private encoder.
        self.encoder.train()
        z_p, z_n = self.encoder(raw_p, raw_n)

        # Cache pre-detach handles so backward() can push gradients into
        # the encoder graph.
        self._last_z_p = z_p
        self._last_z_n = z_n

        # Cross the cut: detach + grant the design-house side autograd.
        z_p_send = z_p.detach().clone().requires_grad_(True)
        z_n_send = z_n.detach().clone().requires_grad_(True)

        return z_p_send, z_n_send

    def backward(self, z_p_send: torch.Tensor, z_n_send: torch.Tensor):
        """Backward pass across the cut.

        After the design-house side has called loss.backward(), the .grad
        on z_p_send / z_n_send holds the cut-layer gradients. Push them
        back into the encoder graph and step the encoder optimizer.
        """

        if self._last_z_p is None or self._last_z_n is None:
            raise RuntimeError(
                "Foundry.backward called without a prior encode_batch")

        if z_p_send.grad is None or z_n_send.grad is None:
            # No gradient flowed back across the cut; nothing to update.
            self._last_z_p = None
            self._last_z_n = None
            return

        self.opt_encoder.zero_grad()
        torch.autograd.backward(
            [self._last_z_p, self._last_z_n],
            grad_tensors=[z_p_send.grad, z_n_send.grad],
        )

        if self.dp_enabled:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.dp_max_grad_norm)
            if self.dp_noise_multiplier > 0:
                for p in self.encoder.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * (self.dp_noise_multiplier * self.dp_max_grad_norm))

        self.opt_encoder.step()

        self._last_z_p = None
        self._last_z_n = None

    @torch.no_grad()
    def encode_for_inference(self, process_name: str):
        """Run the encoder once with grad disabled. Used by predict.py."""

        self.encoder.eval()
        raw_p, raw_n = self._process_table[process_name]
        z_p, z_n = self.encoder(raw_p.unsqueeze(0), raw_n.unsqueeze(0))
        return z_p[0], z_n[0]

    def add_process(self, process_name: str, raw_pmos_unscaled,
                    raw_nmos_unscaled):
        """Register a new process post-init. Scales the raw vectors with
        the foundry's private scalers and stores them. Used at inference
        time when a new .pm file is delivered after training."""

        if self._pmos_scaler is None or self._nmos_scaler is None:
            raise RuntimeError(
                "Cannot add a new process without pmos/nmos scalers")

        raw_p = np.asarray(raw_pmos_unscaled, dtype=np.float32).reshape(1, -1)
        raw_n = np.asarray(raw_nmos_unscaled, dtype=np.float32).reshape(1, -1)
        scaled_p = self._pmos_scaler.transform(raw_p)[0]
        scaled_n = self._nmos_scaler.transform(raw_n)[0]
        self._process_table[process_name] = (
            torch.as_tensor(scaled_p, dtype=torch.float32, device=self.device),
            torch.as_tensor(scaled_n, dtype=torch.float32, device=self.device),
        )

    def state_dict(self):
        """Foundry-private checkpoint payload.

        This dict contains the encoder weights, scalers, and process
        table. It is meant to be persisted under foundry control. The
        design-house checkpoint should NOT include this blob in production;
        for the in-process v3.0 path we still bundle it for convenience.
        """

        # Move tensors back to CPU so the checkpoint is portable.
        cpu_table = {
            name: (p.detach().cpu().numpy(), n.detach().cpu().numpy())
            for name, (p, n) in self._process_table.items()
        }
        return {
            'encoder_state_dict': self.encoder.state_dict(),
            'pmos_scaler': self._pmos_scaler,
            'nmos_scaler': self._nmos_scaler,
            'process_table': cpu_table,
            'embed_dim': self.encoder.embed_dim,
            'n_pmos_params': self.encoder.n_pmos_params,
            'n_nmos_params': self.encoder.n_nmos_params,
            'hidden': self.encoder.hidden,
        }

    @classmethod
    def from_state_dict(cls, sd: dict, lr: float = 1e-4, device=None):
        """Rebuild a Foundry from a state_dict produced by .state_dict()."""

        encoder = ProcessEncoder(
            n_pmos_params=sd['n_pmos_params'],
            n_nmos_params=sd['n_nmos_params'],
            embed_dim=sd['embed_dim'],
            hidden=sd['hidden'],
        )
        encoder.load_state_dict(sd['encoder_state_dict'])
        return cls(
            encoder=encoder,
            process_table=sd['process_table'],
            pmos_scaler=sd.get('pmos_scaler'),
            nmos_scaler=sd.get('nmos_scaler'),
            lr=lr,
            device=device,
        )

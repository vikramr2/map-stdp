"""
Shared SNN forward-pass and training infrastructure for both Map-STDP variants.

Architecture
------------
* 2312 input neurons (34×34×2 N-MNIST DVS channels) connected to community 0.
* 1100 recurrent LIF neurons arranged in 11 communities:
    community 0  – central workspace (100 neurons, receives all input)
    community 1-10 – classification communities (100 neurons each, one per digit)
* Readout: mean spike count per classification community → argmax → digit class.

LIF dynamics (discrete, 1 ms steps)
-------------------------------------
    V[t] = β·V[t-1]·(1 - spk[t-1])          # decay + hard reset
          + W_rec @ spk[t-1]                  # recurrent input
          + W_in  @ x[t]                      # feedforward input
    spk[t] = (V[t] > θ).float()

where β = exp(-dt/τ_m).

STDP and Map-STDP updates are applied *after* each sample presentation.
The E-step (p̃ update) runs online inside the forward pass.
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Resolve models/ directory regardless of cwd
_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from stdp import SuppressionSTDP
from network import (
    load_network,
    make_cross_community_mask,
    make_input_weights,
    update_p_tilde,
    compute_W_tilde_and_s,
    decode_community_rates,
)


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_HP = dict(
    # LIF
    tau_m   = 20.0,    # membrane time constant (ms)
    theta   = 0.5,     # spike threshold
    # With degree-normalised weights, per-neuron input ≈ mean_w ≈ 0.5/(init range)
    # Tuning note: lower theta → more activity; raise if neurons over-fire.
    dt      = 1.0,     # simulation timestep (ms)

    # STDP  (multiplier on the ±1/15 bit step)
    eta_stdp = 1.0,

    # Map-STDP
    eta_map  = 0.005,  # Map gating learning rate (η') — lower than STDP rate
    beta_p   = 0.01,   # E-step momentum for p̃ update

    # Weight bounds (keep in [0,1] for all weights)
    w_min = 0.0,
    w_max = 1.0,

    # Weight initialization (applied in load_network + make_input_weights)
    # Degree-normalized recurrent weights: raw range * (1/avg_degree) ≈ 0.3/69 ≈ 0.004
    # Input weights to workspace community: raw range [0.1, 0.5] / (N_input * sparsity)
    w_in_init_lo = 0.0005,   # W_in per-synapse init low  (2312 afferents → each ~0.001)
    w_in_init_hi = 0.002,    # W_in per-synapse init high

    # Training
    n_epochs  = 10,
    max_t     = 300,   # max timesteps per sample (truncate long events)
    eval_every = 1,    # evaluate after this many epochs
    log_every  = 200,  # print training loss every N samples
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_dataloaders(data_root: str, time_window_us: int = 1000):
    """
    Returns (train_loader, test_loader) over tonic N-MNIST.

    Each item: (frames, label)
      frames shape: (T, 2, 34, 34)  — T varies per sample
    """
    import tonic
    import tonic.transforms as transforms
    from torch.utils.data import DataLoader
    import tonic.collation as collation

    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_tf = transforms.Compose([
        transforms.Denoise(filter_time=10000),          # drop isolated noise events
        transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=time_window_us,                 # 1 ms bins
        ),
    ])

    train_ds = tonic.datasets.NMNIST(
        save_to=data_root, transform=frame_tf, train=True
    )
    test_ds = tonic.datasets.NMNIST(
        save_to=data_root, transform=frame_tf, train=False
    )

    # Pad variable-length sequences in a batch to the same T
    def collate_fn(batch):
        frames_list, labels = zip(*batch)
        # frames_list: list of (T_i, 2, 34, 34) numpy arrays
        import numpy as np
        T_max = max(f.shape[0] for f in frames_list)
        padded = np.zeros((len(frames_list), T_max, 2, 34, 34), dtype=np.float32)
        for i, f in enumerate(frames_list):
            padded[i, :f.shape[0]] = f
        frames_t = torch.from_numpy(padded)  # (B, T, 2, 34, 34)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return frames_t, labels_t

    # Use batch_size=1 for online STDP (one sample at a time)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Base SNN class
# ---------------------------------------------------------------------------

class MapSTDPBase:
    """
    Base class shared by the classic Map-STDP and the LI variant.

    Sub-classes override `compute_map_gating()` to supply their G matrix.
    """

    N_INPUT    = 2312          # 34*34*2 DVS channels
    N_NEURONS  = 1100          # total recurrent neurons
    N_CLASSES  = 10            # digit classes → communities 1-10
    N_WORKSPACE_NEURONS = 100  # nodes in community 0

    def __init__(
        self,
        workspace_root: str,
        hp: dict | None = None,
        device: torch.device | None = None,
    ):
        self.hp = {**DEFAULT_HP, **(hp or {})}
        self.device = device or torch.device("cpu")
        self.beta = torch.exp(
            torch.tensor(-self.hp["dt"] / self.hp["tau_m"])
        ).item()

        # ---- load topology -----------------------------------------------
        wroot = Path(workspace_root)
        self.W, self.mask, self.community = load_network(
            communities_csv=str(wroot / "nmnist_starting_communities.csv"),
            edgelist_csv=str(wroot / "nmnist_starting_edgelist.csv"),
            device=self.device,
        )

        # ---- cross-community mask for G_sim (standard Map-STDP) ----------
        self.cross_mask = make_cross_community_mask(self.community)  # (N,N) bool

        # ---- input weights (workspace community = community 0) -----------
        workspace_idx = (self.community == 0).nonzero(as_tuple=True)[0]  # (100,)
        self.workspace_idx = workspace_idx
        self.W_in = make_input_weights(
            n_input=self.N_INPUT,
            n_workspace=len(workspace_idx),
            device=self.device,
            init_lo=self.hp.get("w_in_init_lo", 0.0005),
            init_hi=self.hp.get("w_in_init_hi", 0.002),
        )  # (100, 2312)

        # full input mask for STDP: (N_INPUT,) → workspace only
        self.W_in_mask = torch.ones_like(self.W_in, dtype=torch.bool)

        # ---- STDP tracker ------------------------------------------------
        # We concatenate [recurrent neurons | input neurons] so the STDP
        # tracker can handle both sets with a single t_last array.
        # Build combined connectivity mask (N+N_in, N+N_in):
        N    = self.N_NEURONS
        N_in = self.N_INPUT
        N_tot = N + N_in
        full_mask = torch.zeros(N_tot, N_tot, dtype=torch.bool, device=self.device)
        # Recurrent block
        full_mask[:N, :N] = self.mask
        # Input→workspace block: full_mask[workspace_idx, N:]
        full_mask[workspace_idx, N:] = True

        self.stdp = SuppressionSTDP(
            n_neurons=N,
            mask=full_mask,
            dt_ms=self.hp["dt"],
            device=self.device,
            input_n=N_in,
        )

        # ---- stationary distribution estimate ----------------------------
        self.p_tilde = torch.ones(N, device=self.device) / N

        # ---- membrane state (reset at each sample) ----------------------
        self.V   = torch.zeros(N, device=self.device)
        self.spk = torch.zeros(N, device=self.device)

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.V.zero_()
        self.spk.zero_()
        self.stdp.reset()
        self.p_tilde = torch.ones(self.N_NEURONS, device=self.device) / self.N_NEURONS

    # ------------------------------------------------------------------
    def _forward_step(self, x_t: torch.Tensor, t: int):
        """
        One LIF timestep.
        x_t : (N_INPUT,) binary input spike vector
        Returns spk : (N_NEURONS,) binary output spike vector
        """
        N = self.N_NEURONS
        # --- membrane dynamics ---
        # Input to workspace neurons from feedforward
        inp = torch.zeros(N, device=self.device)
        inp[self.workspace_idx] = self.W_in @ x_t   # (100,)

        self.V = (
            self.beta * self.V * (1.0 - self.spk)   # decay + reset
            + self.W @ self.spk                      # recurrent
            + inp                                    # feedforward
        )
        self.spk = (self.V > self.hp["theta"]).float()

        # --- E-step: update p̃ ---
        self.p_tilde = update_p_tilde(
            self.p_tilde, self.W, self.spk, beta=self.hp["beta_p"]
        )

        # --- STDP: concatenate [recurrent | input] spike vectors ---
        z_full = torch.cat([self.spk, x_t.float()])  # (N+N_in,)
        self.stdp.step(z_full, t)

        return self.spk

    # ------------------------------------------------------------------
    def run_sample(
        self,
        frames: torch.Tensor,   # (T, 2, 34, 34) float, values ∈ {0,1}
        learn: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate SNN for one sample.

        Returns:
            spike_counts : (N_NEURONS,)  total spikes per neuron
            dW_stdp      : (N_tot, N_tot) accumulated STDP ΔW (before gating)
        """
        self._reset_state()
        T = min(frames.shape[0], self.hp["max_t"])
        spike_counts = torch.zeros(self.N_NEURONS, device=self.device)

        for t in range(T):
            # Flatten frame: (2, 34, 34) → (2312,) binary spikes
            x_t = (frames[t] > 0).float().to(self.device).reshape(-1)
            spk = self._forward_step(x_t, t)
            spike_counts += spk

        dW_stdp = self.stdp.get_and_reset_dW()
        return spike_counts, dW_stdp

    # ------------------------------------------------------------------
    def compute_map_gating(self) -> torch.Tensor:
        """To be overridden.  Return G (N,N) gating matrix."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def apply_weight_update(self, dW_stdp: torch.Tensor):
        """
        M-step: apply STDP + Map gating to weights.

        ΔW = η·STDP - η'·G   (Eq. 1 from Map-STDP paper)
        Applied to recurrent block and input block separately.
        """
        N = self.N_NEURONS
        N_in = self.N_INPUT
        eta  = self.hp["eta_stdp"]
        eta_p = self.hp["eta_map"]

        # --- Recurrent weights ---
        G = self.compute_map_gating()                    # (N,N)
        dW_rec = eta * dW_stdp[:N, :N] - eta_p * G
        dW_rec = dW_rec * self.mask.float()
        self.W = (self.W + dW_rec).clamp(
            self.hp["w_min"], self.hp["w_max"]
        )

        # --- Input weights (no map gating, pure suppression STDP) ---
        # dW_stdp block: rows=workspace_idx, cols=N:N+N_in
        dW_in = eta * dW_stdp[self.workspace_idx, N:]   # (100, N_in)
        self.W_in = (self.W_in + dW_in).clamp(
            self.hp["w_min"], self.hp["w_max"]
        )

    # ------------------------------------------------------------------
    def predict(self, spike_counts: torch.Tensor) -> int:
        """Decode community spike rates → predicted digit (0-9)."""
        rates = decode_community_rates(spike_counts, self.community)
        return int(rates.argmax().item())

    # ------------------------------------------------------------------
    def train_epoch(self, loader, epoch: int) -> float:
        """Train for one epoch, return mean loss (1 - accuracy proxy)."""
        correct = 0
        total   = 0
        for i, (frames_batch, labels_batch) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        ):
            frames = frames_batch[0].to(self.device)  # (T, 2, 34, 34)
            label  = labels_batch[0].item()

            spike_counts, dW_stdp = self.run_sample(frames, learn=True)
            self.apply_weight_update(dW_stdp)

            pred = self.predict(spike_counts)
            correct += int(pred == label)
            total   += 1

            if self.hp["log_every"] and (i + 1) % self.hp["log_every"] == 0:
                acc = correct / total
                print(f"  [{i+1}/{len(loader)}] running acc = {acc:.3f}")

        return 1.0 - correct / max(total, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader) -> float:
        """Evaluate on loader, return accuracy."""
        correct = 0
        total   = 0
        for frames_batch, labels_batch in tqdm(loader, desc="Eval", leave=False):
            frames = frames_batch[0].to(self.device)
            label  = labels_batch[0].item()
            spike_counts, _ = self.run_sample(frames, learn=False)
            pred = self.predict(spike_counts)
            correct += int(pred == label)
            total   += 1
        return correct / max(total, 1)

    # ------------------------------------------------------------------
    def fit(self, workspace_root: str, data_root: str, save_dir: str | None = None):
        """Full training loop."""
        train_loader, test_loader = make_dataloaders(data_root)
        save_path = Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        hp = self.hp
        for epoch in range(1, hp["n_epochs"] + 1):
            train_loss = self.train_epoch(train_loader, epoch)

            if epoch % hp["eval_every"] == 0:
                acc = self.evaluate(test_loader)
                print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | test_acc={acc:.4f}")
            else:
                print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f}")

            if save_path:
                torch.save(
                    {"W": self.W, "W_in": self.W_in, "p_tilde": self.p_tilde},
                    save_path / f"checkpoint_epoch{epoch:03d}.pt",
                )

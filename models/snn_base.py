"""
Shared SNN forward-pass and training infrastructure for both Map-STDP variants.

Performance optimisations applied here
---------------------------------------
1.  Sparse forward pass   — instead of the dense  W @ spk  (N² ops), we compute
    W_T[active].sum(0)  where W_T = W.T.contiguous().  With ~30 Hz firing and N=1100,
    active ≈ 33 neurons per step → ~33× fewer operations.

2.  Cached W̃ for E-step  — W̃ = W * p̃ and s̃ = W̃.sum(1) are computed once at the
    start of each sample and refreshed every CACHE_INTERVAL steps.  Within each
    window we then do the sparse  (T̃ @ z)[i] ≈ W̃_T[active].sum(0) / s̃  instead
    of the full (N×N) matmul.

3.  Edge-indexed STDP     — SuppressionSTDP now works on flat edge arrays (E_tot ≈ 307 K)
    rather than (N_tot)² ≈ 11 M dense outer products per step.

4.  Sparse weight update  — M-step scatter-adds into W[rec_post, rec_pre] rather
    than materialising a full (N×N) ΔW matrix.

LIF dynamics (discrete, 1 ms steps)
-------------------------------------
    V[t] = β·V[t-1]·(1 − spk[t-1])   +   W_T[active]ᵀ.sum()   +   inp_ff
    spk[t] = (V[t] > θ).float()

where β = exp(−dt/τ_m) and inp_ff routes input spikes to workspace neurons only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm, trange

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from stdp import SuppressionSTDP
from network import (
    load_network,
    make_cross_community_mask,
    make_input_weights,
    compute_W_tilde_and_s,
    decode_community_rates,
)


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_HP = dict(
    # LIF
    tau_m  = 20.0,   # membrane time constant (ms)
    theta  = 0.5,    # spike threshold; lower → more activity
    dt     = 1.0,    # timestep (ms)

    # STDP multiplier on the ±1/15 bit-step
    eta_stdp = 1.0,

    # Map-STDP  (η' << η so structure pressure < pattern pressure)
    eta_map  = 0.005,
    beta_p   = 0.01,   # E-step momentum for p̃

    # Weight bounds
    w_min = 0.0,
    w_max = 1.0,

    # Input weight init (W_in; 2312 afferents → workspace community)
    w_in_init_lo = 0.0005,
    w_in_init_hi = 0.002,

    # Training
    n_epochs    = 10,
    max_t       = 300,   # max timesteps per sample
    max_samples = None,  # if set, cap each epoch to this many samples
    eval_every  = 1,
    log_every   = 500,   # samples between tqdm postfix refresh

    # E-step cache refresh interval (steps)
    cache_interval = 50,
)

# How often to print within an epoch (in samples)
_POSTFIX_EVERY = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_dataloaders(data_root: str, time_window_us: int = 1000):
    """
    Returns (train_loader, test_loader) over tonic N-MNIST.
    frames shape per sample: (T, 2, 34, 34), values are event counts.
    """
    import tonic
    import tonic.transforms as transforms
    from torch.utils.data import DataLoader

    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_tf = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=time_window_us),
    ])

    train_ds = tonic.datasets.NMNIST(save_to=data_root, transform=frame_tf, train=True)
    test_ds  = tonic.datasets.NMNIST(save_to=data_root, transform=frame_tf, train=False)

    def collate_fn(batch):
        import numpy as np
        frames_list, labels = zip(*batch)
        T_max  = max(f.shape[0] for f in frames_list)
        padded = np.zeros((len(frames_list), T_max, 2, 34, 34), dtype=np.float32)
        for i, f in enumerate(frames_list):
            padded[i, :f.shape[0]] = f
        return torch.from_numpy(padded), torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Base SNN
# ---------------------------------------------------------------------------

class MapSTDPBase:
    N_INPUT   = 2312   # 34*34*2 DVS channels (fixed by N-MNIST sensor)
    N_CLASSES = 10     # digit classes 0-9    (fixed by N-MNIST)
    # N_NEURONS and N_WS are set dynamically in __init__ from the loaded network.

    def __init__(
        self,
        workspace_root: str,
        hp: dict | None = None,
        device: torch.device | None = None,
    ):
        self.hp     = {**DEFAULT_HP, **(hp or {})}
        self.device = device or torch.device("cpu")
        self.beta   = float(torch.exp(
            torch.tensor(-self.hp["dt"] / self.hp["tau_m"])
        ))

        # ---- topology -------------------------------------------------------
        wroot = Path(workspace_root)
        self.W, self.mask, self.community = load_network(
            communities_csv=str(wroot / "nmnist_starting_communities.csv"),
            edgelist_csv=str(wroot / "nmnist_starting_edgelist.csv"),
            device=self.device,
        )
        self.cross_mask = make_cross_community_mask(self.community)

        # Derive dimensions from the loaded network (works for any node count)
        self.N_NEURONS = self.W.shape[0]
        self.workspace_idx = (self.community == 0).nonzero(as_tuple=True)[0]
        self.N_WS = self.workspace_idx.numel()
        n_ws = self.N_WS

        # ---- input weights (workspace community only) -----------------------
        self.W_in = make_input_weights(
            n_input=self.N_INPUT,
            n_workspace=n_ws,
            device=self.device,
            init_lo=self.hp["w_in_init_lo"],
            init_hi=self.hp["w_in_init_hi"],
        )  # (n_ws, N_INPUT)

        # ---- build edge arrays for STDP ------------------------------------
        rec_post, rec_pre = self.mask.nonzero(as_tuple=True)   # (E_rec,) each

        # Local index of workspace neurons inside the recurrent neuron array
        ws_local_idx = torch.arange(n_ws, device=self.device)  # community 0 = nodes [0,n_ws)
        # Confirm: the workspace nodes are contiguous starting at 0
        # (SBM generator assigns node i to community i//nodes_per_community)
        assert (self.workspace_idx == torch.arange(n_ws, device=self.device)).all(), \
            "workspace neurons must be nodes 0..n_ws-1"

        self.stdp = SuppressionSTDP(
            n_rec=self.N_NEURONS,
            n_in=self.N_INPUT,
            rec_post=rec_post,
            rec_pre=rec_pre,
            ws_local_idx=ws_local_idx,
            dt_ms=self.hp["dt"],
            device=self.device,
        )

        # ---- stationary distribution ----------------------------------------
        self.p_tilde = torch.ones(self.N_NEURONS, device=self.device) / self.N_NEURONS

        # ---- membrane state (reset per sample) ------------------------------
        self.V   = torch.zeros(self.N_NEURONS, device=self.device)
        self.spk = torch.zeros(self.N_NEURONS, device=self.device)

        # ---- transposed W for sparse column access --------------------------
        # W_T[j] = column j of W = all weights INTO neurons from neuron j
        self._refresh_W_T()

    # ------------------------------------------------------------------
    def _refresh_W_T(self):
        """Cache W.T.contiguous() so that column access = row access (fast)."""
        self.W_T = self.W.t().contiguous()

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.V.zero_()
        self.spk.zero_()
        self.stdp.reset()
        # Reset p̃ to uniform at the start of each sample
        self.p_tilde = torch.ones(self.N_NEURONS, device=self.device) / self.N_NEURONS

    # ------------------------------------------------------------------
    def _forward_step(
        self,
        x_t: torch.Tensor,               # (N_INPUT,) binary
        t: int,
        W_tilde_T: torch.Tensor,         # (N, N) = W_tilde.T.contiguous()
        s_tilde: torch.Tensor,           # (N,)
    ):
        """One LIF step.  Sparse in both the recurrent drive and the E-step."""
        N = self.N_NEURONS

        # ---- Sparse recurrent input ----------------------------------------
        # W_T[j] = j-th column of W = weights FROM pre=j TO all post neurons
        inp_rec = torch.zeros(N, device=self.device)
        active_rec = self.spk.nonzero(as_tuple=True)[0]
        if active_rec.numel() > 0:
            inp_rec = self.W_T[active_rec].sum(dim=0)   # (n_active, N).sum(0)

        # ---- Sparse feedforward input (workspace only) ----------------------
        inp_ff = torch.zeros(N, device=self.device)
        active_in = x_t.nonzero(as_tuple=True)[0]
        if active_in.numel() > 0:
            inp_ff[self.workspace_idx] = self.W_in[:, active_in].sum(dim=1)

        # ---- LIF dynamics ---------------------------------------------------
        self.V   = self.beta * self.V * (1.0 - self.spk) + inp_rec + inp_ff
        self.spk = (self.V > self.hp["theta"]).float()

        # ---- E-step: sparse T̃@z  (cached W̃) --------------------------------
        # (T̃ @ z)[i] = Σ_{j active} W̃_ij / s̃_i  = W̃_T[j active].sum(0) / s̃
        active_spk = self.spk.nonzero(as_tuple=True)[0]
        if active_spk.numel() > 0:
            T_z = W_tilde_T[active_spk].sum(dim=0) / s_tilde
        else:
            T_z = torch.zeros(N, device=self.device)

        beta_p   = self.hp["beta_p"]
        p_new    = (1.0 - beta_p) * self.p_tilde + beta_p * T_z
        p_new    = p_new.clamp(min=0.0)
        total    = p_new.sum()
        if total > 1e-10:
            p_new = p_new / total
        self.p_tilde = p_new

        # ---- STDP -----------------------------------------------------------
        self.stdp.step(self.spk, x_t, t)

        return self.spk

    # ------------------------------------------------------------------
    def run_sample(
        self,
        frames: torch.Tensor,   # (T, 2, 34, 34)
        learn: bool = True,
    ) -> torch.Tensor:
        """
        Simulate one sample.  Returns spike_counts (N_NEURONS,).
        STDP updates accumulate in self.stdp and are applied by apply_weight_update().
        """
        self._reset_state()
        T = min(frames.shape[0], self.hp["max_t"])
        spike_counts = torch.zeros(self.N_NEURONS, device=self.device)

        # Cache W̃ and s̃; refresh every CACHE_INTERVAL steps.
        # W doesn't change within a sample, so stale T̃ is a minor approximation.
        interval = self.hp["cache_interval"]
        W_tilde_T, s_tilde = self._make_W_tilde_cache()

        for t in range(T):
            if t > 0 and t % interval == 0:
                W_tilde_T, s_tilde = self._make_W_tilde_cache()

            x_t = (frames[t] > 0).float().to(self.device).reshape(-1)
            spk = self._forward_step(x_t, t, W_tilde_T, s_tilde)
            spike_counts += spk

        return spike_counts

    # ------------------------------------------------------------------
    def _make_W_tilde_cache(self):
        """Compute W̃ = W*p̃, s̃ = W̃.sum(1), return (W̃.T.contiguous(), s̃)."""
        W_tilde   = self.W * self.p_tilde.unsqueeze(0)          # (N,N)
        s_tilde   = W_tilde.sum(dim=1).clamp(min=1e-10)         # (N,)
        W_tilde_T = W_tilde.t().contiguous()                    # (N,N)
        return W_tilde_T, s_tilde

    # ------------------------------------------------------------------
    def compute_map_gating(self) -> torch.Tensor:
        """Override to return G (N,N).  Called once per sample in apply_weight_update."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def apply_weight_update(self):
        """
        M-step: ΔW = η·STDP − η'·G, applied sparsely onto existing edges.
        """
        N      = self.N_NEURONS
        eta    = self.hp["eta_stdp"]
        eta_p  = self.hp["eta_map"]
        w_min  = self.hp["w_min"]
        w_max  = self.hp["w_max"]

        # ---- Recurrent weights (edge-level scatter) ----------------------
        G           = self.compute_map_gating()                      # (N,N)
        G_edges     = G[self.stdp.rec_post, self.stdp.rec_pre]       # (E_rec,)
        dW_rec      = eta * self.stdp.dW_rec - eta_p * G_edges       # (E_rec,)

        self.W[self.stdp.rec_post, self.stdp.rec_pre] += dW_rec
        self.W.clamp_(w_min, w_max)
        self.W[~self.mask] = 0.0          # safety: clear any non-edge weights

        # Refresh W_T after recurrent weight update
        self._refresh_W_T()

        # ---- Input weights -----------------------------------------------
        # stdp.dW_in is already (n_ws, N_INPUT) — direct add, no reshape
        self.W_in = (self.W_in + eta * self.stdp.dW_in).clamp_(w_min, w_max)

        # Reset STDP accumulator for next sample
        self.stdp.reset_dW()

    # ------------------------------------------------------------------
    def predict(self, spike_counts: torch.Tensor) -> int:
        rates = decode_community_rates(spike_counts, self.community)
        return int(rates.argmax().item())

    # ------------------------------------------------------------------
    def train_epoch(self, loader, epoch: int) -> float:
        correct = 0
        total   = 0
        avg_spk = 0.0

        pbar = tqdm(
            loader, desc=f"Ep {epoch:2d}",
            leave=True, ncols=90,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {postfix}]"
            ),
        )

        for i, (frames_batch, labels_batch) in enumerate(pbar):
            frames = frames_batch[0].to(self.device)
            label  = int(labels_batch[0].item())

            spike_counts = self.run_sample(frames, learn=True)
            self.apply_weight_update()

            pred    = self.predict(spike_counts)
            correct += int(pred == label)
            total   += 1
            avg_spk  = 0.9 * avg_spk + 0.1 * spike_counts.mean().item()

            max_samples = self.hp.get("max_samples")
            if max_samples and total >= max_samples:
                break

            if (i + 1) % _POSTFIX_EVERY == 0:
                pbar.set_postfix(
                    acc=f"{correct/total:.3f}",
                    spk=f"{avg_spk:.1f}",
                    refresh=False,
                )

        pbar.set_postfix(acc=f"{correct/total:.3f}", spk=f"{avg_spk:.1f}")
        return 1.0 - correct / max(total, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader) -> float:
        correct = 0
        total   = 0
        for frames_batch, labels_batch in tqdm(loader, desc="  eval", leave=False, ncols=80):
            frames = frames_batch[0].to(self.device)
            label  = int(labels_batch[0].item())
            spike_counts = self.run_sample(frames, learn=False)
            correct += int(self.predict(spike_counts) == label)
            total   += 1
        return correct / max(total, 1)

    # ------------------------------------------------------------------
    def fit(self, workspace_root: str, data_root: str, save_dir: str | None = None):
        from pathlib import Path as _Path
        train_loader, test_loader = make_dataloaders(data_root)
        save_path = _Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        hp = self.hp
        epoch_bar = trange(
            1, hp["n_epochs"] + 1,
            desc="Training", leave=True, ncols=90,
        )
        for epoch in epoch_bar:
            train_loss = self.train_epoch(train_loader, epoch)

            if epoch % hp["eval_every"] == 0:
                acc = self.evaluate(test_loader)
                epoch_bar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{acc:.4f}")
                print(f"\nEpoch {epoch:3d} | train_loss={train_loss:.4f} | test_acc={acc:.4f}")
            else:
                epoch_bar.set_postfix(loss=f"{train_loss:.4f}")

            if save_path:
                torch.save(
                    {"W": self.W, "W_in": self.W_in, "p_tilde": self.p_tilde},
                    save_path / f"checkpoint_epoch{epoch:03d}.pt",
                )

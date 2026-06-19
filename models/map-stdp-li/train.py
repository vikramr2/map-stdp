"""
Map-STDP with Lateral Inhibition (LI) variant training script.

ΔW_ij   = η·STDP(i,j) − η'·G_LI(i,j)
ΔI_ij   = η_inh·p̃_i·p̃_j          (anti-Hebbian inhibitory update, Eq. end of Sec 4)

G_LI(i,j) = φ_ij − ē_i^LI          (Eq. 6, Map-STDP paper)

where:
  φ_ij    = I_ij·p̃_j / (Σ_k I_ik·p̃_k)
  e_i^LI  = Σ_j φ_ij·W_ij·p̃_j
  ē_i^LI  = e_i^LI / s̃_i

Key difference from classic Map-STDP: there is NO explicit community assignment.
The φ_ij competition score replaces the I[j ∉ m(i)] indicator — neurons that
strongly inhibit each other are treated as belonging to different communities.

Usage
-----
    conda run -n map-stdp python models/map-stdp-li/train.py \
        --workspace workspace/ \
        --data data/  \
        --epochs 10 \
        --save models/map-stdp-li/checkpoints/
"""

import sys
import argparse
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from snn_base import MapSTDPBase, DEFAULT_HP
from network import compute_W_tilde_and_s, compute_G_LI


# ---------------------------------------------------------------------------

LI_HP = {
    **DEFAULT_HP,
    "eta_inh":    0.005,   # learning rate for inhibitory weights
    "inh_init_lo": 0.01,   # initial inhibitory weight range
    "inh_init_hi": 0.05,
}


class MapSTDP_LI(MapSTDPBase):
    """
    Map-STDP with lateral inhibition, replacing explicit community membership
    with competitive inhibitory weights.

    Adds:
      I_inh (N,N) inhibitory weight matrix, updated anti-Hebbianly.
      G_LI gating function that uses φ_ij (inhibitory competition score).
    """

    def __init__(self, workspace_root: str, hp: dict | None = None,
                 device: torch.device | None = None):
        # Merge LI-specific defaults, but honour caller overrides
        merged_hp = {**LI_HP, **(hp or {})}
        super().__init__(workspace_root, hp=merged_hp, device=device)

        N = self.N_NEURONS
        lo = self.hp.get("inh_init_lo", 0.01)
        hi = self.hp.get("inh_init_hi", 0.05)
        # Inhibitory weights: neuron j → neuron i (same layout as W)
        # Only initialise for existing edges to keep locality
        self.I_inh = torch.zeros(N, N, device=self.device)
        self.I_inh[self.mask] = torch.empty(
            self.mask.sum().item(), device=self.device
        ).uniform_(lo, hi)

    # ------------------------------------------------------------------
    def compute_map_gating(self) -> torch.Tensor:
        W_tilde, s_tilde = compute_W_tilde_and_s(self.W, self.p_tilde)
        G_LI, _ = compute_G_LI(W_tilde, s_tilde, self.I_inh, self.p_tilde)
        return G_LI

    # ------------------------------------------------------------------
    def apply_weight_update(self, dW_stdp: torch.Tensor):
        """Extend base: also update inhibitory weights anti-Hebbianly."""
        super().apply_weight_update(dW_stdp)

        # ΔI_ij = η_inh · p̃_i · p̃_j
        eta_inh = self.hp.get("eta_inh", 0.005)
        dI = eta_inh * torch.outer(self.p_tilde, self.p_tilde)
        # Apply only on existing edges (locality), keep positive
        self.I_inh = (self.I_inh + dI * self.mask.float()).clamp(min=0.0)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Map-STDP with Lateral Inhibition")
    p.add_argument("--workspace", default="workspace/")
    p.add_argument("--data",      default="data/")
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--save",      default=None)
    p.add_argument("--eta-stdp",  type=float, default=LI_HP["eta_stdp"])
    p.add_argument("--eta-map",   type=float, default=LI_HP["eta_map"])
    p.add_argument("--eta-inh",   type=float, default=LI_HP["eta_inh"])
    p.add_argument("--tau-m",     type=float, default=LI_HP["tau_m"])
    p.add_argument("--theta",     type=float, default=LI_HP["theta"])
    p.add_argument("--beta-p",    type=float, default=LI_HP["beta_p"])
    p.add_argument("--max-t",     type=int,   default=LI_HP["max_t"])
    p.add_argument("--device",    default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    hp = {
        **LI_HP,
        "n_epochs":  args.epochs,
        "eta_stdp":  args.eta_stdp,
        "eta_map":   args.eta_map,
        "eta_inh":   args.eta_inh,
        "tau_m":     args.tau_m,
        "theta":     args.theta,
        "beta_p":    args.beta_p,
        "max_t":     args.max_t,
    }

    ws = Path(args.workspace)
    if not ws.is_absolute():
        ws = Path(__file__).resolve().parents[2] / ws

    model = MapSTDP_LI(
        workspace_root=str(ws),
        hp=hp,
        device=device,
    )

    print("=== Map-STDP (Lateral Inhibition) ===")
    print(f"  Neurons     : {model.N_NEURONS}")
    print(f"  Edges       : {int(model.mask.sum().item())}")
    print(f"  η_stdp      : {hp['eta_stdp']}")
    print(f"  η_map       : {hp['eta_map']}")
    print(f"  η_inh       : {hp['eta_inh']}")
    print(f"  τ_m         : {hp['tau_m']} ms")
    print(f"  θ           : {hp['theta']}")
    print(f"  β_p         : {hp['beta_p']}")
    print(f"  max_t       : {hp['max_t']}")
    print(f"  device      : {device}")
    print()

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parents[2] / data_root

    save_dir = None
    if args.save:
        save_dir = Path(args.save)
        if not save_dir.is_absolute():
            save_dir = Path(__file__).resolve().parents[2] / save_dir

    model.fit(
        workspace_root=str(ws),
        data_root=str(data_root),
        save_dir=str(save_dir) if save_dir else None,
    )


if __name__ == "__main__":
    main()

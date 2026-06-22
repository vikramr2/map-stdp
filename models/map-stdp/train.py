"""
Map-STDP (traditional variant) training script.

ΔW_ij = η·STDP(i,j) − η'·G_sim(i,j)

G_sim(i,j) = I[j ∉ m(i)] − ē_i       (Eq. 3, simplified Map-STDP paper)

where ē_i = e_i/s̃_i and e_i = Σ_{j'∉m(i)} W_ij'·p̃_{j'}.

Community assignments (M) are kept static from the SBM starting topology:
  Community 0  → workspace (receives N-MNIST input)
  Communities 1-10 → one per digit class

Usage
-----
    conda run -n map-stdp python models/map-stdp/train.py \
        --workspace workspace/ \
        --data data/  \
        --epochs 10 \
        --save models/map-stdp/checkpoints/
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import torch

# Make models/ importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from snn_base import MapSTDPBase, DEFAULT_HP
from network import compute_W_tilde_and_s, compute_G_sim


# ---------------------------------------------------------------------------

class MapSTDP(MapSTDPBase):
    """
    Classic Map-STDP with fixed community assignments.

    Gating function: G_sim(i,j) = I[j ∉ m(i)] − ē_i
    """

    def compute_map_gating(self) -> torch.Tensor:
        W_tilde, s_tilde = compute_W_tilde_and_s(self.W, self.p_tilde)
        return compute_G_sim(W_tilde, s_tilde, self.cross_mask)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Map-STDP (classic)")
    p.add_argument("--workspace", default="workspace/",
                   help="Directory with nmnist_starting_*.csv files")
    p.add_argument("--data",      default="data/",
                   help="Root for tonic N-MNIST download cache")
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--save",      default=None,
                   help="Directory to save checkpoints (optional)")
    p.add_argument("--eta-stdp",  type=float, default=DEFAULT_HP["eta_stdp"])
    p.add_argument("--eta-map",   type=float, default=DEFAULT_HP["eta_map"])
    p.add_argument("--tau-m",     type=float, default=DEFAULT_HP["tau_m"])
    p.add_argument("--theta",     type=float, default=DEFAULT_HP["theta"])
    p.add_argument("--beta-p",    type=float, default=DEFAULT_HP["beta_p"])
    p.add_argument("--max-t",       type=int,   default=DEFAULT_HP["max_t"])
    p.add_argument("--max-samples", type=int,   default=None,
                   help="Cap training samples per epoch (for quick smoke tests)")
    p.add_argument("--device",    default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    hp = {
        **DEFAULT_HP,
        "n_epochs":  args.epochs,
        "eta_stdp":  args.eta_stdp,
        "eta_map":   args.eta_map,
        "tau_m":     args.tau_m,
        "theta":     args.theta,
        "beta_p":    args.beta_p,
        "max_t":      args.max_t,
        "max_samples": args.max_samples,
    }

    # Resolve workspace path relative to repo root
    ws = Path(args.workspace)
    if not ws.is_absolute():
        ws = Path(__file__).resolve().parents[2] / ws

    model = MapSTDP(
        workspace_root=str(ws),
        hp=hp,
        device=device,
    )

    print("=== Map-STDP (classic) ===")
    print(f"  Neurons     : {model.N_NEURONS}")
    print(f"  Edges       : {int(model.mask.sum().item())}")
    print(f"  η_stdp      : {hp['eta_stdp']}")
    print(f"  η_map       : {hp['eta_map']}")
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

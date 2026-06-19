"""
Utilities for loading the SBM starting topology and computing Map-STDP quantities.

Network layout (nmnist_starting_*):
  - 1 100 nodes, 11 communities (100 nodes each)
  - Community 0  = central workspace module (receives all input)
  - Communities 1-10 = classification modules (one per N-MNIST digit class)
  - Edges are undirected → stored as both W[i,j] and W[j,i]

Map-STDP quantities (from the paper, Section 3):
  E-step  (online, per-timestep):
    p̃_i  ←  (1-β)·p̃_i  +  β · Σ_j T̃_ij · z_j(t)
    where  W̃_ij = W_ij · p̃_j,  s̃_i = Σ_k W_ik · p̃_k,  T̃_ij = W̃_ij / s̃_i

  M-step  (per-sample, Eq. 3 simplified):
    G_sim(i,j) = I[j ∉ m(i)] - ē_i
    where  e_i  = Σ_{j'∉m(i)} W_ij' · p̃_j',  ē_i = e_i / s̃_i

  Lateral-inhibition variant (Eq. 6):
    φ_ij    = I_ij · p̃_j / (Σ_k I_ik · p̃_k)
    e_i^LI  = Σ_j φ_ij · W_ij · p̃_j
    ē_i^LI  = e_i^LI / s̃_i
    G_LI(i,j) = φ_ij - ē_i^LI
"""

import csv
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_network(
    communities_csv: str,
    edgelist_csv: str,
    device: torch.device = None,
    weight_init: str = "uniform",   # "uniform" | "constant"
    init_lo: float = 0.3,
    init_hi: float = 0.7,
    normalize_by_degree: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load SBM topology.

    With normalize_by_degree=True (default), each neuron's incoming weights
    are divided by its in-degree so that the total synaptic drive at each
    timestep is in the same ballpark as the threshold regardless of connectivity.

    Returns:
        W        : (N, N) initial weight matrix (0 for non-edges)
        mask     : (N, N) boolean connectivity mask
        community: (N,)  integer community label for each node
    """
    device = device or torch.device("cpu")

    # --- community assignments ---
    community_map = {}
    with open(communities_csv) as f:
        for row in csv.DictReader(f):
            community_map[int(row["node"])] = int(row["community"])
    N = len(community_map)
    community = torch.tensor(
        [community_map[i] for i in range(N)], dtype=torch.long, device=device
    )

    # --- edge list → directed adjacency (undirected → both directions) ---
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    with open(edgelist_csv) as f:
        for row in csv.DictReader(f):
            u, v = int(row["source"]), int(row["target"])
            mask[v, u] = True   # W[post, pre]: pre=u → post=v
            mask[u, v] = True   # and reverse

    # --- initialize weights ---
    if weight_init == "uniform":
        W = torch.zeros(N, N, device=device)
        W[mask] = torch.empty(mask.sum().item(), device=device).uniform_(init_lo, init_hi)
    else:
        W = mask.float() * 0.5

    if normalize_by_degree:
        # Per-neuron in-degree normalization: keeps total synaptic drive ≈ constant
        # regardless of how many incoming connections a neuron has.
        in_deg = mask.float().sum(dim=1).clamp(min=1.0)   # (N,)
        W = W / in_deg.unsqueeze(1)

    return W, mask, community


# ---------------------------------------------------------------------------
# Community cross-mask
# ---------------------------------------------------------------------------

def make_cross_community_mask(community: torch.Tensor) -> torch.Tensor:
    """
    Returns (N,N) bool tensor: True where i and j belong to DIFFERENT communities.
    G_sim uses this as I[j ∉ m(i)].
    """
    return community.unsqueeze(1) != community.unsqueeze(0)   # (N,N)


# ---------------------------------------------------------------------------
# E-step: online stationary-distribution estimate
# ---------------------------------------------------------------------------

def update_p_tilde(
    p_tilde: torch.Tensor,  # (N,) current estimate
    W: torch.Tensor,        # (N,N) weight matrix
    z: torch.Tensor,        # (N,) spike vector at current timestep
    beta: float = 0.01,
) -> torch.Tensor:
    """
    Online power-iteration step (Eq. 2 from Map-STDP paper):
      p̃_i  ←  (1-β)·p̃_i  +  β · (T̃ p̃)_i
    evaluated stochastically via current spikes z(t).

    The stochastic approximation replaces p̃_j with z_j(t) in the inner product.
    """
    # W̃_ij = W_ij · p̃_j  (weight × destination stationary prob)
    W_tilde = W * p_tilde.unsqueeze(0)           # (N,N)
    s_tilde = W_tilde.sum(dim=1).clamp(min=1e-10) # (N,)

    # T̃_ij = W̃_ij / s̃_i
    T_tilde = W_tilde / s_tilde.unsqueeze(1)      # (N,N)

    # Stochastic update: use z(t) as noisy sample of p̃
    p_new = (1.0 - beta) * p_tilde + beta * (T_tilde @ z)
    # Renormalize to prevent drift
    p_new = p_new.clamp(min=0.0)
    total = p_new.sum()
    if total > 1e-10:
        p_new = p_new / total
    return p_new


# ---------------------------------------------------------------------------
# M-step quantities shared by both variants
# ---------------------------------------------------------------------------

def compute_W_tilde_and_s(
    W: torch.Tensor,
    p_tilde: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """W̃_ij = W_ij·p̃_j  and  s̃_i = Σ_k W_ik·p̃_k."""
    W_tilde = W * p_tilde.unsqueeze(0)             # (N,N)
    s_tilde = W_tilde.sum(dim=1).clamp(min=1e-10)  # (N,)
    return W_tilde, s_tilde


# ---------------------------------------------------------------------------
# M-step: classic Map-STDP gating (Eq. 3 simplified)
# ---------------------------------------------------------------------------

def compute_G_sim(
    W_tilde: torch.Tensor,      # (N,N)
    s_tilde: torch.Tensor,      # (N,)
    cross_mask: torch.Tensor,   # (N,N) bool: True where j ∉ m(i)
) -> torch.Tensor:
    """
    G_sim(i,j) = I[j ∉ m(i)] - ē_i
    where ē_i = e_i / s̃_i, e_i = Σ_{j'∉m(i)} W_ij' · p̃_j'
    """
    e_i = (W_tilde * cross_mask.float()).sum(dim=1)  # (N,)
    e_bar_i = e_i / s_tilde                          # (N,)
    G = cross_mask.float() - e_bar_i.unsqueeze(1)    # (N,N)
    return G


# ---------------------------------------------------------------------------
# M-step: lateral-inhibition variant (Eq. 6)
# ---------------------------------------------------------------------------

def compute_G_LI(
    W_tilde: torch.Tensor,   # (N,N) W̃_ij = W_ij·p̃_j
    s_tilde: torch.Tensor,   # (N,) s̃_i
    I_inh: torch.Tensor,     # (N,N) inhibitory weight matrix
    p_tilde: torch.Tensor,   # (N,) stationary distribution
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    φ_ij     = I_ij · p̃_j / (Σ_k I_ik · p̃_k)
    e_i^LI   = Σ_j φ_ij · W_ij · p̃_j
    ē_i^LI   = e_i^LI / s̃_i
    G_LI(i,j) = φ_ij - ē_i^LI

    Returns G_LI and phi (needed for inhibitory weight update).
    """
    I_weighted = I_inh * p_tilde.unsqueeze(0)              # (N,N): I_ij·p̃_j
    I_sum = I_weighted.sum(dim=1).clamp(min=1e-10)         # (N,): Σ_k I_ik·p̃_k
    phi = I_weighted / I_sum.unsqueeze(1)                   # (N,N): φ_ij

    e_i_LI = (phi * W_tilde).sum(dim=1)                    # (N,)
    e_bar_i_LI = e_i_LI / s_tilde                          # (N,)
    G_LI = phi - e_bar_i_LI.unsqueeze(1)                   # (N,N)
    return G_LI, phi


# ---------------------------------------------------------------------------
# Input weight helpers
# ---------------------------------------------------------------------------

def make_input_weights(
    n_input: int,
    n_workspace: int,
    device: torch.device = None,
    init_lo: float = 0.1,
    init_hi: float = 0.5,
) -> torch.Tensor:
    """
    Initialise W_in (n_workspace, n_input): random weights from input channels
    to the central workspace community.
    """
    device = device or torch.device("cpu")
    W_in = torch.empty(n_workspace, n_input, device=device).uniform_(init_lo, init_hi)
    return W_in


# ---------------------------------------------------------------------------
# Output decoding
# ---------------------------------------------------------------------------

def decode_community_rates(
    spike_counts: torch.Tensor,   # (N,) total spikes over presentation
    community: torch.Tensor,      # (N,) integer community labels
    n_classes: int = 10,
    workspace_community: int = 0,
) -> torch.Tensor:
    """
    For each classification community c in [1, n_classes], compute mean spike
    count across the 100 neurons in that community.  Return a (n_classes,)
    logit vector; argmax gives the predicted digit.
    """
    rates = torch.zeros(n_classes, device=spike_counts.device)
    for c in range(1, n_classes + 1):
        idx = (community == c).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            rates[c - 1] = spike_counts[idx].float().mean()
    return rates

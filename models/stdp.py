"""
Simplified suppression-based STDP (Gautam & Kohno 2023, Eq. 7).

Two separate STDP paths — chosen to minimise memory traffic:

Recurrent edges (≤ ~700 with 10 nodes/community)
    Stored as flat arrays (rec_post, rec_pre).  Per step: index into t_last and z
    with ~700 elements — very fast.

Input edges  (n_ws × N_in = 10 × 2312 = 23 120 with 10-node communities)
    NOT stored as a flat edge list; instead we keep separate t_last_ws (n_ws,) and
    t_last_in (N_in,) vectors and compute (n_ws × N_in) outer products ONLY when
    at least one workspace neuron or input channel is active.  No fancy indexing.

LTP/LTD rule (4-bit, simplified Eq. 7)
    LTP: post fires, Δt_since_post > St_post, 0 < Δt_since_pre < t_pre
    LTD: pre  fires, Δt_since_pre  > St_pre,  0 < Δt_since_post < t_post
    step = ±1/15
"""
from __future__ import annotations

import torch


class SuppressionSTDP:
    T_PRE_MS   =  5.7
    T_POST_MS  = 28.5
    ST_PRE_MS  = 17.1
    ST_POST_MS = 20.1
    WEIGHT_STEP = 1.0 / 15.0

    def __init__(
        self,
        n_rec: int,                    # number of recurrent neurons
        n_in: int,                     # number of input channels
        rec_post: torch.Tensor,        # (E_rec,) post in [0, n_rec)
        rec_pre: torch.Tensor,         # (E_rec,) pre  in [0, n_rec)
        ws_local_idx: torch.Tensor,    # (n_ws,)  local indices of workspace neurons in rec
        dt_ms: float = 1.0,
        device: torch.device | None = None,
    ):
        self.device  = device or torch.device("cpu")
        self.N       = n_rec
        self.N_in    = n_in
        self.n_ws    = ws_local_idx.numel()
        self.dt      = dt_ms

        self.t_pre   = int(self.T_PRE_MS   / dt_ms)
        self.t_post  = int(self.T_POST_MS  / dt_ms)
        self.st_pre  = int(self.ST_PRE_MS  / dt_ms)
        self.st_post = int(self.ST_POST_MS / dt_ms)

        # Recurrent edges
        self.rec_post = rec_post.to(self.device)
        self.rec_pre  = rec_pre.to(self.device)
        self.E_rec    = rec_post.numel()

        # Workspace neuron indices (local to recurrent array)
        self.ws_idx = ws_local_idx.to(self.device)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        NEG_INF = -1_000_000.0
        # Timing for recurrent neurons
        self.t_last_rec = torch.full((self.N,),    NEG_INF, dtype=torch.float32, device=self.device)
        # Timing for input channels
        self.t_last_in  = torch.full((self.N_in,), NEG_INF, dtype=torch.float32, device=self.device)
        # Accumulators
        self.dW_rec = torch.zeros(self.E_rec,             dtype=torch.float32, device=self.device)
        self.dW_in  = torch.zeros(self.n_ws, self.N_in,  dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def step(self, z_rec: torch.Tensor, z_in: torch.Tensor, t: int):
        """
        z_rec : (N_rec,) float spike vector
        z_in  : (N_in,)  float spike vector
        t     : integer timestep (ms)
        """
        step  = self.WEIGHT_STEP
        t_f   = float(t)

        # ---- Recurrent STDP (flat edge approach) -----------------------
        ts_rec = t_f - self.t_last_rec            # (N_rec,) time-since-last-spike

        ts_post = ts_rec[self.rec_post]            # (E_rec,)
        ts_pre  = ts_rec[self.rec_pre]             # (E_rec,)

        ltp_r = (
            (z_rec[self.rec_post] > 0)
            & (ts_post > self.st_post)
            & (ts_pre  > 0) & (ts_pre < self.t_pre)
        )
        self.dW_rec[ltp_r] += step

        ltd_r = (
            (z_rec[self.rec_pre] > 0)
            & (ts_pre  > self.st_pre)
            & (ts_post > 0) & (ts_post < self.t_post)
        )
        self.dW_rec[ltd_r] -= step

        # ---- Input STDP (2D outer-product approach) ---------------------
        # We work with ts_ws (n_ws,) and ts_in (N_in,) — no fancy indexing.
        ts_ws = ts_rec[self.ws_idx]                # (n_ws,)  t_since for workspace neurons
        ts_in = t_f - self.t_last_in               # (N_in,)  t_since for input channels

        z_ws  = z_rec[self.ws_idx]                 # (n_ws,)  current ws spike

        # LTP: workspace (post) fires, suppression ok, input (pre) in window
        ws_ltp_ok = z_ws * (ts_ws > self.st_post).float()   # (n_ws,)
        in_ltp_ok = ((ts_in > 0) & (ts_in < self.t_pre)).float()  # (N_in,)
        if ws_ltp_ok.any() and in_ltp_ok.any():
            self.dW_in += step * torch.outer(ws_ltp_ok, in_ltp_ok)

        # LTD: input (pre) fires, suppression ok, workspace (post) in window
        ws_ltd_ok = ((ts_ws > 0) & (ts_ws < self.t_post)).float()  # (n_ws,)
        in_ltd_ok = z_in * (ts_in > self.st_pre).float()            # (N_in,)
        if ws_ltd_ok.any() and in_ltd_ok.any():
            self.dW_in -= step * torch.outer(ws_ltd_ok, in_ltd_ok)

        # ---- Update timing ------------------------------------------------
        self.t_last_rec[z_rec > 0] = t_f
        self.t_last_in [z_in  > 0] = t_f

    # ------------------------------------------------------------------
    def reset_dW(self):
        self.dW_rec.zero_()
        self.dW_in.zero_()

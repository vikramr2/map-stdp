"""
Simplified suppression-based STDP rule (Gautam & Kohno 2023).

Equation (7) from the paper:
  Δw = +1 bit  if Δt >= 0 AND Δt < t_pre  AND Δt_post > St_post   (LTP)
  Δw = -1 bit  if Δt < 0  AND |Δt| < t_post AND Δt_pre > St_pre   (LTD)

where:
  Δt      = t_post - t_pre   (signed spike interval for the current pair)
  Δt_post = time since the PREVIOUS postsynaptic spike (suppresses potentiation)
  Δt_pre  = time since the PREVIOUS presynaptic spike  (suppresses depression)

Parameters used for the 4-bit variant (Table I / Fig 3(b)):
  t_pre   =  5.7 ms  → window for causal LTP
  t_post  = 28.5 ms  → window for anti-causal LTD
  St_pre  = 17.1 ms  → pre-suppression threshold
  St_post = 20.1 ms  → post-suppression threshold
  weight resolution: 4-bit, step = 1/15

All times are in simulation steps (1 step = 1 ms).
"""

import torch


class SuppressionSTDP:
    """
    Tracks spike timing for all neurons and accumulates ΔW updates.

    Convention: W[i, j]  =  synaptic weight from pre-neuron j to post-neuron i.

    LTP fires when post=i fires and pre=j fired within [0, t_pre) steps ago,
         provided no recent previous post spike suppresses it.
    LTD fires when pre=j fires and post=i fired within (0, t_post) steps ago,
         provided no recent previous pre spike suppresses it.
    """

    # 4-bit fixed-point parameters from the paper (all in ms / timesteps at 1 ms resolution)
    T_PRE_MS   =  5.7
    T_POST_MS  = 28.5
    ST_PRE_MS  = 17.1
    ST_POST_MS = 20.1
    WEIGHT_STEP = 1.0 / 15.0   # 4-bit resolution

    def __init__(
        self,
        n_neurons: int,
        mask: torch.Tensor,          # (N, N) bool/float connectivity mask
        dt_ms: float = 1.0,
        device: torch.device = None,
        input_n: int = 0,            # extra input neurons prepended (virtual)
    ):
        self.n     = n_neurons
        self.n_in  = input_n
        self.n_tot = n_neurons + input_n
        self.dt    = dt_ms
        self.device = device or torch.device("cpu")

        # Convert ms thresholds to integer timesteps
        self.t_pre   = int(self.T_PRE_MS   / dt_ms)
        self.t_post  = int(self.T_POST_MS  / dt_ms)
        self.st_pre  = int(self.ST_PRE_MS  / dt_ms)
        self.st_post = int(self.ST_POST_MS / dt_ms)

        self.mask = mask.to(self.device)  # (N_tot, N_tot) or relevant sub-block

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Call between samples to clear timing state."""
        NEG_INF = -100_000
        # t_last[k] = timestep of the most-recent spike of neuron k (before current t)
        self.t_last = torch.full((self.n_tot,), NEG_INF,
                                 dtype=torch.float32, device=self.device)
        # Accumulated weight delta; reset each sample, applied in bulk at end
        self.dW = torch.zeros(self.n_tot, self.n_tot,
                              dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def step(self, z: torch.Tensor, t: int):
        """
        Update STDP accumulators for one timestep.

        Args:
            z : spike vector (n_tot,) in {0, 1}, float
            t : current integer timestep
        """
        # Time since each neuron's last spike — using OLD t_last (before update)
        t_since = t - self.t_last          # (N_tot,)

        # ---- LTP -------------------------------------------------------
        # Row = post (currently firing), Col = pre (fired recently)
        # supp_ok_post[i]: time since i's previous spike > St_post
        supp_ok_post = (t_since > self.st_post).float()
        # ltp_elig[j]: j fired within [1, t_pre) steps ago
        #   (use >0 to exclude t_since=0, i.e., j has never spiked or fired simultaneously)
        ltp_elig = ((t_since > 0) & (t_since < self.t_pre)).float()

        # outer(post_fires * supp, pre_elig) → shape (N_tot, N_tot)
        dW_ltp = torch.outer(z * supp_ok_post, ltp_elig)

        # ---- LTD -------------------------------------------------------
        # Row = post (fired recently), Col = pre (currently firing)
        supp_ok_pre = (t_since > self.st_pre).float()
        ltd_elig = ((t_since > 0) & (t_since < self.t_post)).float()

        dW_ltd = torch.outer(ltd_elig, z * supp_ok_pre)

        # Apply connectivity mask and accumulate
        self.dW += (dW_ltp - dW_ltd) * self.mask * self.WEIGHT_STEP

        # ---- Update t_last ---------------------------------------------
        # Only update for neurons that actually fired
        fired = z > 0
        self.t_last[fired] = float(t)

    # ------------------------------------------------------------------
    def get_and_reset_dW(self) -> torch.Tensor:
        """Return accumulated ΔW and zero the accumulator."""
        dW = self.dW.clone()
        self.dW.zero_()
        return dW

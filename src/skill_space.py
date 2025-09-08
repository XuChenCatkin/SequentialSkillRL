from __future__ import annotations
from dataclasses import dataclass
from math import fabs
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# ---------- helpers ----------------------------------------------------------

def stick_breaking(beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Truncated stick-breaking (finite K).
    beta: [K] in (0,1)
    returns:
      pi:   [K] with pi_k = beta_k * prod_{i<k} (1 - beta_i)
      rest: scalar remaining mass (prod_{i=1..K} (1 - beta_i))
    """
    K = beta.numel()
    one = torch.ones(1, device=beta.device, dtype=beta.dtype)
    remain_prefix = torch.cumprod(torch.cat([one, 1.0 - beta[:-1]], dim=0), dim=0)  # [K]
    pi = beta * remain_prefix
    rest = torch.prod(1.0 - beta)
    return pi, rest

def chol_inv_logdet(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    S: [..., D, D] SPD
    returns (S^{-1}, log|S|)
    """
    L = torch.linalg.cholesky(S)
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
    inv = torch.cholesky_inverse(L)
    return inv, logdet

# ---------- priors/posteriors containers ------------------------------------

@dataclass
class NIWPrior:
    mu0: torch.Tensor        # [D]
    kappa0: float
    Psi0: torch.Tensor       # [D,D] (scale matrix of IW)
    nu0: float               # > D-1

@dataclass
class NIWPosterior:
    mu: torch.Tensor         # [Kp1,D]
    kappa: torch.Tensor      # [Kp1]
    Psi: torch.Tensor        # [Kp1,D,D]
    nu: torch.Tensor         # [Kp1]

@dataclass
class DirPosterior:
    # Row-wise parameters for q(Φ_k) = Dir(φ_k), φ_k in R^{Kp1}
    phi: torch.Tensor        # [Kp1,Kp1]

@dataclass
class StickyHDPHMMParams:
    alpha: float = 4.0
    kappa: float = 4.0
    gamma: float = 1.0           # Beta(1, gamma) on stick v_k
    K: int = 16                  # number of explicit states (excl. remainder)
    D: int = 64
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

# ---------- main class -------------------------------------------------------

class StickyHDPHMMVI(nn.Module):
    """
    Sticky HDP-HMM with Gaussian emissions (NIW prior), truncated to (K+1) states:
      - first K are explicit states,
      - (K+1)-th is the 'remainder' absorbing the tail mass of the GEM.

    Encoder covariance Σ_t = diag(σ_t^2) + F_t F_t^T (optional).

    Unified update:
      - Call update(..., rho=1.0) for non-streaming (full batch override)
      - Call update(..., 0<rho<1) for streaming EMA
    """

    # ---- init ----------------------------------------------------------------
    def __init__(
        self,
        p: StickyHDPHMMParams,
        niw_prior: NIWPrior,
        rho_emission: float = 0.05,
        rho_transition: Optional[float] = None
    ):
        super().__init__()
        self.p = p
        D, K = p.D, p.K
        Kp1 = K + 1
        dev, dt = p.device, p.dtype

        # Emission NIW posteriors (now K+1 states)
        self.niw = NIWPosterior(
            mu=torch.stack([niw_prior.mu0.clone().to(device=dev, dtype=dt) for _ in range(Kp1)], dim=0),
            kappa=torch.full((Kp1,), niw_prior.kappa0, device=dev, dtype=dt),
            Psi=torch.stack([niw_prior.Psi0.clone().to(device=dev, dtype=dt) for _ in range(Kp1)], dim=0),
            nu=torch.full((Kp1,), niw_prior.nu0, device=dev, dtype=dt),
        )

        # Global sticks β for the first K components; π_{K+1} is the remainder mass
        beta = torch.tensor([1.0 / (K + 2 - k) for k in range(1, K+1)], device=dev, dtype=dt)
        self.register_parameter("u_beta", nn.Parameter(torch.log(beta) - torch.log1p(-beta)))
        
        # Transition posteriors q(Φ_k)=Dir(φ_k), rows and cols = K+1
        pi_full = self._Epi() * self.p.alpha + self.p.kappa * torch.eye(Kp1, device=dev, dtype=dt)
        self.dir = DirPosterior(
            phi=pi_full
        )

        # Cache for E[Λ], E[log|Λ|]
        self._cache_fresh = False
        self._E_Lambda = None          # [Kp1,D,D]
        self._E_logdet_Lambda = None   # [Kp1]

        # Save NIW prior
        self.register_buffer("mu0", niw_prior.mu0.to(device=dev, dtype=dt))
        self.kappa0 = float(niw_prior.kappa0)
        self.register_buffer("Psi0", niw_prior.Psi0.to(device=dev, dtype=dt))
        self.nu0 = float(niw_prior.nu0)

        # Streaming defaults + lazy buffers
        self.stream_rho = float(rho_emission)
        self.stream_rho_trans = float(rho_transition) if rho_transition is not None else None
        self._stream_allocated = False
        self.streaming_reset()
        
    def reset(self):
        """Reset model parameters to prior values."""
        Kp1, D = self.niw.mu.shape[0], self.p.D
        self.niw = NIWPosterior(
            mu=torch.stack([self.mu0.clone().to(device=self.p.device, dtype=self.p.dtype) for _ in range(Kp1)], dim=0),
            kappa=torch.full((Kp1,), self.kappa0, device=self.p.device, dtype=self.p.dtype),
            Psi=torch.stack([self.Psi0.clone().to(device=self.p.device, dtype=self.p.dtype) for _ in range(Kp1)], dim=0),
            nu=torch.full((Kp1,), self.nu0, device=self.p.device, dtype=self.p.dtype),
        )
        beta = torch.tensor([1.0 / (self.p.K + 2 - k) for k in range(1, Kp1)], device=self.p.device, dtype=self.p.dtype)
        self.u_beta.data.copy_(torch.log(beta) - torch.log1p(-beta))
        pi_full = self._Epi() * self.p.alpha + self.p.kappa * torch.eye(Kp1, device=self.p.device, dtype=self.p.dtype)
        self.dir.phi.data.copy_(pi_full)
        self._cache_fresh = False
        self.streaming_reset()


    # --- ELBO terms ------------------------------------------

    @staticmethod
    def _dirichlet_logC(row_params: torch.Tensor) -> torch.Tensor:
        """
        Sum over rows of Dirichlet log-normalizer:
        log C(α) = log Γ(∑ α_j) - ∑ log Γ(α_j)
        row_params: [Kp1,Kp1]
        """
        row = torch.clamp(row_params, min=1e-8)
        return torch.lgamma(row.sum(dim=1)).sum() - torch.lgamma(row).sum()

    @staticmethod
    def _logZ_invwishart(Psi: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """
        log normalizer of IW(Ψ, ν), summed over rows if batched:
        log Z_IW = (ν D /2) log 2 + (D(D-1)/4) log π + (ν/2) log|Ψ| + ∑_i log Γ((ν+1-i)/2)
        Psi: [Kp1,D,D] or [D,D]
        nu:  [Kp1]      or scalar
        returns scalar (sum across rows if batched)
        """
        if Psi.dim() == 2:
            Psi = Psi.unsqueeze(0)
            nu = nu.view(1)
        Kp1, D, _ = Psi.shape
        # log|Ψ|
        _, logdet_Psi = chol_inv_logdet(Psi)
        # multivariate gamma term
        i = torch.arange(1, D + 1, device=Psi.device, dtype=Psi.dtype).view(1, D)
        lgamma_sum = torch.sum(torch.lgamma((nu.view(Kp1, 1) + 1.0 - i) / 2.0), dim=1)  # [Kp1]
        term = (nu * D / 2.0) * torch.log(torch.tensor(2.0, device=Psi.device, dtype=Psi.dtype)) \
            + (D * (D - 1) / 4.0) * torch.log(torch.tensor(torch.pi, device=Psi.device, dtype=Psi.dtype)) \
            - 0.5 * nu * logdet_Psi + lgamma_sum
        return term.sum()

    def _niw_elbo_term(self, mu_hat, k_hat, Psi_hat, nu_hat) -> torch.Tensor:
        """
        Sum_k ( E_q[log p(μ_k,Σ_k)] - E_q[log q(μ_k,Σ_k)] ) under NIW prior/posterior.
        Uses the same parameterization as this module (Σ ~ IW(Ψ, ν), μ|Σ ~ N(μ0, Σ/κ0)).
        Returns scalar tensor.
        """
        mu0, k0, Psi0, nu0 = self.mu0, self.kappa0, self.Psi0, self.nu0
        Kp1, D = mu_hat.shape[0], mu_hat.shape[1]

        # Expectations under q
        E_Lambda, E_logdet_Lambda = StickyHDPHMMVI._calc_Lambda_expectations(Psi_hat, nu_hat)  # [Kp1,D,D]

        # --- IW part: -KL(q(Σ)||p(Σ)) ---
        # log Z_IW(q) - log Z_IW(p)
        logZ_p = StickyHDPHMMVI._logZ_invwishart(Psi0, torch.tensor(nu0, device=Psi_hat.device, dtype=Psi_hat.dtype))
        logZ_q = StickyHDPHMMVI._logZ_invwishart(Psi_hat, nu_hat)
        logZ_term = logZ_q - logZ_p * Kp1

        # -0.5 * (ν_hat - ν₀) * E_q[log|Λ|]
        # Note: E_q[log|Σ|] = -E_q[log|Λ|]
        logdet_term = -0.5 * torch.sum((nu_hat - nu0) * E_logdet_Lambda)

        # +0.5 * Tr((Ψ_hat - Ψ₀) * E_q[Λ])
        tr_term = 0.5 * torch.einsum('kij,kji->', (Psi_hat - Psi0.unsqueeze(0)), E_Lambda)
        
        iw_term = logZ_term + logdet_term + tr_term

        # Normal part: E_q[log N(μ | μ0, Σ/κ0)] - E_q[log N(μ | μ_hat, Σ/κ_hat)]
        # = 0.5 * ∑_k [ D*(log(κ0/κ_hat) + 1 - κ0/κ_hat) - κ0 (μ_hat_k - μ0)^T E[Λ]_k (μ_hat_k - μ0) ]
        diff = (mu_hat - mu0.view(1, D)).unsqueeze(-1)           # [Kp1,D,1]
        quad = torch.einsum('kde,kef,kdf->k', E_Lambda, diff, diff)  # [Kp1]
        log_k_ratio = torch.log(torch.clamp(k0 / k_hat, min=1e-9))
        normal_term = 0.5 * (D * (log_k_ratio + 1.0 - (k0 / k_hat)) - k0 * quad).sum()

        return iw_term + normal_term

    # ---- cache E[Λ], E[log|Λ|] -----------------------------------------------
    def _refresh_emission_cache(self):
        Kp1, D = self.niw.mu.shape[0], self.p.D
        Psi = self.niw.Psi
        nu = self.niw.nu.view(Kp1, 1, 1)
        Psi_inv, logdet_Psi = chol_inv_logdet(Psi)
        E_Lambda = nu * Psi_inv  # [Kp1,D,D]

        i = torch.arange(1, D + 1, device=Psi.device, dtype=Psi.dtype).view(1, D)
        E_logdet = torch.sum(torch.special.digamma((self.niw.nu.view(Kp1, 1) + 1 - i) / 2.0), dim=1)
        E_logdet = E_logdet + D * torch.log(torch.tensor(2.0, device=Psi.device, dtype=Psi.dtype)) - logdet_Psi

        self._E_Lambda = E_Lambda
        self._E_logdet_Lambda = E_logdet
        self._cache_fresh = True

    def _get_E_Lambda(self):
        if not self._cache_fresh:
            self._refresh_emission_cache()
        return self._E_Lambda

    def _get_E_logdet_Lambda(self):
        if not self._cache_fresh:
            self._refresh_emission_cache()
        return self._E_logdet_Lambda
    
    @staticmethod
    def _calc_Lambda_expectations(Psi: torch.Tensor, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given IW parameters, compute E[Λ] and E[log|Λ|].
        Psi: [Kp1,D,D]
        nu:  [Kp1]
        returns (E_Lambda [Kp1,D,D], E_logdet [Kp1])
        """
        Kp1, D = Psi.shape[0], Psi.shape[1]
        Psi_inv, logdet_Psi = chol_inv_logdet(Psi)
        E_Lambda = nu.view(Kp1, 1, 1) * Psi_inv  # [Kp1,D,D]
        i = torch.arange(1, D + 1, device=Psi.device, dtype=Psi.dtype).view(1, D)
        E_logdet = torch.sum(torch.special.digamma((nu.view(Kp1, 1) + 1 - i) / 2.0), dim=1)
        E_logdet = E_logdet + D * torch.log(torch.tensor(2.0, device=Psi.device, dtype=Psi.dtype)) - logdet_Psi

        return E_Lambda, E_logdet

    # ---- expected emission log-likelihood logB_tk ----------------------------
    @staticmethod
    @torch.no_grad()
    def expected_emission_loglik(
        mu_hat: torch.Tensor,                          # [Kp1,D]
        k_hat: torch.Tensor,                           # [Kp1]
        Psi_hat: torch.Tensor,                         # [Kp1,D,D]
        nu_hat: torch.Tensor,                          # [Kp1]
        mu_t: torch.Tensor,                            # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,     # [T,D] or [B,T,D]
        F_t: Optional[torch.Tensor] = None,            # [T,D,R] or [B,T,D,R]
        mask: Optional[torch.Tensor] = None            # [T] or [B,T] (1=valid, 0=ignore)
    ) -> torch.Tensor:
        """
        Returns logB: [T,Kp1] (or [B,T,Kp1]) with E_q[log p(z_t | h_t=k)].
        E_q[...] = 0.5*(Elog|Λ_k| - D log(2π))
                   - 0.5*( Tr(EΛ Σ_t)
                          + (m_t-μ_k)^T EΛ (m_t-μ_k)
                          + D / κ_k )
        Mask: invalid frames act as 'no observation' => logB[t,:] = 0 at those steps.
        """
        Kp1, D = mu_hat.shape
        E_Lam, E_logdet = StickyHDPHMMVI._calc_Lambda_expectations(Psi_hat, nu_hat)  # [Kp1,D,D], [Kp1]

        # shape to [B,T,...]
        input_was_2d = mu_t.dim() == 2
        if mu_t.dim() == 2:
            mu_t = mu_t.unsqueeze(0)
            if diag_var_t is not None: diag_var_t = diag_var_t.unsqueeze(0)
            if F_t is not None: F_t = F_t.unsqueeze(0)
            if mask is not None: mask = mask.unsqueeze(0)

        B, T, D_ = mu_t.shape
        assert D_ == D

        if diag_var_t is None:
            diag_var_t = torch.zeros(B, T, D, device=mu_t.device, dtype=mu_t.dtype)

        # constant term
        const = 0.5 * (E_logdet - D * torch.log(torch.tensor(2.0 * torch.pi, device=mu_t.device, dtype=mu_t.dtype)))  # [Kp1]

        # (m_t - μ_k)
        diff = mu_t.unsqueeze(2) - mu_hat.view(1, 1, Kp1, D)    # [B,T,Kp1,D]

        # quadratic term: (m-μ)^T EΛ (m-μ)
        quad_mean = torch.einsum('btkd,kde,btke->btk', diff, E_Lam, diff)  # [B,T,Kp1]

        # trace term: Tr(EΛ Σ_t) = sum_d EΛ[k,d,d]*σ_t[d] + sum_r f_{t,r}^T EΛ f_{t,r}
        E_Lam_diag = torch.diagonal(E_Lam, dim1=1, dim2=2)  # [Kp1,D]
        tr_diag = torch.einsum('kd,btd->btk', E_Lam_diag, diag_var_t)      # [B,T,Kp1]

        if F_t is not None:
            # sum_r f^T EΛ f
            tr_lr = torch.einsum('btdr,kde,bter->btrk', F_t, E_Lam, F_t).sum(dim=2)  # [B,T,Kp1]
        else:
            tr_lr = 0.0

        # NIW mean-uncertainty term: D / kappa_k
        D_over_kappa = (D / k_hat).view(1, 1, Kp1)  # [1,1,Kp1]

        logB = const.view(1, 1, Kp1) - 0.5 * (quad_mean + tr_diag + tr_lr + D_over_kappa)  # [B,T,Kp1]

        if mask is not None:
            # no-observation: add 0 contribution (i.e., keep alpha/beta recursion well-defined)
            m = mask.view(B, T, 1).bool()
            logB = torch.where(m, logB, torch.zeros_like(logB))

        return logB if not input_was_2d else logB.squeeze(0)

    # ---- forward-backward in log-space --------------------------------------
    @staticmethod
    @torch.no_grad()
    def forward_backward(
        log_pi: torch.Tensor,   # [Kp1]
        ElogA: torch.Tensor,    # [Kp1,Kp1]
        logB: torch.Tensor      # [T,Kp1]
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Standard HMM FB in log-domain with 'no observation' allowed (logB[t,:]==0).
        Returns rhat [T,Kp1], xihat [T-1,Kp1,Kp1], loglik (scalar).
        """
        T, Kp1 = logB.shape
        dev = logB.device
        orig_dtype = logB.dtype

        # Work in float64 for stability
        log_pi = log_pi.to(torch.float64)
        ElogA  = ElogA.to(torch.float64)
        logB   = logB.to(torch.float64)
        # α
        log_alpha = torch.empty((T, Kp1), device=dev, dtype=torch.float64)
        c = torch.empty(T, device=dev, dtype=torch.float64)

        log_alpha[0] = log_pi + logB[0]
        c[0] = torch.logsumexp(log_alpha[0], dim=-1)
        log_alpha[0] -= c[0]

        for t in range(1, T):
            prev = log_alpha[t-1].unsqueeze(1) + ElogA          # [Kp1,Kp1]
            log_alpha[t] = logB[t] + torch.logsumexp(prev, 0)   # [Kp1]
            c[t] = torch.logsumexp(log_alpha[t], dim=-1)
            log_alpha[t] -= c[t]

        ll = float(c.sum().item())

        # β
        log_beta = torch.zeros((T, Kp1), device=dev, dtype=torch.float64)
        for t in range(T-2, -1, -1):
            tmp = ElogA + (logB[t+1] + log_beta[t+1]).unsqueeze(0)  # [Kp1,Kp1]
            log_beta[t] = torch.logsumexp(tmp, dim=1) - c[t+1]      # [Kp1]

        # posteriors
        log_gamma = log_alpha + log_beta                            # [T,Kp1]
        den_g = torch.logsumexp(log_gamma, dim=1, keepdim=True)     # [T,1]
        rhat = torch.exp(log_gamma - den_g).to(logB.dtype)          # sums to 1 per row

        xihat = torch.empty((T-1, Kp1, Kp1), device=dev, dtype=logB.dtype)
        for t in range(T-1):
            tmp = (log_alpha[t].unsqueeze(1) + ElogA
                + logB[t+1].unsqueeze(0) + log_beta[t+1].unsqueeze(0))   # [Kp1,Kp1]
            den_xi = torch.logsumexp(tmp.view(-1), dim=0)
            xihat[t] = torch.exp(tmp - den_xi)

        return rhat.to(orig_dtype), xihat.to(orig_dtype), ll

    @staticmethod
    @torch.no_grad()
    def viterbi(
        log_pi: torch.Tensor,   # [Kp1]
        ElogA: torch.Tensor,    # [Kp1,Kp1]
        logB: torch.Tensor      # [T,Kp1]
    ) -> Tuple[torch.Tensor, float]:
        """
        Viterbi algorithm to find the most likely state sequence.
        
        Args:
            log_pi: Log initial state probabilities [Kp1]
            ElogA: Log transition matrix [Kp1,Kp1] 
            logB: Log emission probabilities [T,Kp1]
            
        Returns:
            path: Most likely state sequence [T] (int64)
            log_prob: Log probability of the most likely path (scalar)
        """
        T, Kp1 = logB.shape
        
        # Viterbi forward pass: compute max probabilities and backtrack pointers
        log_delta = torch.full((T, Kp1), -float('inf'), device=logB.device, dtype=logB.dtype)
        psi = torch.zeros((T, Kp1), dtype=torch.long, device=logB.device)
        
        # Initialize
        log_delta[0] = log_pi + logB[0]
        
        # Forward pass
        for t in range(1, T):
            # For each state k at time t, find the best previous state
            prev_scores = log_delta[t-1].unsqueeze(1) + ElogA  # [Kp1,Kp1]
            log_delta[t], psi[t] = torch.max(prev_scores, dim=0)  # [Kp1], [Kp1]
            log_delta[t] = log_delta[t] + logB[t]
        
        # Find the best final state
        log_prob, best_last_state = torch.max(log_delta[-1], dim=0)
        
        # Backward pass: reconstruct the path
        path = torch.zeros(T, dtype=torch.long, device=logB.device)
        path[-1] = best_last_state
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
            
        return path, float(log_prob.item())
    
    @torch.no_grad()
    def viterbi_paths(
        self,
        mu_t, diag_var_t=None, F_t=None, mask=None
    ):
        """Decode each sequence; return list of 1D long tensors (one per sequence)."""
        logB = StickyHDPHMMVI.expected_emission_loglik(
            self.niw.mu, self.niw.kappa, self.niw.Psi, self.niw.nu,
            mu_t, diag_var_t, F_t, mask
        )
        if logB.dim() == 2: logB = logB.unsqueeze(0)
        B, T, K = logB.shape
        log_pi = torch.log(torch.clamp(self._Epi(), min=1e-30))
        logA = self._ElogA()
        paths = []
        for b in range(B):
            lb = logB[b]
            if mask is not None:
                m = mask[b].bool()
                z_full = torch.full((T,), K-1, dtype=torch.long, device=lb.device)  # any default
                z_obs, _ = StickyHDPHMMVI.viterbi(log_pi, logA, lb[m])
                z_full[m] = z_obs
                z = z_full[m]    # only valid frames
            else:
                z, _ = StickyHDPHMMVI.viterbi(log_pi, logA, lb)
            paths.append(z)
        return paths

    @staticmethod
    @torch.no_grad()
    def dwell_stats_from_path(z: torch.Tensor, K: int):
        """Return per-state list of run-lengths and their means from a 1D path."""
        lens = [[] for _ in range(K)]
        if z.numel() == 0: return lens, [0. for _ in range(K)]
        prev, r = int(z[0]), 1
        for t in range(1, z.numel()):
            cur = int(z[t])
            if cur == prev:
                r += 1
            else:
                lens[prev].append(r)
                prev, r = cur, 1
        lens[prev].append(r)
        means = [float(torch.tensor(ls).float().mean().item()) if len(ls) else 0. for ls in lens]
        return lens, means

    # ---- M-step: sufficient stats from encoder -------------------------------
    @torch.no_grad()
    def _moments_from_encoder(
        self,
        mu_t: torch.Tensor, rhat: torch.Tensor,
        diag_var_t: Optional[torch.Tensor] = None,
        F_t: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (Nk [Kp1], M1 [Kp1,D], M2 [Kp1,D,D]) where
        M2 accumulates E[zz^T] = diag(var_t) + μ_t μ_t^T + Σ_r f_{t,r} f_{t,r}^T.
        If mask is provided, masked steps do not contribute.
        """
        if mu_t.dim() == 3:
            mu_t = mu_t.reshape(-1, mu_t.size(-1))
            if diag_var_t is not None:
                diag_var_t = diag_var_t.reshape(-1, diag_var_t.size(-1))
            if F_t is not None:
                F_t = F_t.reshape(-1, F_t.size(-2), F_t.size(-1))  # [BT,D,R]
            if mask is not None:
                mask = mask.reshape(-1)  # [BT]
        elif mu_t.dim() == 1:
            # Handle case where mu_t is somehow 1D - reshape to [1, D]
            mu_t = mu_t.unsqueeze(0)
            if diag_var_t is not None and diag_var_t.dim() == 1:
                diag_var_t = diag_var_t.unsqueeze(0)
            if F_t is not None and F_t.dim() == 2:
                F_t = F_t.unsqueeze(0)
            if mask is not None and mask.dim() == 0:
                mask = mask.unsqueeze(0)

        T, D = mu_t.shape
        Kp1 = self.niw.mu.shape[0]
        rhat = rhat.reshape(-1, Kp1)  # [T,Kp1]

        if mask is not None:
            m = mask.to(rhat.dtype).view(-1, 1)
            rhat = rhat * m  # zero out masked frames

        Nk = rhat.sum(dim=0)                            # [Kp1]
        M1 = torch.einsum('tk,td->kd', rhat, mu_t)      # [Kp1,D]

        if diag_var_t is None:
            diag_var_t = torch.zeros_like(mu_t)

        M2 = torch.zeros(Kp1, D, D, device=mu_t.device, dtype=mu_t.dtype)

        # diagonal part
        for k in range(Kp1):
            w_diag = (rhat[:, k].unsqueeze(1) * diag_var_t).sum(dim=0)  # [D]
            M2[k] += torch.diag(w_diag)

        # mean outer products
        M2 += torch.einsum('tk,td,te->kde', rhat, mu_t, mu_t)

        # low-rank
        if F_t is not None:
            R = F_t.size(-1)
            for r in range(R):
                f = F_t[..., r]  # [T,D]
                M2 += torch.einsum('tk,td,te->kde', rhat, f, f)

        return Nk, M1, M2
    
    @torch.no_grad()
    def _update_moments(
        self,
        Nk: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        xi_counts: torch.Tensor
    ) -> None:
        """
        Update sufficient statistics for the NIW prior.
        """
        self.S_Nk = Nk
        self.S_M1 = M1
        self.S_M2 = M2
        self.S_counts = xi_counts

    @torch.no_grad()
    def _calc_NIW_posterior(self, Nk, M1, M2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute NIW posterior parameters from soft moments."""
        Kp1, D = M1.shape[0], M1.shape[1]
        mu0, k0, Psi0, nu0 = self.mu0, self.kappa0, self.Psi0, self.nu0
        k_hat = k0 + Nk
        mu_hat = (k0 * mu0.unsqueeze(0) + M1) / k_hat.unsqueeze(1)
        nu_hat = nu0 + Nk

        Psi_hat = Psi0.unsqueeze(0).expand(Kp1, D, D).clone()
        Psi_hat = Psi_hat + M2 + k0 * torch.einsum('d,e->de', mu0, mu0).unsqueeze(0) \
                  - torch.einsum('k,kd,ke->kde', k_hat, mu_hat, mu_hat)

        # small jitter to keep SPD
        eps = 1e-6
        I = torch.eye(D, device=Psi_hat.device, dtype=Psi_hat.dtype)
        Psi_hat = Psi_hat + eps * I.unsqueeze(0)

        return mu_hat, k_hat, Psi_hat, nu_hat

    @torch.no_grad()
    def _update_NIW(self, mu_hat, k_hat, Psi_hat, nu_hat):
        """Update NIW posterior params in-place."""

        self.niw.mu = mu_hat
        self.niw.kappa = k_hat
        self.niw.Psi = Psi_hat
        self.niw.nu = nu_hat
        self._cache_fresh = False
        
    @torch.no_grad()
    def _update_u_beta(self, u_beta: torch.Tensor):
        self.u_beta.data.copy_(u_beta.detach())

    # ---- transitions & π -----------------------------------------------------
    @torch.no_grad()
    def _EA(self) -> torch.Tensor:
        """Row-wise mean of transitions under Dirichlet posterior."""
        phi = torch.clamp(self.dir.phi, min=1e-8)
        return phi / phi.sum(dim=1, keepdim=True)
    
    def _ElogA(self) -> torch.Tensor:
        phi = self.dir.phi  # [Kp1,Kp1]
        return torch.special.digamma(phi) - torch.special.digamma(phi.sum(dim=1, keepdim=True))

    @torch.no_grad()
    def _Epi(self) -> torch.Tensor:
        """Return π over K+1 states: concat(stick_breaking(β), remainder)."""
        beta = torch.sigmoid(self.u_beta)     # [K]
        piK, rest = stick_breaking(beta)      # [K], scalar
        pi_full = torch.cat([piK, rest.view(1)], dim=0)  # [Kp1]
        return pi_full
    
    @staticmethod
    def _calc_ElogA(phi: torch.Tensor) -> torch.Tensor:
        phi_stable = torch.clamp(phi, min=1e-8)
        return torch.special.digamma(phi_stable) - torch.special.digamma(phi_stable.sum(dim=1, keepdim=True))

    @staticmethod
    @torch.no_grad()
    def _calc_Epi(u_beta: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(u_beta)     # [K]
        piK, rest = stick_breaking(beta)      # [K], scalar
        pi_full = torch.cat([piK, rest.view(1)], dim=0)  # [Kp1]
        return pi_full

    @torch.no_grad()
    def _calc_dir_posterior(self, xihat: torch.Tensor, pi_star: torch.Tensor) -> torch.Tensor:
        """
        Calculate row-wise Dirichlet params with sticky prior and counts.
        xihat: [Kp1,Kp1], pi_star: [Kp1]
        returns phi: [Kp1,Kp1]
        """
        Kp1 = pi_star.shape[0]
        phi = self.p.alpha * pi_star.view(1, Kp1) + self.p.kappa * torch.eye(Kp1, device=xihat.device, dtype=xihat.dtype) + xihat
        return phi

    @torch.no_grad()
    def _update_transitions(self, phihat: torch.Tensor):
        """
        Update row-wise Dirichlet params with sticky prior and counts.
        """
        self.dir.phi = phihat

    def _optimize_pi_from_r1(self, r1: torch.Tensor, steps: int = 200, lr: float = 0.05) -> torch.Tensor:
        """
        Optimize global sticks β (via logits u_beta) using a β-restricted ELBO:
            L(β) = E_q[log p(Φ | π)] + E_q[log p(h1 | π)] + log p(β),
        with π = concat(SB(σ(u_beta)), remainder).
        Args:
            r1: average initial-state responsibilities (Kp1,)
        Returns:
            π* (detached, shape [Kp1])
        """
        K = self.p.K
        alpha, kappa, gamma = self.p.alpha, self.p.kappa, self.p.gamma
        ElogA = self._ElogA().detach()
        r1 = r1.detach()

        opt = torch.optim.Adam([self.u_beta], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            beta = torch.sigmoid(self.u_beta)          # [K]
            piK, rest = stick_breaking(beta)           # [K], scalar
            pi = torch.cat([piK, rest.view(1)], dim=0) # [Kp1]
            # Dirichlet prior rows a_k = α π + κ δ_k, with π sum=1
            a = alpha * pi.unsqueeze(0) + kappa * torch.eye(K + 1, device=pi.device, dtype=pi.dtype)  # [Kp1,Kp1]
            a = a.clamp(min=1e-8)  # avoid NaNs
            # E_q[log p(Φ | π)] under q(Φ) with row-wise Dirichlet φ
            const = (K + 1) * torch.lgamma(torch.tensor(alpha + kappa, device=pi.device, dtype=pi.dtype))
            L1 = ((a - 1.0) * ElogA).sum() - torch.lgamma(a).sum() + const
            # E_q[log p(h1 | π)]
            L2 = torch.sum(r1 * torch.log(torch.clamp(pi, min=1e-30)))
            # log p(β) with Beta(1,γ) sticks: ∑ (γ-1) log(1-β_k)
            L3 = (gamma - 1.0) * torch.sum(torch.log(torch.clamp(1.0 - beta, min=1e-30)))
            loss = -(L1 + L2 + L3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.u_beta], max_norm=10.0)
            opt.step()
            with torch.no_grad():
                self.u_beta.clamp_(-10.0, 10.0)

        with torch.no_grad():
            beta = torch.sigmoid(self.u_beta)
            piK, rest = stick_breaking(beta)
            pi = torch.cat([piK, rest.view(1)], dim=0)
        return pi.detach()
    
    @staticmethod
    def _optimize_u_beta(
        u_beta: torch.Tensor,
        r1: torch.Tensor,
        ElogA: torch.Tensor,
        alpha: float,
        kappa: float,
        gamma: float,
        K: int,
        steps: int = 200,
        lr: float = 0.05
    ) -> torch.Tensor:
        """
        Optimize global sticks β (via logits u_beta) using a β-restricted ELBO:
            L(β) = E_q[log p(Φ | π)] + E_q[log p(h1 | π)] + log p(β),
        with π = concat(SB(σ(u_beta)), remainder).
        Args:
            u_beta: initial logits for β (K,)
            r1: average initial-state responsibilities (Kp1,)
            ElogA: expected log transition matrix (Kp1,Kp1)
            alpha, kappa, gamma: sticky HDP-HMM hyperparameters
            K: number of explicit states
            steps: optimization steps
            lr: learning rate
        Returns:
            optimized u_beta (detached, shape [K])
        """
        ElogA = ElogA.detach()
        r1 = r1.detach()

        u_beta_opt = u_beta.clone().requires_grad_(True)
        opt = torch.optim.Adam([u_beta_opt], lr=lr)
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                beta = torch.sigmoid(u_beta_opt)          # [K]
                piK, rest = stick_breaking(beta)           # [K], scalar
                pi = torch.cat([piK, rest.view(1)], dim=0)  # [Kp1]
                # Dirichlet prior rows a_k = α π + κ δ_k, with π sum=1
                a = alpha * pi.unsqueeze(0) + kappa * torch.eye(K + 1, device=pi.device, dtype=pi.dtype)  # [Kp1,Kp1]
                a = a.clamp(min=1e-8)  # avoid NaNs
                # E_q[log p(Φ | π)] under q(Φ) with row-wise Dirichlet φ
                const = (K + 1) * torch.lgamma(torch.tensor(alpha + kappa, device=pi.device, dtype=pi.dtype))
                L1 = ((a - 1.0) * ElogA).sum() - torch.lgamma(a).sum() + const
                # E_q[log p(h1 | π)]
                L2 = torch.sum(r1 * torch.log(torch.clamp(pi, min=1e-30)))
                # log p(β) with Beta(1,γ) sticks: ∑ (γ-1) log(1-β_k)
                L3 = (gamma - 1.0) * torch.sum(torch.log(torch.clamp(1.0 - beta, min=1e-30)))
                loss = -(L1 + L2 + L3)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([u_beta_opt], max_norm=10.0)
                opt.step()
                with torch.no_grad():
                    u_beta_opt.clamp_(-10.0, 10.0)

        with torch.no_grad():
            return u_beta_opt.detach()

    # ---- streaming buffers ---------------------------------------------------
    def _alloc_stream_buffers(self):
        if self._stream_allocated:
            return
        Kp1, D = self.niw.mu.shape[0], self.p.D
        dev, dt = self.mu0.device, self.mu0.dtype
        self.register_buffer("S_Nk",     torch.zeros(Kp1,       device=dev, dtype=dt))
        self.register_buffer("S_M1",     torch.zeros(Kp1, D,    device=dev, dtype=dt))
        self.register_buffer("S_M2",     torch.zeros(Kp1, D, D, device=dev, dtype=dt))
        self.register_buffer("S_counts", torch.zeros(Kp1, Kp1,  device=dev, dtype=dt))
        self.register_buffer("S_steps",  torch.tensor(0.0,      device=dev, dtype=dt))
        self._stream_allocated = True

    @torch.no_grad()
    def streaming_reset(self):
        self._alloc_stream_buffers()
        self.S_Nk.zero_(); self.S_M1.zero_(); self.S_M2.zero_(); self.S_counts.zero_()
        self.S_steps.fill_(0.0)

    # ---- ELBO component calculations ----------------------------------------
    @staticmethod
    def _hmm_local_free_energy_fixed_qh(r, xi, log_pi, ElogA, logB, eps=1e-12, t0_override=None) -> Dict[str, float]:
        # r: [T,K], xi: [T-1,K,K], logB: [T,K], ElogA: [K,K]
        t0 = 0 if t0_override is None else t0_override
        init   = (r[t0] * log_pi).sum()
        trans  = (xi * ElogA).sum()
        emit   = (r * logB).sum()
        # Bethe entropy for a chain (exact)
        H_pairs = -(xi.clamp_min(eps) * (xi.clamp_min(eps).log())).sum()
        # sum over internal nodes only (2..T-1)
        H_nodes = -(r[1:-1].clamp_min(eps) * r[1:-1].clamp_min(eps).log()).sum()
        H_q = H_pairs - H_nodes
        return {
            'init': init.item(),
            'trans': trans.item(),
            'emit': emit.item(),
            'entropy': H_q.item()
        }
    
    @torch.no_grad()
    def _calculate_dirichlet_term(
        self,
        phi: torch.Tensor,                             # [Kp1,Kp1] Dirichlet posterior parameters
        pi_full: torch.Tensor,                         # [Kp1] state probabilities
        ElogA: Optional[torch.Tensor] = None           # [Kp1,Kp1] expected log transition matrix (optional)
    ) -> float:
        """
        Calculate the Dirichlet term in ELBO: E_q[log p(Φ)] - E_q[log q(Φ)] = -KL(q(Φ)||p(Φ))
        
        Args:
            phi: Dirichlet posterior parameters [Kp1,Kp1]
            pi_full: State probabilities [Kp1]  
            ElogA: Expected log transition matrix (optional, computed if None)
            
        Returns:
            Dirichlet term contribution to ELBO (scalar)
        """
        Kp1 = phi.shape[0]
        
        if ElogA is None:
            ElogA = StickyHDPHMMVI._calc_ElogA(phi)
        
        # Dirichlet prior parameters: a_k = α π + κ δ_k
        a = (self.p.alpha * pi_full.unsqueeze(0) + 
             self.p.kappa * torch.eye(Kp1, device=self.mu0.device, dtype=self.mu0.dtype))
        
        # KL(q||p) = logC(φ) - logC(a) + (φ-a)ᵀ E[logΦ]
        # ELBO term is -KL, so we have logC(a) - logC(φ) + (a-φ)ᵀ E[logΦ]
        dir_term = (StickyHDPHMMVI._dirichlet_logC(a) - StickyHDPHMMVI._dirichlet_logC(phi) + 
                   ((a - phi) * ElogA).sum()).item()
        
        return dir_term
    
    @torch.no_grad() 
    def _calculate_beta_prior_term(self, u_beta: torch.Tensor) -> float:
        """
        Calculate the stick-breaking prior term for β: E_q[log p(β|γ)]
        
        Args:
            u_beta: Beta logits [K]
            
        Returns:
            Beta prior term contribution to ELBO (scalar)
        """
        beta = torch.sigmoid(u_beta)
        logp_beta = ((self.p.gamma - 1.0) * torch.log(torch.clamp(1.0 - beta, min=1e-30))).sum().item()
        return logp_beta

    # ---- ELBO calculation ---------------------------------------------------
    @torch.no_grad()
    def calculate_elbo(
        self,
        mu_t: torch.Tensor,                            # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,     # same shape as mu_t
        F_t: Optional[torch.Tensor] = None,            # [T,D,R] or [B,T,D,R]
        mask: Optional[torch.Tensor] = None,           # [T] or [B,T]
        mu_hat: Optional[torch.Tensor] = None,         # [Kp1,D] - if None, uses self.niw.mu
        k_hat: Optional[torch.Tensor] = None,          # [Kp1] - if None, uses self.niw.kappa
        Psi_hat: Optional[torch.Tensor] = None,        # [Kp1,D,D] - if None, uses self.niw.Psi
        nu_hat: Optional[torch.Tensor] = None,         # [Kp1] - if None, uses self.niw.nu
        phi: Optional[torch.Tensor] = None,            # [Kp1,Kp1] - if None, uses self.dir.phi
        u_beta: Optional[torch.Tensor] = None,         # [K] - if None, uses self.u_beta
        rhat: Optional[torch.Tensor] = None,           # [B,T,Kp1] - if provided, skips FB
        xihat: Optional[torch.Tensor] = None           # [B,T-1,Kp1,Kp1] - if provided, skips FB
    ) -> Dict[str, float]:
        """
        Calculate ELBO given statistics and parameters.
        
        ELBO = E[log p(data, states)] - E[log q(states, params)]
             = LL + Dirichlet_term + NIW_term + Beta_prior_term
        
        Args:
            mu_t, diag_var_t, F_t, mask: Data tensors
            mu_hat, k_hat, Psi_hat, nu_hat: NIW posterior parameters (optional, uses current if None)
            phi: Dirichlet posterior parameters (optional, uses current if None) 
            u_beta: Beta logits (optional, uses current if None)
            rhat, xihat: Precomputed posteriors (optional, if provided skips FB)
            
        Returns:
            Dict containing ELBO and its components
        """
        # Use current parameters if not provided
        if mu_hat is None: mu_hat = self.niw.mu
        if k_hat is None: k_hat = self.niw.kappa
        if Psi_hat is None: Psi_hat = self.niw.Psi
        if nu_hat is None: nu_hat = self.niw.nu
        if phi is None: phi = self.dir.phi
        if u_beta is None: u_beta = self.u_beta
        
        # Ensure [B,T,D] shapes for uniform processing
        if mu_t.dim() == 2:
            mu_t = mu_t.unsqueeze(0)
            if diag_var_t is not None: diag_var_t = diag_var_t.unsqueeze(0)
            if F_t is not None: F_t = F_t.unsqueeze(0)
            if mask is not None: mask = mask.unsqueeze(0)
        
        B, T, D = mu_t.shape
        Kp1 = mu_hat.shape[0]
        
        # Calculate emission log-likelihoods
        logB = StickyHDPHMMVI.expected_emission_loglik(
            mu_hat, k_hat, Psi_hat, nu_hat,
            mu_t, diag_var_t, F_t, mask)  # [B,T,Kp1]
        
        # Calculate transition and initial state parameters
        ElogA = StickyHDPHMMVI._calc_ElogA(phi)  # [Kp1,Kp1]
        pi_full = StickyHDPHMMVI._calc_Epi(u_beta)  # [Kp1]
        log_pi = torch.log(torch.clamp(pi_full, min=1e-30))
        
        # 1. Expected complete data log-likelihood (LL term)
        init = 0.0
        trans = 0.0
        emit = 0.0
        entropy = 0.0
        for b in range(B):
            if rhat is not None and xihat is not None:
                r_b = rhat[b]
                xi_b = xihat[b]
            else:
                r_b, xi_b, _ = StickyHDPHMMVI.forward_backward(log_pi, ElogA, logB[b])
            if mask is not None:
                m = mask[b]
                pair_m = (m[:-1] * m[1:]).view(-1, 1, 1)  # [T-1,1,1]
                r_b = r_b * m.view(-1, 1)  # [T,Kp1]
                xi_b = xi_b * pair_m
                t0 = int(torch.nonzero(m, as_tuple=False)[0])
            else:
                t0 = 0
            stats = StickyHDPHMMVI._hmm_local_free_energy_fixed_qh(r_b, xi_b, log_pi, ElogA, logB[b], t0_override=t0)
            init += stats['init']
            trans += stats['trans']
            emit += stats['emit']
            entropy += stats['entropy']
        
        # 2. Dirichlet term: E_q[log p(Φ)] - E_q[log q(Φ)] = -KL(q(Φ)||p(Φ))
        dir_term = self._calculate_dirichlet_term(phi, pi_full, ElogA)
        
        # 3. Stick-breaking prior term for β: E_q[log p(β|γ)]
        logp_beta = self._calculate_beta_prior_term(u_beta)
        
        # 4. NIW term: ∑_k [E log p(μ_k,Σ_k) - E log q(μ_k,Σ_k)]
        niw_term = float(self._niw_elbo_term(mu_hat, k_hat, Psi_hat, nu_hat).item())

        # Complete ELBO
        elbo = init + trans + emit + entropy + dir_term + niw_term + logp_beta
        
        return {
            "elbo": elbo,
            "init": init,
            "trans": trans,
            "emit": emit,
            "entropy": entropy,
            "dir_term": dir_term,
            "niw_term": niw_term,
            "logp_beta": logp_beta
        }
    
    # ---- unified update (streaming & non-streaming) --------------------------
    @torch.no_grad()
    def update(
        self,
        mu_t: torch.Tensor,                            # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,     # same shape as mu_t
        F_t: Optional[torch.Tensor] = None,            # [T,D,R] or [B,T,D,R]
        mask: Optional[torch.Tensor] = None,           # [T] or [B,T]
        max_iters: int = 7,                            # inner full-loop limit
        tol: float = 1e-4,                             # ELBO gain tolerance
        elbo_drop_tol: float = 10.0,                   # ELBO drop tolerance for early stopping
        rho: Optional[float] = 1.0,                    # 1.0 => non-streaming; 0<rho<1 => EMA
        optimize_pi: bool = True,                      # optimize π using mean r1
        pi_steps: int = 200,                           # π opt steps
        pi_lr: float = 0.05,                           # π opt learning rate
        offline: bool = False,                         # offline π opt (full data) if True
        logger: Optional[logging.Logger] = None        # for debug output
    ) -> Dict[str, torch.Tensor]:
        """
        Batch-global full inner loop:
        (1) build logB from current NIW
        (2) FB for all sequences -> rhat/xi, accumulate batch stats
        (3) blend stats (EMA/non-streaming) -> update NIW
        (4) optionally optimize π using mean r1
        (5) update transitions with blended counts via _update_transitions
        (6) evaluate full inner ELBO; early stop if plateau or large drop

        Early stopping logic:
        - If ELBO increases but change < tol: stop and use current parameters
        - If ELBO drops by more than elbo_drop_tol: stop and use previous best parameters
        - Otherwise continue optimization

        Returns diagnostics from the last pass.
        """
        self._alloc_stream_buffers()

        # Ensure [B,T,D] shapes for a uniform code path
        if mu_t.dim() == 2:
            mu_t = mu_t.unsqueeze(0)
            if diag_var_t is not None: diag_var_t = diag_var_t.unsqueeze(0)
            if F_t is not None: F_t = F_t.unsqueeze(0)
            if mask is not None: mask = mask.unsqueeze(0)

        B, T, D = mu_t.shape
        Kp1 = self.niw.mu.shape[0]

        # Resolve streaming coefficients
        if rho is None:
            rho_eff = self.stream_rho
        else:
            rho_eff = float(max(0.0, min(1.0, rho)))
        rho_tr = self.stream_rho_trans if self.stream_rho_trans is not None else rho_eff
        rho_tr = float(max(0.0, min(1.0, rho_tr)))

        # ELBO trackers
        elbo_history = []
        total_ll_history = []
        dir_term_history = []
        iw_term_history = []
        logp_beta_history = []

        last_out = None
        this_mu_hat = self.niw.mu.clone()
        this_k_hat = self.niw.kappa.clone()
        this_Psi_hat = self.niw.Psi.clone()
        this_nu_hat = self.niw.nu.clone()
        this_phi = self.dir.phi.clone()
        this_u_beta = self.u_beta.detach().clone()
        this_S_Nk = self.S_Nk.clone()
        this_S_M1 = self.S_M1.clone()
        this_S_M2 = self.S_M2.clone()
        this_S_counts = self.S_counts.clone()
        early_stopping = False
        
        # Track best parameters for early stopping
        best_mu_hat = None
        best_k_hat = None
        best_Psi_hat = None
        best_nu_hat = None
        best_phi = None
        best_u_beta = None
        best_S_Nk = None
        best_S_M1 = None
        best_S_M2 = None
        best_S_counts = None
        
        elbo_before_loop = self.calculate_elbo(
            mu_t, diag_var_t, F_t, mask,
            this_mu_hat, this_k_hat, this_Psi_hat, this_nu_hat,
            this_phi, this_u_beta
        )
        init_before_val = elbo_before_loop["init"]
        trans_before_val = elbo_before_loop["trans"]
        emit_before_val = elbo_before_loop["emit"]
        entropy_before_val = elbo_before_loop["entropy"]
        ll_before_val = init_before_val + trans_before_val + emit_before_val + entropy_before_val
        dir_before_val = elbo_before_loop["dir_term"]
        niw_before_val = elbo_before_loop["niw_term"]
        logp_beta_before_val = elbo_before_loop["logp_beta"]
        if logger is not None:
            logger.debug(f"Initial ELBO: {elbo_before_loop['elbo']:.4f}")
            logger.debug(f"- Init {init_before_val:.4f}")
            logger.debug(f"- Trans {trans_before_val:.4f}")
            logger.debug(f"- Emit {emit_before_val:.4f}")
            logger.debug(f"- Entropy {entropy_before_val:.4f}")
            logger.debug(f"- Dir {dir_before_val:.4f}")
            logger.debug(f"- NIW {niw_before_val:.4f}")
            logger.debug(f"- Beta prior {logp_beta_before_val:.4f}")

        # ---- inner full loop ----------------------------------------------------
        for it in range(max(1, int(max_iters))):
            # (1) Emission potentials with current NIW
            this_logB = StickyHDPHMMVI.expected_emission_loglik(
                this_mu_hat, this_k_hat, this_Psi_hat, this_nu_hat,
                mu_t, diag_var_t, F_t, mask)  # [B,T,Kp1]
            this_ElogA = StickyHDPHMMVI._calc_ElogA(this_phi)  # [Kp1,Kp1]
            this_pi_full = StickyHDPHMMVI._calc_Epi(this_u_beta)  # [Kp1]
            this_log_pi = torch.log(torch.clamp(this_pi_full, min=1e-30))

            # (2) FB across batch -> batch-level sufficient statistics
            this_acc_counts = torch.zeros(Kp1, Kp1, device=self.mu0.device, dtype=self.mu0.dtype)
            this_acc_Nk     = torch.zeros(Kp1,    device=self.mu0.device, dtype=self.mu0.dtype)
            this_acc_M1     = torch.zeros(Kp1, D, device=self.mu0.device, dtype=self.mu0.dtype)
            this_acc_M2     = torch.zeros(Kp1, D, D, device=self.mu0.device, dtype=self.mu0.dtype)
            this_r1_sum = torch.zeros(Kp1, device=self.mu0.device, dtype=self.mu0.dtype)
            rhat_list = []
            xihat_list = []

            for b in range(B):
                rhat, xihat, ll = StickyHDPHMMVI.forward_backward(this_log_pi, this_ElogA, this_logB[b])
                Nk, M1, M2 = self._moments_from_encoder(
                    mu_t[b],
                    rhat,
                    (diag_var_t[b] if diag_var_t is not None else None),
                    (F_t[b] if F_t is not None else None),
                    (mask[b] if mask is not None else None)
                )
                this_acc_Nk += Nk; this_acc_M1 += M1; this_acc_M2 += M2
                if mask is not None:
                    m = mask[b]
                    pair_m = (m[:-1] * m[1:]).view(-1, 1, 1)  # [T-1,1,1]
                    xihat = xihat * pair_m  # zero out transitions from/to masked frames
                    t0 = int(torch.nonzero(m, as_tuple=False)[0])
                else:
                    t0 = 0
                this_acc_counts += xihat.sum(dim=0)
                this_r1_sum += rhat[t0]
                last_out = {"rhat": rhat.detach(), "xihat": xihat.detach(), "loglik": torch.tensor(ll)}
                rhat_list.append(rhat)
                xihat_list.append(xihat)
            rhat_all = torch.stack(rhat_list, dim=0)       # [B,T,Kp1]
            xihat_all = torch.stack(xihat_list, dim=0)     # [B,T-1,Kp1,Kp1]
            elbo_after_fb_all = self.calculate_elbo(
                mu_t, diag_var_t, F_t, mask,
                this_mu_hat, this_k_hat, this_Psi_hat, this_nu_hat,
                this_phi, this_u_beta, rhat_all, xihat_all
            )
            init_before_val = elbo_after_fb_all["init"]
            trans_before_val = elbo_after_fb_all["trans"]
            emit_before_val = elbo_after_fb_all["emit"]
            entropy_before_val = elbo_after_fb_all["entropy"]
            this_ll = init_before_val + trans_before_val + emit_before_val + entropy_before_val
            elbo_after_fb = this_ll + dir_before_val + niw_before_val + logp_beta_before_val
            if logger is not None:
                logger.debug(f"Iter {it}: ELBO after FB: {elbo_after_fb:.4f} (LL {this_ll:.4f}, Δ={(this_ll - ll_before_val):.4f})")

            # (3) Blend stats (EMA or full replace) and update NIW
            this_S_Nk     = (1.0 if offline else (1.0 - rho_eff)) * self.S_Nk     + rho_eff * this_acc_Nk
            this_S_M1     = (1.0 if offline else (1.0 - rho_eff)) * self.S_M1     + rho_eff * this_acc_M1
            this_S_M2     = (1.0 if offline else (1.0 - rho_eff)) * self.S_M2     + rho_eff * this_acc_M2
            this_S_counts = (1.0 if offline else (1.0 - rho_tr)) * self.S_counts + rho_tr  * this_acc_counts

            this_mu_hat, this_k_hat, this_Psi_hat, this_nu_hat = \
                self._calc_NIW_posterior(this_S_Nk, this_S_M1, this_S_M2)
                
            elbo_after_niw_all = self.calculate_elbo(
                mu_t, diag_var_t, F_t, mask,
                this_mu_hat, this_k_hat, this_Psi_hat, this_nu_hat,
                this_phi, this_u_beta, rhat_all, xihat_all
            )
            niw_after = elbo_after_niw_all["niw_term"]
            emit_after = elbo_after_niw_all["emit"]
            assert abs(init_before_val - elbo_after_niw_all["init"]) < 1e-4, "Initial term changed after NIW update!"
            assert abs(trans_before_val - elbo_after_niw_all["trans"]) < 1e-4, "Transition term changed after NIW update!"
            assert abs(entropy_before_val - elbo_after_niw_all["entropy"]) < 1e-4, "Entropy term changed after NIW update!"
            assert abs(dir_before_val - elbo_after_niw_all["dir_term"]) < 1e-4, "Dirichlet term changed after NIW update!"
            assert abs(logp_beta_before_val - elbo_after_niw_all["logp_beta"]) < 1e-4, "Beta prior term changed after NIW update!"
            this_ll = this_ll - emit_before_val + emit_after
            elbo_after_niw = this_ll + dir_before_val + niw_after + logp_beta_before_val
            if logger is not None:
                logger.debug(f"Iter {it}: ELBO after NIW update: {elbo_after_niw:.4f} (Emit LL {emit_after:.4f}, NIW term {niw_after:.4f}, Δ={(niw_after + emit_after - niw_before_val - emit_before_val):.4f})")
            emit_before_val = emit_after
            
            # (4) Update transitions from *blended* counts using helper
            #     Make a single "time-slice" that sums to S_counts so the helper
            #     sees exactly those counts.
            this_phi = self._calc_dir_posterior(this_S_counts, this_pi_full)
            this_ElogA = StickyHDPHMMVI._calc_ElogA(this_phi)  # [Kp1,Kp1]
            
            elbo_after_dir_all = self.calculate_elbo(
                mu_t, diag_var_t, F_t, mask,
                this_mu_hat, this_k_hat, this_Psi_hat, this_nu_hat,
                this_phi, this_u_beta, rhat_all, xihat_all
            )
            dir_after = elbo_after_dir_all["dir_term"]
            trans_after = elbo_after_dir_all["trans"]
            assert abs(init_before_val - elbo_after_dir_all["init"]) < 1e-4, "Initial term changed after Dirichlet update!"
            assert abs(emit_before_val - elbo_after_dir_all["emit"]) < 1e-4, "Emission term changed after Dirichlet update!"
            assert abs(entropy_before_val - elbo_after_dir_all["entropy"]) < 1e-4, "Entropy term changed after Dirichlet update!"
            assert abs(niw_after - elbo_after_dir_all["niw_term"]) < 1e-4, "NIW term changed after Dirichlet update!"
            assert abs(logp_beta_before_val - elbo_after_dir_all["logp_beta"]) < 1e-4, "Beta prior term changed after Dirichlet update!"
            this_ll = this_ll - trans_before_val + trans_after
            elbo_after_dir = this_ll + dir_after + niw_after + logp_beta_before_val
            if logger is not None:
                logger.debug(f"Iter {it}: ELBO after Dirichlet update: {elbo_after_dir:.4f} (Trans LL {trans_after:.4f}, Dir term {dir_after:.4f}, Δ={(dir_after + trans_after - dir_before_val - trans_before_val):.4f})")
            trans_before_val = trans_after
            dir_before_val = dir_after

            # (5) Optimize β/π from average r1 across sequences
            if optimize_pi and B > 0:
                r1_mean = (this_r1_sum / B).clamp_min(1e-12)
                r1_mean = r1_mean / r1_mean.sum()
                this_u_beta = StickyHDPHMMVI._optimize_u_beta(
                    this_u_beta, r1_mean, this_ElogA,
                    self.p.alpha, self.p.kappa, self.p.gamma, self.p.K,
                    steps=pi_steps, lr=pi_lr
                )
                elbo_after_beta_all = self.calculate_elbo(
                    mu_t, diag_var_t, F_t, mask,
                    this_mu_hat, this_k_hat, this_Psi_hat, this_nu_hat,
                    this_phi, this_u_beta, rhat_all, xihat_all
                )
                logp_beta_after = elbo_after_beta_all["logp_beta"]
                init_after = elbo_after_beta_all["init"]
                dir_after = elbo_after_beta_all["dir_term"]
                assert abs(trans_before_val - elbo_after_beta_all["trans"]) < 1e-4, "Transitional term changed after β update!"
                assert abs(emit_before_val - elbo_after_beta_all["emit"]) < 1e-4, "Emission term changed after β update!"
                assert abs(entropy_before_val - elbo_after_beta_all["entropy"]) < 1e-4, "Entropy term changed after β update!"
                assert abs(niw_after - elbo_after_beta_all["niw_term"]) < 1e-4, "NIW term changed after β update!"
                this_ll = this_ll - init_before_val + init_after
                elbo_after_beta = this_ll + dir_after + niw_after + logp_beta_after
                if logger is not None:
                    logger.debug(f"Iter {it}: ELBO after β update: {elbo_after_beta:.4f} (Init LL {init_after:.4f}, Dir term {dir_after:.4f}, logp(β) {logp_beta_after:.4f}, Δ={(logp_beta_after + init_after + dir_after - logp_beta_before_val - init_before_val - dir_before_val):.4f})")
                init_before_val = init_after
                dir_before_val = dir_after
            else:
                logp_beta_after = logp_beta_before_val
                elbo_after_beta = elbo_after_dir

            elbo_history.append(elbo_after_beta)
            total_ll_history.append(this_ll) 
            dir_term_history.append(dir_after)
            iw_term_history.append(niw_after)
            logp_beta_history.append(logp_beta_after)
            ll_before_val = this_ll
            niw_before_val = niw_after
            logp_beta_before_val = logp_beta_after

            # Improved Early stopping logic
            if len(elbo_history) == 1:
                # First iteration: save as best
                best_mu_hat = this_mu_hat.clone()
                best_k_hat = this_k_hat.clone()
                best_Psi_hat = this_Psi_hat.clone()
                best_nu_hat = this_nu_hat.clone()
                best_phi = this_phi.clone()
                best_u_beta = this_u_beta.clone()
                best_S_Nk = this_S_Nk.clone()
                best_S_M1 = this_S_M1.clone()
                best_S_M2 = this_S_M2.clone()
                best_S_counts = this_S_counts.clone()
            elif len(elbo_history) > 1:
                prev_elbo = elbo_history[-2]
                curr_elbo = elbo_history[-1]
                elbo_change = curr_elbo - prev_elbo
                
                if elbo_change > 0:
                    # ELBO improved: update best parameters
                    best_mu_hat = this_mu_hat.clone()
                    best_k_hat = this_k_hat.clone()
                    best_Psi_hat = this_Psi_hat.clone()
                    best_nu_hat = this_nu_hat.clone()
                    best_phi = this_phi.clone()
                    best_u_beta = this_u_beta.clone()
                    best_S_Nk = this_S_Nk.clone()
                    best_S_M1 = this_S_M1.clone()
                    best_S_M2 = this_S_M2.clone()
                    best_S_counts = this_S_counts.clone()
                    
                    # Check if improvement is small enough to stop
                    if elbo_change < tol:
                        early_stopping = True
                        # Use current (best) iteration parameters
                        self._update_NIW(this_mu_hat, this_k_hat, this_Psi_hat, this_nu_hat)
                        self._update_transitions(this_phi)
                        self._update_u_beta(this_u_beta)
                        self._update_moments(this_S_Nk, this_S_M1, this_S_M2, this_S_counts)
                        break
                else:
                    # ELBO decreased: check if drop is too large
                    if abs(elbo_change) > elbo_drop_tol:
                        early_stopping = True
                        # Use previous (best) iteration parameters
                        self._update_NIW(best_mu_hat, best_k_hat, best_Psi_hat, best_nu_hat)
                        self._update_transitions(best_phi)
                        self._update_u_beta(best_u_beta)
                        self._update_moments(best_S_Nk, best_S_M1, best_S_M2, best_S_counts)
                        break
                
        if not early_stopping:
            # Use the best parameters from the last iteration
            self._update_NIW(best_mu_hat if best_mu_hat is not None else this_mu_hat, 
                            best_k_hat if best_k_hat is not None else this_k_hat, 
                            best_Psi_hat if best_Psi_hat is not None else this_Psi_hat, 
                            best_nu_hat if best_nu_hat is not None else this_nu_hat)
            self._update_transitions(best_phi if best_phi is not None else this_phi)
            self._update_u_beta(best_u_beta if best_u_beta is not None else this_u_beta)
            self._update_moments(best_S_Nk if best_S_Nk is not None else this_S_Nk, 
                               best_S_M1 if best_S_M1 is not None else this_S_M1, 
                               best_S_M2 if best_S_M2 is not None else this_S_M2, 
                               best_S_counts if best_S_counts is not None else this_S_counts)

        return {
            "rhat": last_out["rhat"],
            "xihat": last_out["xihat"],
            "loglik": last_out["loglik"],
            "pi_star": self._Epi().detach(),   # [K+1]
            "ElogA": self._ElogA().detach(),   # [K+1,K+1]
            "elbo_history": torch.tensor(elbo_history),
            "inner_elbo": elbo_history[-1] if len(elbo_history) > 0 else float('nan'),
            "n_iterations": len(elbo_history)
        }


    # ---- accessors & (de)serialization --------------------------------------
    def get_posterior_params(self) -> Dict[str, torch.Tensor]:
        return {
            "mu": self.niw.mu.detach(),        # [Kp1,D]
            "kappa": self.niw.kappa.detach(),  # [Kp1]
            "Psi": self.niw.Psi.detach(),      # [Kp1,D,D]
            "nu": self.niw.nu.detach(),        # [Kp1]
            "phi": self.dir.phi.detach(),      # [Kp1,Kp1]
            "beta_u": self.u_beta.detach(),    # [K]
        }

    def get_emission_expectations(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.niw.mu, self._get_E_Lambda(), self._get_E_logdet_Lambda()

    def load_posterior_params(self, params: Dict[str, torch.Tensor]) -> None:
        dev, dt = self.mu0.device, self.mu0.dtype
        self.niw.mu    = params["mu"].to(device=dev, dtype=dt)
        self.niw.kappa = params["kappa"].to(device=dev, dtype=dt)
        self.niw.Psi   = params["Psi"].to(device=dev, dtype=dt)
        self.niw.nu    = params["nu"].to(device=dev, dtype=dt)
        if "phi" in params:     self.dir.phi.copy_(params["phi"].to(device=dev, dtype=dt))
        if "beta_u" in params:  self.u_beta.data.copy_(params["beta_u"].to(device=dev, dtype=dt))
        self._cache_fresh = False

    # --- compact accessors ----------------------------------------------------
    @torch.no_grad()
    def get_rho_emission(self) -> torch.Tensor:
        """Alias for streaming rho used in code paths that expect get_rho_emission()."""
        return torch.tensor(float(self.stream_rho), device=self.mu0.device, dtype=self.mu0.dtype)

    @torch.no_grad()
    def get_rho_transition(self) -> torch.Tensor:
        """Alias for transition streaming rho."""
        val = float(self.stream_rho_trans) if self.stream_rho_trans is not None else float('nan')
        return torch.tensor(val, device=self.mu0.device, dtype=self.mu0.dtype)

    # ---- diagnostics ---------------------------------------------------------
    @torch.no_grad()
    def diagnostics(
        self,
        mu_t: torch.Tensor,                            # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,
        F_t: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor | float]:
        """
        Quick diagnostics with current params.
        """
        # responsibilities with current params
        logB = StickyHDPHMMVI.expected_emission_loglik(
            self.niw.mu, self.niw.kappa, self.niw.Psi, self.niw.nu,
            mu_t, diag_var_t, F_t, mask
        )
        is_batched = (logB.dim() == 3)
        B = logB.size(0) if is_batched else 1

        r_all = []
        xi_all = []
        ll_total = 0.0
        steps = 0
        for b in range(B):
            _logB = logB[b] if is_batched else logB
            with torch.no_grad():
                pi_full = self._Epi()
                log_pi = torch.log(torch.clamp(pi_full, min=1e-30))
                rhat, xihat, ll = StickyHDPHMMVI.forward_backward(log_pi, self._ElogA(), _logB)
            r_all.append(rhat)
            xi_all.append(xihat)
            # count valid steps
            T = _logB.size(0)
            if mask is not None:
                _msk = (mask[b] if is_batched else mask).to(_logB.dtype)
                steps += int(_msk.sum().item())
            else:
                steps += T
            ll_total += ll

        r_cat = torch.cat(r_all, dim=0)   # [sum_T, Kp1]
        xi_cat = torch.cat(xi_all, dim=0) if len(xi_all) > 0 else torch.zeros(0)
        Nk = r_cat.sum(dim=0) + 1e-12
        pi_hat = (Nk / Nk.sum()).cpu()

        # entropy over states
        ent = (- (r_cat.clamp_min(1e-9) * r_cat.clamp_min(1e-9).log()).sum(dim=1)).mean().item()

        # transition diagnostics from ElogA proxy
        EA = self._EA().cpu()  # [Kp1,Kp1]
        diag_ratio = float(torch.diagonal(EA, dim1=0, dim2=1).mean().item())

        # effective number of skills
        effK = float(torch.exp(-(pi_hat * (pi_hat + 1e-12).log()).sum()).item())

        topk_vals, topk_idx = torch.topk(pi_hat, k=min(5, self.niw.mu.shape[0]))
        avg_ll_per_step = float(ll_total / max(steps, 1))

        paths = self.viterbi_paths(mu_t, diag_var_t, F_t, mask=mask)
        Kp1 = self.niw.mu.size(0)
        all_lens = [[] for _ in range(Kp1)]
        for z in paths:
            lens, _ = StickyHDPHMMVI.dwell_stats_from_path(z, Kp1)
            for k in range(Kp1):
                all_lens[k].extend(lens[k])

        p_stay = EA.diag()                     # E[A_kk]
        mean_geo = (1.0 / (1.0 - p_stay)).clamp(max=1e6)  # expected dwell for geometric
        emp_mean = torch.tensor([np.mean(L) if len(L) else 0. for L in all_lens])

        return {
            "avg_loglik_per_step": avg_ll_per_step,
            "state_entropy": ent,
            "occupancy_pi_hat": pi_hat,
            "effective_K": effK,
            "top5_pi": topk_vals,
            "top5_idx": topk_idx,
            "stickiness_diag_mean": diag_ratio,
            "p_stay": p_stay,
            "expected_dwell_length_per_state": mean_geo,
            "empirical_dwell_length_per_state": emp_mean,
            "all_lengths": all_lens
        }

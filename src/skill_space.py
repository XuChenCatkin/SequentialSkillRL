# ===== skill_space.py ======================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------- helpers ------------------------------------------------------------

def stick_breaking(beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    beta: [K] with elements in (0,1)
    returns pi: [K], rest: scalar (pi_{K+1})
    """
    K = beta.numel()
    one = torch.ones(1, device=beta.device, dtype=beta.dtype)
    cprod = torch.cumprod(torch.cat([one, 1 - beta[:-1]], dim=0), dim=0)  # [K]
    pi = beta * cprod  # [K]
    rest = torch.prod(1 - beta)
    return pi, rest

def chol_inv_logdet(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    S: [..., D, D] SPD
    returns (S^{-1}, log|S|)
    """
    L = torch.linalg.cholesky(S)                             # [..., D, D]
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
    inv = torch.cholesky_inverse(L)                          # [..., D, D]
    return inv, logdet

# --------- priors/posteriors containers --------------------------------------

@dataclass
class NIWPrior:
    mu0: torch.Tensor        # [D]
    kappa0: float
    Psi0: torch.Tensor       # [D,D] (scale matrix of IW)
    nu0: float               # > D-1

@dataclass
class NIWPosterior:
    mu: torch.Tensor         # [K,D]
    kappa: torch.Tensor      # [K]
    Psi: torch.Tensor        # [K,D,D]
    nu: torch.Tensor         # [K]

@dataclass
class DirPosterior:
    # Row-wise parameters for q(Φ_k) = Dir(φ_k), φ_k in R^K
    phi: torch.Tensor        # [K,K]

@dataclass
class StickyHDPHMMParams:
    alpha: float = 4.0
    kappa: float = 4.0
    gamma: float = 1.0           # Beta(1, gamma) on stick v_k
    K: int = 16
    D: int = 64
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

# --------- main class ---------------------------------------------------------

class StickyHDPHMMVI(nn.Module):
    """
    Sticky HDP-HMM with Gaussian emissions, NIW prior, truncated to K.
    Supports encoder covariance of the form diag(sigma^2) + F F^T (optional).
    """

    def __init__(self, p: StickyHDPHMMParams, niw_prior: NIWPrior, rho_emission: float = 0.05, rho_transition: Optional[float] = None):
        """
        Initialize Sticky HDP-HMM with given parameters.
        """
        super().__init__()
        self.p = p

        D, K = p.D, p.K
        dev, dt = p.device, p.dtype

        # --- variational params ---
        # Emission NIW posteriors
        self.niw = NIWPosterior(
            mu=torch.zeros(K, D, device=dev, dtype=dt),
            kappa=torch.full((K,), niw_prior.kappa0, device=dev, dtype=dt),
            Psi=torch.stack([niw_prior.Psi0.clone().to(dev=dev, dtype=dt) for _ in range(K)], dim=0),
            nu=torch.full((K,), niw_prior.nu0, device=dev, dtype=dt),
        )

        # Transition posteriors q(Φ_k)=Dir(φ_k)
        self.dir = DirPosterior(
            phi=torch.full((K, K), fill_value=p.alpha / K, device=dev, dtype=dt)
        )

        # Global sticks (point estimate): β unconstrained params u -> σ(u)
        self.u_beta = nn.Parameter(torch.zeros(K, device=dev, dtype=dt))

        # Cache: precision expectations and Elog|Λ|
        self._cache_fresh = False
        self._E_Lambda = None      # [K,D,D]
        self._E_logdet_Lambda = None  # [K]

        # Save NIW prior
        self.register_buffer("mu0", niw_prior.mu0.to(dev=dev, dtype=dt))
        self.kappa0 = float(niw_prior.kappa0)
        self.register_buffer("Psi0", niw_prior.Psi0.to(dev=dev, dtype=dt))
        self.nu0 = float(niw_prior.nu0)
        
        # ---- streaming buffers (allocated lazily) ----
        self.stream_rho = rho_emission             # default EMA step for emissions
        self.stream_rho_trans = rho_transition     # default None -> use stream_rho for transitions too

    # ---- expectations for emissions -----------------------------------------
    def _refresh_emission_cache(self):
        """Compute E[Λ_k] and E[log|Λ_k|] under NIW posteriors."""
        K, D = self.p.K, self.p.D
        Psi = self.niw.Psi                        # [K,D,D]
        nu = self.niw.nu.view(K, 1, 1)            # [K,1,1]

        # E[Λ] = nu * Psi^{-1}
        Psi_inv, logdet_Psi = chol_inv_logdet(Psi)
        E_Lambda = nu * Psi_inv                   # [K,D,D]

        # E[log |Λ|] = sum_i ψ((ν+1-i)/2) + D log 2 - log|Ψ|
        i = torch.arange(1, D + 1, device=Psi.device, dtype=Psi.dtype).view(1, D)
        E_logdet = torch.sum(torch.special.digamma((self.niw.nu.view(K, 1) + 1 - i) / 2.0), dim=1)
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

    # ---- expected emission log-likelihood B_tk -------------------------------
    def expected_emission_loglik(
        self,
        mu_t: torch.Tensor,                 # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,  # [T,D] or [B,T,D]
        F_t: Optional[torch.Tensor] = None,         # [T,D,R] or [B,T,D,R]
        mask: Optional[torch.Tensor] = None         # [T] or [B,T] (0=ignore)
    ) -> torch.Tensor:
        """
        Returns log B: [T,K] (or [B,T,K]) with E[log p(z_t|h_t=k)].
        Uses: Tr(E[Λ] Σ_q) + (m_t-μ_k)^T E[Λ] (m_t-μ_k) and E[log|Λ|].
        """
        K, D = self.p.K, self.p.D
        E_Lam = self._get_E_Lambda()               # [K,D,D]
        E_logdet = self._get_E_logdet_Lambda()     # [K]
        dev = mu_t.device

        # Bring to [B,T,...]
        if mu_t.dim() == 2:
            mu_t = mu_t.unsqueeze(0)
            if diag_var_t is not None: diag_var_t = diag_var_t.unsqueeze(0)
            if F_t is not None: F_t = F_t.unsqueeze(0)
            if mask is not None: mask = mask.unsqueeze(0)

        B, T, D_ = mu_t.shape
        assert D_ == D

        # Moments of q(z_t): E[z] and Σ_q
        m = mu_t                                     # [B,T,D]
        diagv = (torch.zeros_like(m) if diag_var_t is None else diag_var_t).clamp_min(0.0)  # [B,T,D]

        # diff_{b,t,k} = m_{b,t} - mu_k
        diff = m.unsqueeze(2) - self.niw.mu.unsqueeze(0).unsqueeze(0)     # [B,T,K,D]

        # Quadratic mean term: (m-μ)^T E[Λ] (m-μ)
        # shape juggling: EΛ [K,D,D], diff [B,T,K,D]
        quad_mean = torch.einsum('btkd,kde,btke->btk', diff, E_Lam, diff)  # [B,T,K]

        # Trace term for diagonal covariance: Tr(EΛ diag(v))
        # = sum_i EΛ_{ii} * v_i
        ELam_diag = torch.diagonal(E_Lam, dim1=-2, dim2=-1)               # [K,D]
        trace_diag = torch.einsum('btd,kd->btk', diagv, ELam_diag)        # [B,T,K]

        # Low-rank part: Tr(EΛ F F^T) = sum_r f_r^T EΛ f_r
        if F_t is not None:
            # F: [B,T,D,R]; compute (EΛ @ F) -> [K,D,R] then f·(...) per (b,t)
            # We have per (b,t) a matrix F_{bt} [D,R]
            R = F_t.size(-1)
            trace_lr = []
            ELam = E_Lam  # [K,D,D]
            for r in range(R):
                f_r = F_t[..., r]                           # [B,T,D]
                # f_r^T EΛ f_r -> [B,T,K]
                term = torch.einsum('btd,kde,bte->btk', f_r, ELam, f_r)
                trace_lr.append(term)
            trace_lr = torch.stack(trace_lr, dim=0).sum(0)  # [B,T,K]
        else:
            trace_lr = 0.0

        quad = quad_mean + trace_diag + trace_lr            # [B,T,K]

        const = (-0.5 * self.p.D) * torch.log(torch.tensor(2.0 * torch.pi, device=dev, dtype=mu_t.dtype))
        logB = 0.5 * self._get_E_logdet_Lambda().view(1, 1, K) + const - 0.5 * quad  # [B,T,K]

        if mask is not None:
            msk = mask.to(dtype=mu_t.dtype).view(B, T, 1)   # 1=keep, 0=ignore
            # Put -inf where masked out, so FB ignores them
            logB = torch.where(msk > 0, logB, torch.full_like(logB, float('-inf')))

        return logB.squeeze(0) if logB.size(0) == 1 else logB  # [T,K] or [B,T,K]

    # ---- forward-backward ----------------------------------------------------
    def forward_backward(
        self,
        log_pi: torch.Tensor,         # [K]
        ElogA: torch.Tensor,          # [K,K] (row-wise E[log Φ_kj])
        logB: torch.Tensor,           # [T,K]
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns:
          rhat: [T,K]     (state marginals)
          xihat: [T-1,K,K] (pairwise transitions)
          ll: scalar log-likelihood
        """
        T, K = logB.shape
        # forward
        alpha = torch.empty(T, K, device=logB.device, dtype=logB.dtype)
        logZ = torch.zeros(T, device=logB.device, dtype=logB.dtype)

        a0 = log_pi + logB[0]                         # [K]
        logZ[0] = torch.logsumexp(a0, dim=-1)
        alpha[0] = a0 - logZ[0]

        for t in range(1, T):
            # log p(h_t=j | y_1..t-1) = logsum_i alpha_{t-1,i} + ElogA_{i,j}
            m = alpha[t-1].unsqueeze(1) + ElogA       # [K,K]
            pred = torch.logsumexp(m, dim=0)          # [K]
            a = pred + logB[t]                        # [K]
            logZ[t] = torch.logsumexp(a, dim=-1)
            alpha[t] = a - logZ[t]

        ll = logZ.sum().item()

        # backward
        beta = torch.zeros_like(alpha)
        for t in reversed(range(T-1)):
            # beta_t(i) = logsum_j ElogA_{i,j} + logB_{t+1,j} + beta_{t+1}(j)
            b = ElogA + logB[t+1].unsqueeze(0) + beta[t+1].unsqueeze(0)  # [K,K]
            beta[t] = torch.logsumexp(b, dim=1) - logZ[t+1]

        # posteriors
        rhat_log = alpha + beta
        rhat_log = rhat_log - torch.logsumexp(rhat_log, dim=1, keepdim=True)
        rhat = torch.exp(rhat_log)

        # pairwise posteriors
        xihat = torch.empty(T-1, K, K, device=logB.device, dtype=logB.dtype)
        for t in range(T-1):
            x = (alpha[t].unsqueeze(1) + ElogA + logB[t+1].unsqueeze(0) + beta[t+1].unsqueeze(0))  # [K,K]
            x = x - torch.logsumexp(x.view(-1), dim=0)
            xihat[t] = torch.exp(x)

        return rhat, xihat, ll

    # ---- M-step: NIW & transition -------------------------------------------
    def _moments_from_encoder(
        self, mu_t: torch.Tensor, rhat: torch.Tensor,
        diag_var_t: Optional[torch.Tensor] = None,
        F_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          Nk: [K], M1: [K,D], M2: [K,D,D]
        """
        if mu_t.dim() == 3:
            # [B,T,D]
            mu_t = mu_t.reshape(-1, mu_t.size(-1))
            if diag_var_t is not None:
                diag_var_t = diag_var_t.reshape(-1, diag_var_t.size(-1))
            if F_t is not None:
                F_t = F_t.reshape(-1, F_t.size(-2), F_t.size(-1))  # [BT,D,R]

        T, D = mu_t.shape
        K = self.p.K
        rhat = rhat.reshape(-1, K)             # [T,K]

        Nk = rhat.sum(dim=0)                   # [K]
        M1 = torch.einsum('tk,td->kd', rhat, mu_t)  # [K,D]

        # E[z z^T] for each t: diag(var) + μμ^T + F F^T
        if diag_var_t is None:
            diag_var_t = torch.zeros_like(mu_t)

        # accumulate M2 = sum_t r_tk E[zz^T]
        M2 = torch.zeros(K, D, D, device=mu_t.device, dtype=mu_t.dtype)

        # diagonal contribution
        for k in range(K):
            # sum_t r_tk * diag(var_t)
            w_diag = (rhat[:, k].unsqueeze(1) * diag_var_t).sum(dim=0)  # [D]
            M2[k] += torch.diag(w_diag)

        # mean outer products
        # sum_t r_tk * μ_t μ_t^T
        M2 += torch.einsum('tk,td,te->kde', rhat, mu_t, mu_t)

        # low-rank contribution
        if F_t is not None:
            # For each t, E[FF^T] = F F^T
            # sum_t r_tk * F_t F_t^T
            R = F_t.size(-1)
            for r in range(R):
                f = F_t[..., r]  # [T,D]
                M2 += torch.einsum('tk,td,te->kde', rhat, f, f)

        return Nk, M1, M2

    def _update_NIW(self, Nk, M1, M2):
        """Closed-form NIW updates from soft moments."""
        K, D = self.p.K, self.p.D
        mu0 = self.mu0
        k0 = self.kappa0
        Psi0 = self.Psi0
        nu0 = self.nu0

        k_hat = k0 + Nk                            # [K]
        mu_hat = (k0 * mu0.unsqueeze(0) + M1) / k_hat.unsqueeze(1)  # [K,D]
        nu_hat = nu0 + Nk                           # [K]
        # Ψ̂ = Ψ0 + M2 + κ0 μ0 μ0^T − κ̂ μ̂ μ̂^T
        Psi_hat = Psi0.unsqueeze(0).expand(K, D, D).clone()
        Psi_hat = Psi_hat + M2 + k0 * torch.einsum('d,e->de', mu0, mu0).unsqueeze(0) - \
                  torch.einsum('k,kd,ke->kde', k_hat, mu_hat, mu_hat)

        # Ensure SPD numerically (tiny jitter)
        eps = 1e-6
        I = torch.eye(D, device=Psi_hat.device, dtype=Psi_hat.dtype)
        Psi_hat = Psi_hat + eps * I.unsqueeze(0)

        self.niw.mu = mu_hat
        self.niw.kappa = k_hat
        self.niw.Psi = Psi_hat
        self.niw.nu = nu_hat
        self._cache_fresh = False

    def _ElogA(self) -> torch.Tensor:
        """E[log Φ] row-wise from Dirichlet φ."""
        phi = self.dir.phi
        return torch.special.digamma(phi) - torch.special.digamma(phi.sum(dim=1, keepdim=True))
    
    def _Epi(self) -> torch.Tensor:
        """
        Return π* (point estimate) implied by the current stick β via stick-breaking.
        Shape: [K]
        """
        with torch.no_grad():
            beta = torch.sigmoid(self.u_beta)               # [K]
            pi, _ = stick_breaking(beta)                    # [K]
        return pi

    def _update_transitions(self, xihat: torch.Tensor, pi_star: torch.Tensor):
        """Dirichlet rows update with stickiness."""
        K = self.p.K
        alpha, kappa = self.p.alpha, self.p.kappa
        # counts
        counts = xihat.sum(dim=0)                      # [K,K]
        prior_rows = alpha * pi_star.view(1, K) + kappa * torch.eye(K, device=counts.device, dtype=counts.dtype)
        self.dir.phi = prior_rows + counts

    # ---- π-step: optimize beta (point estimate) ------------------------------
    def _optimize_pi(self, rhat: torch.Tensor) -> torch.Tensor:
        """
        Maximize the β-restricted ELBO:
            L(β) = E_q[log p(Φ | π)] + E_q[log p(h1 | π)] + log p(β)
        with π = SB(β). Uses autograd on u_beta.
        Returns π* (detached).
        """
        K = self.p.K
        alpha, kappa, gamma = self.p.alpha, self.p.kappa, self.p.gamma
        phi = self.dir.phi.detach()
        ElogA = torch.special.digamma(phi) - torch.special.digamma(phi.sum(1, keepdim=True))  # [K,K]
        r1 = rhat[0]  # [K]

        opt = torch.optim.Adam([self.u_beta], lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            beta = torch.sigmoid(self.u_beta)              # [K]
            pi, _ = stick_breaking(beta)                   # [K]
            # prior rows a_k = α π + κ δ_k
            a = alpha * pi.unsqueeze(0) + kappa * torch.eye(K, device=pi.device, dtype=pi.dtype)  # [K,K]

            # E[log p(Φ | π)] under q(Φ): Dirichlet expectation
            # = sum_k [ sum_j (a_kj - 1) ElogA_kj - (logB(a_k)) ]  ; logB(a)=sum_j lgamma(a_j)-lgamma(sum_j a_j)
            term1 = (a - 1.0) * ElogA
            lnorm = torch.lgamma(a).sum(dim=1) - torch.lgamma(a.sum(dim=1))
            L1 = term1.sum() - lnorm.sum()

            # E[log p(h1 | π)] = sum_k r1_k log π_k
            L2 = torch.sum(r1 * torch.log(torch.clamp(pi, min=1e-30)))

            # log p(β) = sum_k (γ-1) log(1-β_k)  (since Beta(1,γ))
            L3 = (gamma - 1.0) * torch.sum(torch.log(torch.clamp(1.0 - beta, min=1e-30)))

            L = -(L1 + L2 + L3)  # minimize negative
            L.backward()
            opt.step()

        with torch.no_grad():
            beta = torch.sigmoid(self.u_beta)
            pi, _ = stick_breaking(beta)
        return pi.detach()
    
    # ---- streaming helpers ---------------------------------------------------
    def _alloc_stream_buffers(self):
        """Allocate running (EMA) sufficient statistics if not present."""
        if hasattr(self, "S_Nk"):
            return
        K, D = self.p.K, self.p.D
        dev, dt = self.mu0.device, self.mu0.dtype
        self.register_buffer("S_Nk",      torch.zeros(K,        device=dev, dtype=dt))
        self.register_buffer("S_M1",      torch.zeros(K, D,     device=dev, dtype=dt))
        self.register_buffer("S_M2",      torch.zeros(K, D, D,  device=dev, dtype=dt))
        self.register_buffer("S_counts",  torch.zeros(K, K,     device=dev, dtype=dt))
        self.register_buffer("S_steps",   torch.tensor(0.0,     device=dev, dtype=dt))

    def enable_streaming(self, rho: float = 0.05, rho_trans: float | None = None):
        """
        Turn on streaming/EMA updates.
        rho in (0,1]: step for emission sufficient stats;
        rho_trans: step for transition counts (defaults to rho if None).
        """
        self._alloc_stream_buffers()
        self.stream_rho = float(max(0.0, min(1.0, rho)))
        self.stream_rho_trans = float(max(0.0, min(1.0, rho_trans))) if rho_trans is not None else None

    @torch.no_grad()
    def streaming_reset(self):
        """Zero the running sufficient statistics."""
        self._alloc_stream_buffers()
        self.S_Nk.zero_(); self.S_M1.zero_(); self.S_M2.zero_(); self.S_counts.zero_()
        self.S_steps.fill_(0.0)

    def get_stream_state(self) -> dict:
        """Optionally persist streaming stats along with posterior params."""
        if not hasattr(self, "S_Nk"):
            return {}
        return {
            "S_Nk": self.S_Nk.detach().clone(),
            "S_M1": self.S_M1.detach().clone(),
            "S_M2": self.S_M2.detach().clone(),
            "S_counts": self.S_counts.detach().clone(),
            "S_steps": self.S_steps.detach().clone(),
            "stream_rho": float(self.stream_rho),
            "stream_rho_trans": float(self.stream_rho_trans) if self.stream_rho_trans is not None else None,
        }

    def load_stream_state(self, state: dict):
        """Restore streaming stats (call after __init__)."""
        if not state:
            return
        self._alloc_stream_buffers()
        dev, dt = self.mu0.device, self.mu0.dtype
        for k in ["S_Nk", "S_M1", "S_M2", "S_counts", "S_steps"]:
            if k in state and state[k] is not None:
                setattr(self, k, state[k].to(device=dev, dtype=dt))
        if "stream_rho" in state and state["stream_rho"] is not None:
            self.stream_rho = float(state["stream_rho"])
        if "stream_rho_trans" in state:
            self.stream_rho_trans = float(state["stream_rho_trans"]) if state["stream_rho_trans"] is not None else None

    # ---- public update API ---------------------------------------------------
    def update_from_encoder(
        self,
        mu_t: torch.Tensor,                     # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,  # same shape as mu_t
        F_t: Optional[torch.Tensor] = None,         # [T,D,R] or [B,T,D,R]
        mask: Optional[torch.Tensor] = None,        # [T] or [B,T] (1=valid)
        n_e_steps: int = 1,
        optimize_pi: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        One outer VI step: (E-step -> M-step -> (optional) π-step).
        Returns a dict with rhat, xihat and current ELBO-ish pieces.
        """
        # (a) build emission log-likelihoods
        logB = self.expected_emission_loglik(mu_t, diag_var_t, F_t, mask)  # [T,K] or [B,T,K]

        if logB.dim() == 3:
            # process sequences independently then pool moments
            outs = []
            for b in range(logB.size(0)):
                outs.append(self._update_single_seq(mu_t[b], diag_var_t[b] if diag_var_t is not None else None,
                                                   F_t[b] if F_t is not None else None,
                                                   mask[b] if mask is not None else None,
                                                   logB[b], n_e_steps, optimize_pi))
            # merge rhat for reference; NIW/Dir are global already
            return outs[-1]  # return last (parameters already updated globally)
        else:
            return self._update_single_seq(mu_t, diag_var_t, F_t, mask, logB, n_e_steps, optimize_pi)

    def _update_single_seq(
        self, mu_t, diag_var_t, F_t, mask, logB, n_e_steps, optimize_pi
    ):
        K = self.p.K
        # init π from current β
        with torch.no_grad():
            pi, _ = stick_breaking(torch.sigmoid(self.u_beta))
        log_pi = torch.log(torch.clamp(pi, min=1e-30))

        for _ in range(n_e_steps):
            # E-step
            ElogA = self._ElogA()                              # [K,K]
            rhat, xihat, ll = self.forward_backward(log_pi, ElogA, logB)

            # M-step: emissions
            Nk, M1, M2 = self._moments_from_encoder(mu_t, rhat, diag_var_t, F_t)
            self._update_NIW(Nk, M1, M2)

            # π-step (optional)
            if optimize_pi:
                pi = self._optimize_pi(rhat)
                log_pi = torch.log(torch.clamp(pi, min=1e-30))

            # M-step: transitions (uses π* through its prior rows)
            self._update_transitions(xihat, pi)

            # recompute logB because NIW changed
            logB = self.expected_emission_loglik(mu_t, diag_var_t, F_t, mask)

        return {
            "rhat": rhat.detach(),            # [T,K]
            "xihat": xihat.detach(),          # [T-1,K,K]
            "loglik": torch.tensor(ll),
            "pi_star": pi.detach(),           # [K]
            "ElogA": self._ElogA().detach(),  # [K,K]
        }
        
    @torch.no_grad()
    def update_streaming(
        self,
        mu_t: torch.Tensor,                     # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,  # same shape as mu_t
        F_t: Optional[torch.Tensor] = None,         # [T,D,R] or [B,T,D,R]
        mask: Optional[torch.Tensor] = None,        # [T] or [B,T]
        rho: Optional[float] = None,                # override step; defaults to self.stream_rho
        optimize_pi: bool = False,                  # rarely needed online
    ) -> Dict[str, torch.Tensor]:
        """
        Streaming (EMA) update using the current encoder outputs.
        Steps:
          1) E-step per sequence -> rhat, xihat.
          2) Aggregate moments {Nk, M1, M2} and transition counts over the batch.
          3) EMA-update running sufficient stats (S_*).
          4) Refresh NIW/Dir posteriors from EMA stats.
        Returns a small dict of diagnostics.
        """
        self._alloc_stream_buffers()
        K, D = self.p.K, self.p.D

        # choose step sizes
        rho_e = float(self.stream_rho if rho is None else rho)
        rho_a = float(self.stream_rho_trans if self.stream_rho_trans is not None else rho_e)
        rho_e = max(0.0, min(1.0, rho_e))
        rho_a = max(0.0, min(1.0, rho_a))

        # prepare shapes and iterate sequences if needed
        if mu_t.dim() == 2:
            mu_bt = mu_t.unsqueeze(0)
            dv_bt = diag_var_t.unsqueeze(0) if diag_var_t is not None else None
            F_bt  = F_t.unsqueeze(0)        if F_t is not None else None
            m_bt  = mask.unsqueeze(0)       if mask is not None else None
        else:
            mu_bt, dv_bt, F_bt, m_bt = mu_t, diag_var_t, F_t, mask

        B = mu_bt.size(0)
        pi_star = self._Epi()                                # [K]
        log_pi  = torch.log(torch.clamp(pi_star, min=1e-30))
        ElogA   = self._ElogA()                              # [K,K]

        # aggregate moments and counts over the mini-batch
        Nk_acc     = torch.zeros(K,       device=mu_bt.device, dtype=mu_bt.dtype)
        M1_acc     = torch.zeros(K, D,    device=mu_bt.device, dtype=mu_bt.dtype)
        M2_acc     = torch.zeros(K, D, D, device=mu_bt.device, dtype=mu_bt.dtype)
        counts_acc = torch.zeros(K, K,    device=mu_bt.device, dtype=mu_bt.dtype)
        ll_list = []

        for b in range(B):
            logB_b = self.expected_emission_loglik(
                mu_bt[b],
                dv_bt[b] if dv_bt is not None else None,
                F_bt[b]  if F_bt  is not None else None,
                m_bt[b]  if m_bt  is not None else None
            )  # [T,K]
            rhat_b, xihat_b, ll_b = self.forward_backward(log_pi, ElogA, logB_b)
            Nk_b, M1_b, M2_b = self._moments_from_encoder(
                mu_bt[b], rhat_b,
                dv_bt[b] if dv_bt is not None else None,
                F_bt[b]  if F_bt  is not None else None
            )
            Nk_acc += Nk_b; M1_acc += M1_b; M2_acc += M2_b
            counts_acc += xihat_b.sum(dim=0)  # [K,K]
            ll_list.append(ll_b)

            if optimize_pi and b == 0:
                # Optional: refine pi using first sequence’s rhat (cheap surrogate)
                pi_star = self._optimize_pi(rhat_b)
                log_pi  = torch.log(torch.clamp(pi_star, min=1e-30))

        # ---- EMA update of sufficient stats ----
        # emissions
        self.S_Nk.mul_(1.0 - rho_e).add_(rho_e * Nk_acc)
        self.S_M1.mul_(1.0 - rho_e).add_(rho_e * M1_acc)
        self.S_M2.mul_(1.0 - rho_e).add_(rho_e * M2_acc)
        # transitions
        self.S_counts.mul_(1.0 - rho_a).add_(rho_a * counts_acc)
        self.S_steps.add_(1.0)

        # ---- refresh posteriors from running stats ----
        # NIW from (S_Nk, S_M1, S_M2)
        self._update_NIW(self.S_Nk, self.S_M1, self.S_M2)
        # Dirichlet rows from EMA counts with sticky prior and current π*
        K = self.p.K
        alpha, kappa = self.p.alpha, self.p.kappa
        I = torch.eye(K, device=self.S_counts.device, dtype=self.S_counts.dtype)
        prior_rows = alpha * pi_star.view(1, K) + kappa * I
        self.dir.phi = prior_rows + self.S_counts

        # Diagnostics
        out = {
            "rho_emission": rho_e,
            "rho_transition": rho_a,
            "Nk_batch_sum": Nk_acc.sum().item(),
            "ll_mean": float(np.mean(ll_list)) if len(ll_list) > 0 else float("nan"),
            "pi_star": pi_star.detach(),
            "ElogA": self._ElogA().detach(),
        }
        return out

    # ---- accessors -----------------------------------------------------------
    def get_posterior_params(self) -> Dict[str, torch.Tensor]:
        return {
            "mu": self.niw.mu.detach(),             # [K,D]
            "kappa": self.niw.kappa.detach(),       # [K]
            "Psi": self.niw.Psi.detach(),           # [K,D,D]
            "nu": self.niw.nu.detach(),             # [K]
            "phi": self.dir.phi.detach(),           # [K,K]
            "beta_u": self.u_beta.detach(),         # [K]
        }
        
    def get_emission_expectations(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (μ_k, E[Λ_k], E[log|Λ_k|]) under the current NIW posteriors.
        Shapes: μ_k:[K,D], E[Λ_k]:[K,D,D], E[log|Λ_k|]:[K]
        """
        return self.niw.mu, self._get_E_Lambda(), self._get_E_logdet_Lambda()
    
    def get_streaming_rho_emission(self) -> float:
        """Return the current emission streaming step."""
        return float(self.stream_rho)
    
    def get_streaming_rho_transition(self) -> Optional[float]:
        """Return the current transition streaming step (or None)."""
        return float(self.stream_rho_trans) if self.stream_rho_trans is not None else None

    def load_posterior_params(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Inverse of get_posterior_params(): load NIW/Dir/β parameters from disk.
        Tensors are moved to module device/dtype; refresh emission cache.
        """
        dev, dt = self.mu0.device, self.mu0.dtype
        self.niw.mu    = params["mu"].to(dev=dev, dtype=dt)
        self.niw.kappa = params["kappa"].to(dev=dev, dtype=dt)
        self.niw.Psi   = params["Psi"].to(dev=dev, dtype=dt)
        self.niw.nu    = params["nu"].to(dev=dev, dtype=dt)
        if "phi" in params:     self.dir.phi.copy_(params["phi"].to(dev=dev, dtype=dt))
        if "beta_u" in params:  self.u_beta.data.copy_(params["beta_u"].to(dev=dev, dtype=dt))
        self._cache_fresh = False

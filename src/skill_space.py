# ===== skill_space_refactored.py ============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ---------- main class -------------------------------------------------------

class StickyHDPHMMVI(nn.Module):
    """
    Sticky HDP-HMM with Gaussian emissions (NIW prior), truncated to K.
    Supports encoder covariance Σ_t = diag(σ_t^2) + F_t F_t^T (optional).

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
        dev, dt = p.device, p.dtype

        # Emission NIW posteriors
        self.niw = NIWPosterior(
            mu=torch.zeros(K, D, device=dev, dtype=dt),
            kappa=torch.full((K,), niw_prior.kappa0, device=dev, dtype=dt),
            Psi=torch.stack([niw_prior.Psi0.clone().to(device=dev, dtype=dt) for _ in range(K)], dim=0),
            nu=torch.full((K,), niw_prior.nu0, device=dev, dtype=dt),
        )

        # Transition posteriors q(Φ_k)=Dir(φ_k)
        self.dir = DirPosterior(
            phi=torch.full((K, K), fill_value=p.alpha / K, device=dev, dtype=dt)
        )

        # Global sticks β (point estimate via u -> σ(u))
        self.u_beta = nn.Parameter(torch.zeros(K, device=dev, dtype=dt))

        # Cache for E[Λ], E[log|Λ|]
        self._cache_fresh = False
        self._E_Lambda = None      # [K,D,D]
        self._E_logdet_Lambda = None  # [K]

        # Save NIW prior
        self.register_buffer("mu0", niw_prior.mu0.to(device=dev, dtype=dt))
        self.kappa0 = float(niw_prior.kappa0)
        self.register_buffer("Psi0", niw_prior.Psi0.to(device=dev, dtype=dt))
        self.nu0 = float(niw_prior.nu0)

        # Streaming defaults + lazy buffers
        self.stream_rho = float(rho_emission)
        self.stream_rho_trans = float(rho_transition) if rho_transition is not None else None
        self._stream_allocated = False

    # ---- cache E[Λ], E[log|Λ|] -----------------------------------------------
    def _refresh_emission_cache(self):
        K, D = self.p.K, self.p.D
        Psi = self.niw.Psi
        nu = self.niw.nu.view(K, 1, 1)
        Psi_inv, logdet_Psi = chol_inv_logdet(Psi)
        E_Lambda = nu * Psi_inv  # [K,D,D]

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

    # ---- expected emission log-likelihood logB_tk ----------------------------
    @torch.no_grad()
    def expected_emission_loglik(
        self,
        mu_t: torch.Tensor,                            # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,     # [T,D] or [B,T,D]
        F_t: Optional[torch.Tensor] = None,            # [T,D,R] or [B,T,D,R]
        mask: Optional[torch.Tensor] = None            # [T] or [B,T] (1=valid, 0=ignore)
    ) -> torch.Tensor:
        """
        Returns logB: [T,K] (or [B,T,K]) with E_q[log p(z_t | h_t=k)].
        E_q[...] = 0.5*(Elog|Λ_k| - D log(2π)) - 0.5*( Tr(EΛ Σ_t) + (m_t-μ_k)^T EΛ (m_t-μ_k) )
        Mask: invalid frames act as 'no observation' => logB += 0 at those steps.
        """
        K, D = self.p.K, self.p.D
        E_Lam = self._get_E_Lambda()           # [K,D,D]
        E_logdet = self._get_E_logdet_Lambda() # [K]

        # shape to [B,T,...]
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
        const = 0.5 * (E_logdet - D * torch.log(torch.tensor(2.0 * torch.pi, device=mu_t.device, dtype=mu_t.dtype)))  # [K]

        # (m_t - μ_k)
        diff = mu_t.unsqueeze(2) - self.niw.mu.view(1, 1, K, D)    # [B,T,K,D]

        # quadratic term: (m-μ)^T EΛ (m-μ)
        quad_mean = torch.einsum('btkd,kde,btke->btk', diff, E_Lam, diff)  # [B,T,K]

        # trace term: Tr(EΛ Σ_t) = sum_d EΛ[k,d,d]*σ_t[d] + sum_r f_{t,r}^T EΛ f_{t,r}
        E_Lam_diag = torch.diagonal(E_Lam, dim1=1, dim2=2)  # [K,D]
        tr_diag = torch.einsum('kd,btd->btk', E_Lam_diag, diag_var_t)      # [B,T,K]

        if F_t is not None:
            # sum_r f^T EΛ f
            # einsum -> [B,T,R,K] then sum over R
            tr_lr = torch.einsum('btdr,kde,bter->btrk', F_t, E_Lam, F_t).sum(dim=2)  # [B,T,K]
        else:
            tr_lr = 0.0

        logB = const.view(1, 1, K) - 0.5 * (quad_mean + tr_diag + tr_lr)  # [B,T,K]

        if mask is not None:
            # no-observation: add 0 contribution (i.e., keep alpha/beta recursion well-defined)
            m = mask.view(B, T, 1)
            logB = torch.where(m.bool(), logB, torch.zeros_like(logB))

        return logB if mu_t.dim() == 3 else logB.squeeze(0)

    # ---- forward-backward in log-space --------------------------------------
    @torch.no_grad()
    def forward_backward(
        self,
        log_pi: torch.Tensor,   # [K]
        ElogA: torch.Tensor,    # [K,K]
        logB: torch.Tensor      # [T,K]
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Standard HMM FB in log-domain with 'no observation' allowed (logB[t,:]==0).
        Returns rhat [T,K], xihat [T-1,K,K], loglik (scalar).
        """
        T, K = logB.shape
        # α
        log_alpha = torch.full((T, K), -float('inf'), device=logB.device, dtype=logB.dtype)
        log_alpha[0] = log_pi + logB[0]
        for t in range(1, T):
            # logsumexp_i (log_alpha[t-1,i] + ElogA[i,k])
            prev = log_alpha[t-1].unsqueeze(1) + ElogA  # [K,K]
            log_alpha[t] = logB[t] + torch.logsumexp(prev, dim=0)

        ll = torch.logsumexp(log_alpha[-1], dim=0)  # scalar

        # β
        log_beta = torch.zeros(T, K, device=logB.device, dtype=logB.dtype)
        for t in range(T - 2, -1, -1):
            # logsumexp_k (ElogA[i,k] + logB[t+1,k] + log_beta[t+1,k])
            tmp = ElogA + (logB[t + 1] + log_beta[t + 1]).unsqueeze(0)  # [K,K]
            log_beta[t] = torch.logsumexp(tmp, dim=1)

        # posteriors
        log_gamma = log_alpha + log_beta - ll
        rhat = torch.softmax(log_gamma, dim=1)  # [T,K]

        xihat = torch.zeros(T - 1, K, K, device=logB.device, dtype=logB.dtype)
        for t in range(T - 1):
            tmp = (
                log_alpha[t].unsqueeze(1) + ElogA
                + logB[t + 1].unsqueeze(0) + log_beta[t + 1].unsqueeze(0)
                - ll
            )  # [K,K]
            xihat[t] = torch.softmax(tmp.view(-1), dim=0).view(K, K)

        return rhat, xihat, float(ll.item())

    # ---- M-step: sufficient stats from encoder -------------------------------
    @torch.no_grad()
    def _moments_from_encoder(
        self,
        mu_t: torch.Tensor, rhat: torch.Tensor,
        diag_var_t: Optional[torch.Tensor] = None,
        F_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (Nk [K], M1 [K,D], M2 [K,D,D]) where
        M2 accumulates E[zz^T] = diag(var_t) + μ_t μ_t^T + Σ_r f_{t,r} f_{t,r}^T.
        """
        if mu_t.dim() == 3:
            mu_t = mu_t.reshape(-1, mu_t.size(-1))
            if diag_var_t is not None:
                diag_var_t = diag_var_t.reshape(-1, diag_var_t.size(-1))
            if F_t is not None:
                F_t = F_t.reshape(-1, F_t.size(-2), F_t.size(-1))  # [BT,D,R]

        T, D = mu_t.shape
        K = self.p.K
        rhat = rhat.reshape(-1, K)  # [T,K]

        Nk = rhat.sum(dim=0)                            # [K]
        M1 = torch.einsum('tk,td->kd', rhat, mu_t)      # [K,D]

        if diag_var_t is None:
            diag_var_t = torch.zeros_like(mu_t)

        M2 = torch.zeros(K, D, D, device=mu_t.device, dtype=mu_t.dtype)

        # diagonal part
        for k in range(K):
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
    def _update_NIW(self, Nk, M1, M2):
        """Closed-form NIW updates from soft moments."""
        K, D = self.p.K, self.p.D
        mu0, k0, Psi0, nu0 = self.mu0, self.kappa0, self.Psi0, self.nu0

        k_hat = k0 + Nk
        mu_hat = (k0 * mu0.unsqueeze(0) + M1) / k_hat.unsqueeze(1)
        nu_hat = nu0 + Nk

        Psi_hat = Psi0.unsqueeze(0).expand(K, D, D).clone()
        Psi_hat = Psi_hat + M2 + k0 * torch.einsum('d,e->de', mu0, mu0).unsqueeze(0) \
                  - torch.einsum('k,kd,ke->kde', k_hat, mu_hat, mu_hat)

        # small jitter to keep SPD
        eps = 1e-6
        I = torch.eye(D, device=Psi_hat.device, dtype=Psi_hat.dtype)
        Psi_hat = Psi_hat + eps * I.unsqueeze(0)

        self.niw.mu = mu_hat
        self.niw.kappa = k_hat
        self.niw.Psi = Psi_hat
        self.niw.nu = nu_hat
        self._cache_fresh = False

    # ---- transitions & π -----------------------------------------------------
    def _ElogA(self) -> torch.Tensor:
        phi = self.dir.phi
        return torch.special.digamma(phi) - torch.special.digamma(phi.sum(dim=1, keepdim=True))

    @torch.no_grad()
    def _Epi(self) -> torch.Tensor:
        beta = torch.sigmoid(self.u_beta)
        pi, _ = stick_breaking(beta)
        return pi

    @torch.no_grad()
    def _update_transitions(self, xihat: torch.Tensor, pi_star: torch.Tensor):
        K = self.p.K
        alpha, kappa = self.p.alpha, self.p.kappa
        counts = xihat.sum(dim=0)  # [K,K]
        prior_rows = alpha * pi_star.view(1, K) + kappa * torch.eye(K, device=counts.device, dtype=counts.dtype)
        self.dir.phi = prior_rows + counts

    def _optimize_pi_from_r1(self, r1: torch.Tensor, steps: int = 200, lr: float = 0.05) -> torch.Tensor:
        """
        Optimize global sticks β (via logits u_beta) using a β-restricted ELBO:
            L(β) = E_q[log p(Φ | π)] + E_q[log p(h1 | π)] + log p(β),
        with π = SB(σ(u_beta)). Uses autograd only through u_beta.
        Args:
            r1: average initial-state responsibilities (K,)
        Returns:
            π* (detached, shape [K])
        """
        K = self.p.K
        alpha, kappa, gamma = self.p.alpha, self.p.kappa, self.p.gamma
        # Treat these as constants for this step
        ElogA = self._ElogA().detach()
        r1 = r1.detach()
        opt = torch.optim.Adam([self.u_beta], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            beta = torch.sigmoid(self.u_beta)          # [K]
            pi, _ = stick_breaking(beta)               # [K]
            # Dirichlet prior rows a_k = α π + κ δ_k
            a = alpha * pi.unsqueeze(0) + kappa * torch.eye(K, device=pi.device, dtype=pi.dtype)  # [K,K]
            # E_q[log p(Φ | π)] under q(Φ) with row-wise Dirichlet φ
            L1 = ((a - 1.0) * ElogA).sum() - (torch.lgamma(a).sum(dim=1) - torch.lgamma(a.sum(dim=1))).sum()
            # E_q[log p(h1 | π)]
            L2 = torch.sum(r1 * torch.log(torch.clamp(pi, min=1e-30)))
            # log p(β) with Beta(1,γ) sticks: ∑ (γ-1) log(1-β_k)
            L3 = (gamma - 1.0) * torch.sum(torch.log(torch.clamp(1.0 - beta, min=1e-30)))
            loss = -(L1 + L2 + L3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.u_beta], max_norm=10.0)
            opt.step()
            with torch.no_grad():
                # keep logits in a sane range for stability
                self.u_beta.clamp_(-10.0, 10.0)
        with torch.no_grad():
            beta = torch.sigmoid(self.u_beta)
            pi, _ = stick_breaking(beta)
        return pi.detach()

    # ---- streaming buffers ---------------------------------------------------
    def _alloc_stream_buffers(self):
        if self._stream_allocated:
            return
        K, D = self.p.K, self.p.D
        dev, dt = self.mu0.device, self.mu0.dtype
        self.register_buffer("S_Nk",     torch.zeros(K,       device=dev, dtype=dt))
        self.register_buffer("S_M1",     torch.zeros(K, D,    device=dev, dtype=dt))
        self.register_buffer("S_M2",     torch.zeros(K, D, D, device=dev, dtype=dt))
        self.register_buffer("S_counts", torch.zeros(K, K,    device=dev, dtype=dt))
        self.register_buffer("S_steps",  torch.tensor(0.0,    device=dev, dtype=dt))
        self._stream_allocated = True

    @torch.no_grad()
    def streaming_reset(self):
        self._alloc_stream_buffers()
        self.S_Nk.zero_(); self.S_M1.zero_(); self.S_M2.zero_(); self.S_counts.zero_()
        self.S_steps.fill_(0.0)

    # ---- unified update (streaming & non-streaming) --------------------------
    @torch.no_grad()
    def update(
        self,
        mu_t: torch.Tensor,                            # [T,D] or [B,T,D]
        diag_var_t: Optional[torch.Tensor] = None,     # same shape as mu_t
        F_t: Optional[torch.Tensor] = None,            # [T,D,R] or [B,T,D,R]
        mask: Optional[torch.Tensor] = None,           # [T] or [B,T]
        n_e_steps: int = 1,
        rho: Optional[float] = 1.0,                    # 1.0 => non-streaming; 0<rho<1 => EMA
        optimize_pi: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        One outer VI step that also updates parameters either:
          - non-streaming (rho=1.0): replace sufficient stats with current batch
          - streaming (0<rho<1): EMA blend with running stats
        Returns dict with rhat/xihat/loglik from the last processed sequence and current ElogA, pi*.
        """
        self._alloc_stream_buffers()

        # Build emission log-likelihoods
        logB = self.expected_emission_loglik(mu_t, diag_var_t, F_t, mask)  # [T,K] or [B,T,K]

        # Iterate sequences if batched
        is_batched = (logB.dim() == 3)
        B = logB.size(0) if is_batched else 1

        # Accumulate sufficient stats over the batch
        K, D = self.p.K, self.p.D
        acc_counts = torch.zeros(K, K, device=self.mu0.device, dtype=self.mu0.dtype)
        acc_Nk     = torch.zeros(K,    device=self.mu0.device, dtype=self.mu0.dtype)
        acc_M1     = torch.zeros(K, D, device=self.mu0.device, dtype=self.mu0.dtype)
        acc_M2     = torch.zeros(K, D, D, device=self.mu0.device, dtype=self.mu0.dtype)

        last_out = None
        r1_sum = torch.zeros(K, device=self.mu0.device, dtype=self.mu0.dtype)

        for b in range(B):
            _logB = logB[b] if is_batched else logB          # [T,K]
            _mu   = mu_t[b] if is_batched else mu_t          # [T,D]
            _dv   = (diag_var_t[b] if (is_batched and diag_var_t is not None) else diag_var_t)
            _F    = (F_t[b] if (is_batched and F_t is not None) else F_t)
            _msk  = (mask[b] if (is_batched and mask is not None) else mask)

            # init π from current β
            with torch.no_grad():
                pi, _ = stick_breaking(torch.sigmoid(self.u_beta))
            log_pi = torch.log(torch.clamp(pi, min=1e-30))

            # inner E/M (can repeat n times)
            rhat = xihat = ll = None
            for _ in range(n_e_steps):
                ElogA = self._ElogA()
                rhat, xihat, ll = self.forward_backward(log_pi, ElogA, _logB)

                # emissions
                Nk, M1, M2 = self._moments_from_encoder(_mu, rhat, _dv, _F)
                # accumulate
                acc_Nk += Nk; acc_M1 += M1; acc_M2 += M2

                # (optional) π step uses average r1 across sequences
                r1_sum += rhat[0]
                if optimize_pi:
                    # postpone actual optimization until after the loop so we can use mean r1
                    pass

                # transitions
                self._update_transitions(xihat, pi)
                # accumulate transition counts for streaming diagnostics/EMA
                acc_counts += xihat.sum(dim=0)

                # recompute logB with updated NIW
                # (NOTE: we update NIW only after stats blending below to keep unified path)
                _logB = _logB  # keep as is during per-seq loop, NIW will update after blending

            last_out = {"rhat": rhat.detach(), "xihat": xihat.detach(), "loglik": torch.tensor(ll)}

        # Blend sufficient stats (EMA or full replace)
        if rho is None:
            rho = self.stream_rho
        rho_t = self.stream_rho_trans if self.stream_rho_trans is not None else rho
        rho = float(max(0.0, min(1.0, rho)))
        rho_t = float(max(0.0, min(1.0, rho_t)))

        self.S_Nk     = (1.0 - rho)  * self.S_Nk     + rho  * acc_Nk
        self.S_M1     = (1.0 - rho)  * self.S_M1     + rho  * acc_M1
        self.S_M2     = (1.0 - rho)  * self.S_M2     + rho  * acc_M2
        self.S_counts = (1.0 - rho_t) * self.S_counts + rho_t * acc_counts

        # Note: xihat was used inside _update_transitions per sequence to maintain φ rows.
        # If you prefer to base φ strictly on the blended counts, uncomment next line and comment per-seq update:
        # self._update_transitions(acc_xihat, self._Epi())

        # Now update emissions from blended stats
        self._update_NIW(self.S_Nk, self.S_M1, self.S_M2)

        # Optimize π using average r1 across sequences (more stable than single seq)
        if optimize_pi and B > 0:
            r1_mean = (r1_sum / B).clamp_min(1e-12)
            r1_mean = r1_mean / r1_mean.sum()
            # ensure grads are enabled even if caller wrapped update() in no_grad
            self.u_beta.requires_grad_(True)
            with torch.enable_grad():
                pi_star = self._optimize_pi_from_r1(r1_mean)
        else:
            with torch.no_grad():
                pi_star = self._Epi()

        return {
            "rhat": last_out["rhat"],
            "xihat": last_out["xihat"],
            "loglik": last_out["loglik"],
            "pi_star": pi_star.detach(),
            "ElogA": self._ElogA().detach(),
        }

    # ---- accessors & (de)serialization --------------------------------------
    def get_posterior_params(self) -> Dict[str, torch.Tensor]:
        return {
            "mu": self.niw.mu.detach(),
            "kappa": self.niw.kappa.detach(),
            "Psi": self.niw.Psi.detach(),
            "nu": self.niw.nu.detach(),
            "phi": self.dir.phi.detach(),
            "beta_u": self.u_beta.detach(),
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

    # --- compact accessors (compatibility with train_new.py) -----------------
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
        Compute quick sanity diagnostics:
          - avg log-likelihood per (valid) step
          - occupancy π_hat, effective #skills, top-5 skills
          - transition stickiness ratio (diag mass)
          - state entropy (avg over time)
          - expected self-transition probability (from softmax(ElogA))
        """
        # responsibilities with current params
        logB = self.expected_emission_loglik(mu_t, diag_var_t, F_t, mask)
        is_batched = (logB.dim() == 3)
        B = logB.size(0) if is_batched else 1

        r_all = []
        xi_all = []
        ll_total = 0.0
        steps = 0
        for b in range(B):
            _logB = logB[b] if is_batched else logB
            with torch.no_grad():
                pi, _ = stick_breaking(torch.sigmoid(self.u_beta))
                log_pi = torch.log(torch.clamp(pi, min=1e-30))
                rhat, xihat, ll = self.forward_backward(log_pi, self._ElogA(), _logB)
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

        r_cat = torch.cat(r_all, dim=0)   # [sum_T, K]
        xi_cat = torch.cat(xi_all, dim=0) if len(xi_all) > 0 else torch.zeros(0)
        Nk = r_cat.sum(dim=0) + 1e-12
        pi_hat = (Nk / Nk.sum()).cpu()

        # entropy over states
        ent = (- (r_cat.clamp_min(1e-9) * r_cat.clamp_min(1e-9).log()).sum(dim=1)).mean().item()

        # transition diagnostics from ElogA proxy
        A_bar = torch.softmax(self._ElogA(), dim=1)  # not exact E[A], but a reasonable proxy
        diag_ratio = float(torch.diagonal(A_bar, dim1=0, dim2=1).mean().item())

        # effective number of skills
        effK = float(torch.exp(-(pi_hat * (pi_hat + 1e-12).log()).sum()).item())

        topk_vals, topk_idx = torch.topk(pi_hat, k=min(5, self.p.K))
        avg_ll_per_step = float(ll_total / max(steps, 1))

        return {
            "avg_loglik_per_step": avg_ll_per_step,
            "state_entropy": ent,
            "occupancy_pi_hat": pi_hat,
            "effective_K": effK,
            "top5_pi": topk_vals,
            "top5_idx": topk_idx,
            "stickiness_diag_mean": diag_ratio,
        }

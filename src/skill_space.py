# sticky_hdp_hmm.py
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

digamma = torch.special.digamma
lgamma  = torch.special.gammaln
EPS     = 1e-9


@dataclass
class StickyHDPHMMConfig:
    K: int                      # truncation (explicit states); we also allocate K+1 remainder
    D: int                      # latent dimensionality
    alpha: float = 6.0          # DP concentration
    kappa: float = 4.0          # stickiness
    gamma: float = 1.0          # top-level GEM mass
    # NIW hyperparams
    mu0: float = 0.0
    kappa0: float = 1e-2
    nu0: int = 32               # must be > D-1; set larger than D for stability
    psi0_scale: float = 1.0     # Ψ0 = psi0_scale * I
    # Optim
    beta_lr: float = 5e-2       # lr for β (stick weights) optimiser
    device: str = "cpu"


class StickyHDPHMM(nn.Module):
    """
    Truncated sticky-HDP-HMM with NIW emissions; mean-field VI with online updates.
    Matches the derivations in your screenshots (B.1–B.4).
    """
    def __init__(self, cfg: StickyHDPHMMConfig):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K
        self.Kp1 = cfg.K + 1
        self.D = cfg.D
        self.alpha = cfg.alpha
        self.kappa = cfg.kappa
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)

        # ----- Global sticks β_k (k=1..K) as unconstrained u_k parameters -----
        # β_k = sigmoid(u_k); π* constructed via stick-breaking
        u0 = torch.full((self.K,), -math.log(self.K), dtype=torch.float32)  # small sticks initially
        self.u = nn.Parameter(u0)

        # ----- Variational rows q(Φ_k) ~ Dir(φ̂_k) for k=1..K (each length K+1) -----
        # start from prior α π* + κ δ (we’ll fill π* after init)
        self.phi_hat = nn.Parameter(torch.ones(self.K, self.Kp1), requires_grad=False)

        # ----- Variational NIW parameters for emissions (k = 1..K+1) -----
        I = torch.eye(self.D, dtype=torch.float32, device=self.device)
        Psi0 = cfg.psi0_scale * I
        # store as buffers so we can "online add" sufficient stats by direct assignment
        self.register_buffer("mu0", torch.full((self.D,), cfg.mu0))
        self.register_buffer("kappa0", torch.tensor(cfg.kappa0))
        self.register_buffer("nu0", torch.tensor(max(cfg.nu0, self.D + 2)))  # safety
        self.register_buffer("Psi0", Psi0)

        # initialise posteriors with prior
        self.register_buffer("kappa_hat", torch.full((self.Kp1,), float(cfg.kappa0)))
        self.register_buffer("nu_hat",    torch.full((self.Kp1,), float(max(cfg.nu0, self.D + 2))))
        self.register_buffer("mu_hat",    self.mu0.expand(self.Kp1, self.D).clone())
        self.register_buffer("Psi_hat",   Psi0.expand(self.Kp1, self.D, self.D).clone())

        # cached π* from current β
        self._update_pi_from_u()

        # set initial Dirichlet params from the prior
        with torch.no_grad():
            for k in range(self.K):
                ph = self.alpha * self.pi_star.clone()
                ph[k] = ph[k] + self.kappa
                self.phi_hat[k] = torch.clamp(ph, min=1e-3)

        # optimiser only for u (β sticks)
        self.beta_optim = torch.optim.Adam([self.u], lr=cfg.beta_lr)

    # ---------- helpers ----------
    def _beta_from_u(self) -> torch.Tensor:
        return torch.sigmoid(self.u)  # (K,)
    def _pi_from_beta(self, beta: torch.Tensor) -> torch.Tensor:
        # stick-breaking to π* of length K+1 (remainder)
        K = beta.shape[0]
        remain = torch.cumprod(1 - beta, dim=0)
        pi = torch.empty(K + 1, device=beta.device, dtype=beta.dtype)
        pi[:-1] = beta * torch.cat([torch.ones(1, device=beta.device, dtype=beta.dtype), remain[:-1]], dim=0)
        pi[-1]  = remain[-1]
        return torch.clamp(pi, min=EPS)
    def _update_pi_from_u(self):
        with torch.no_grad():
            self.beta = self._beta_from_u().detach()
            self.pi_star = self._pi_from_beta(self.beta)  # (K+1,)

    # expected log A row from Dirichlet params
    @staticmethod
    def dirichlet_expected_log_row(phi_row: torch.Tensor) -> torch.Tensor:
        # phi_row: (K+1,)
        return digamma(phi_row) - digamma(phi_row.sum(-1, keepdim=True))

    # NIW expectations used in B_tk
    def _E_log_det_Sigma(self) -> torch.Tensor:
        # E[log |Σ_k|] under NIW posterior
        #  sum_d ψ((ν+1-d)/2) - D log 2 - log |Ψ|
        D = self.D
        v = self.nu_hat  # (K+1,)
        # log |Ψ_hat|
        logdetPsi = torch.linalg.slogdet(self.Psi_hat)[1]  # (K+1,)
        terms = torch.stack([digamma((v + 1.0 - d) * 0.5) for d in range(1, D + 1)], dim=-1).sum(-1)
        return terms - D * math.log(2.0) - logdetPsi  # (K+1,)

    def _E_Lambda(self) -> torch.Tensor:
        # E[Σ^{-1}] = ν Ψ^{-1}
        # compute inverse of Ψ_hat for all k
        invPsi = torch.linalg.inv(self.Psi_hat)  # (K+1,D,D)
        v = self.nu_hat.view(-1, 1, 1)
        return v * invPsi  # (K+1,D,D)

    # ---------- E-step: emissions, forward-backward ----------
    def _emission_loglik(self, z_mu: torch.Tensor, z_var: torch.Tensor) -> torch.Tensor:
        """
        z_mu: (T,D)   encoder means
        z_var:(T,D)   encoder diag variances
        returns log_B: (T, K+1)  E_q[log N(z_t | μ_k, Σ_k)]
        """
        T, D = z_mu.shape
        E_logdet = self._E_log_det_Sigma()                     # (K+1,)
        Lambda   = self._E_Lambda()                            # (K+1,D,D)
        mu_hat   = self.mu_hat                                  # (K+1,D)
        kappa    = self.kappa_hat                               # (K+1,)
        nu       = self.nu_hat                                  # (K+1,)

        # Precompute Λ diagonal for the diag term
        Lambda_diag = torch.diagonal(Lambda, dim1=-2, dim2=-1)  # (K+1,D)

        # trace(Λ * diag(z_var)) term: (T,K+1)
        t1 = torch.einsum('kd,td->tk', Lambda_diag, z_var)

        # (z - μ_hat)^T Λ (z - μ_hat) term: (T,K+1)
        diff = z_mu.unsqueeze(1) - mu_hat.unsqueeze(0)          # (T,K+1,D)
        t2 = torch.einsum('tki,kij,tkj->tk', diff, Lambda, diff)

        # NIW mean-uncertainty correction: Tr(Λ * Ψ_hat/(κ̂(ν-D-1))) = D * ν / (κ̂ (ν-D-1))
        denom = torch.clamp(nu - D - 1.0, min=1.0)
        t3 = (D * nu / (kappa * denom)).unsqueeze(0).expand(T, -1)  # (T,K+1)

        quad = t1 + t2 + t3

        log_B = -0.5 * (E_logdet.unsqueeze(0) + quad + D * math.log(2 * math.pi))
        return log_B  # (T,K+1)

    def _expected_log_A(self) -> torch.Tensor:
        # (K,K+1) rows for A; last column exists, last row doesn't exist (no row for remainder)
        rows = []
        for k in range(self.K):
            rows.append(self.dirichlet_expected_log_row(self.phi_hat[k]))
        return torch.stack(rows, dim=0)  # (K,K+1)

    def _forward_backward(self, log_B: torch.Tensor, log_pi: torch.Tensor,
                          log_A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        log_B: (T,K+1), log_pi: (K+1,), log_A: (K,K+1)
        Returns:
            gamma: (T,K+1)   posteriors q(h_t=k)
            xi:    (T-1,K,K+1) expected transitions q(h_t=j, h_{t+1}=k)
            logZ:  scalar normaliser
        """
        T, Kp1 = log_B.shape
        K = Kp1 - 1

        # forward (log-space)
        log_alpha = torch.empty(T, Kp1, device=log_B.device)
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            # for each dest k, sum over previous i: logsumexp(log_alpha[t-1,i] + log_A[i,k])
            trans = log_alpha[t - 1, :K].unsqueeze(-1) + log_A  # (K, K+1)
            msg = torch.logsumexp(trans, dim=0)                 # (K+1,)
            # allow transitions from remainder too by treating its row as π* (common trick)
            # i.e., α_{t-1,K+1} + log π* + ...
            msg = torch.logaddexp(msg, log_alpha[t - 1, -1] + log_pi)  # (K+1,)
            log_alpha[t] = log_B[t] + msg

        logZ = torch.logsumexp(log_alpha[-1], dim=-1)

        # backward (log-space)
        log_beta = torch.zeros_like(log_alpha)
        for t in range(T - 2, -1, -1):
            # next step messages
            tmp = log_B[t + 1].unsqueeze(0).expand(K, -1) + log_beta[t + 1].unsqueeze(0).expand(K, -1)  # (K,K+1)
            # transitions from i→k through next observation
            log_beta[t, :K] = torch.logsumexp(log_A + tmp, dim=1)  # sum over k
            # remainder row ~ π*
            log_beta[t, -1] = torch.logsumexp(log_pi + log_B[t + 1] + log_beta[t + 1], dim=-1)

        # marginals γ
        log_gamma = log_alpha + log_beta - logZ
        gamma = torch.softmax(log_gamma, dim=-1)  # (T,K+1)

        # pairwise ξ (t=0..T-2, i in 1..K, k in 1..K+1)
        xi = torch.empty(T - 1, K, Kp1, device=log_B.device)
        for t in range(T - 1):
            temp = (log_alpha[t, :K].unsqueeze(-1) +
                    log_A +
                    log_B[t + 1].unsqueeze(0) +
                    log_beta[t + 1].unsqueeze(0))
            # add remainder source: i=K+1 row ≈ π*
            temp = torch.logaddexp(temp, log_pi + log_B[t + 1] + log_beta[t + 1])
            temp = temp - torch.logsumexp(temp.reshape(-1), dim=0)
            xi[t] = torch.exp(temp)

        return gamma, xi, logZ

    # ---------- M-step / variational updates ----------
    @torch.no_grad()
    def _update_rows_dirichlet(self, xi: torch.Tensor):
        # xi: (T-1,K,K+1); Dirichlet params φ̂_kj = α π*_j + κ δ_{jk} + sum_t ξ̂_{t,kj}
        add = xi.sum(dim=0)  # (K,K+1)
        for k in range(self.K):
            base = self.alpha * self.pi_star.clone()
            base[k] = base[k] + self.kappa
            self.phi_hat[k] = torch.clamp(base + add[k], min=1e-3)

    @torch.no_grad()
    def _update_emissions_NIW(self, z_mu: torch.Tensor, z_var: torch.Tensor, gamma: torch.Tensor):
        """
        Conjugate NIW updates using sufficient stats weighted by gamma.
        """
        T = z_mu.size(0)
        # responsibilities per state
        Nk = gamma.sum(dim=0)                           # (K+1,)
        # first moment
        S1 = torch.einsum('tk,td->kd', gamma, z_mu)     # (K+1,D)
        # second moment: E[zz^T] = diag(var) + mu mu^T
        S2 = torch.einsum('tk,td,te->kde', gamma, z_mu, z_mu)  # (K+1,D,D)
        S2 = S2 + torch.einsum('tk,td->kd', gamma, z_var).unsqueeze(-1) * torch.eye(self.D, device=z_mu.device)

        # updates (B.3 bottom)
        self.kappa_hat = self.kappa0 + Nk
        self.nu_hat    = self.nu0 + Nk

        mu_num = self.kappa0 * self.mu0.unsqueeze(0) + S1
        self.mu_hat = mu_num / self.kappa_hat.unsqueeze(-1)

        # Ψ̂ = Ψ0 + S2 + κ0 μ0 μ0^T − κ̂ μ̂ μ̂^T
        mu0_outer = torch.einsum('d,e->de', self.mu0, self.mu0) * self.kappa0
        mu_hat_outer = torch.einsum('kd,ke->kde', self.mu_hat, self.mu_hat) * self.kappa_hat.unsqueeze(-1).unsqueeze(-1)
        self.Psi_hat = self.Psi0.unsqueeze(0) + S2 + mu0_outer.unsqueeze(0) - mu_hat_outer

        # numerical safety
        for k in range(self.Kp1):
            # make Ψ̂ PD
            self.Psi_hat[k] = 0.5 * (self.Psi_hat[k] + self.Psi_hat[k].T)
            # add tiny jitter if needed
            self.Psi_hat[k] = self.Psi_hat[k] + 1e-6 * torch.eye(self.D, device=self.Psi_hat.device)

    # ----- ELBO restricted to β (B.4), and one gradient step on u -----
    def _beta_elbo(self, gamma1: torch.Tensor) -> torch.Tensor:
        """
        ELBO restricted to β (Eq. B.4). gamma1 = q(h1=k) = gamma[0].
        """
        beta = torch.sigmoid(self.u)                  # (K,)
        pi   = self._pi_from_beta(beta)               # (K+1,)
        # refresh cache for outside users
        self.pi_star = pi.detach()

        # First term: (γ-1) ∑ log(1-β_k)
        term1 = (self.gamma - 1.0) * torch.log1p(-beta + EPS).sum()

        # Second term: ∑_k ∑_j [ (α π*_j + κ δ_{jk} − 1)(ψ(φ̂_kj) − ψ(∑ φ̂_km)) − log Γ(α π*_j + κ δ_{jk}) ]
        phi = self.phi_hat.detach()                   # (K,K+1)
        ElogA = digamma(phi) - digamma(phi.sum(-1, keepdim=True))  # (K,K+1)
        pi_row = self.alpha * pi.unsqueeze(0).expand(self.K, -1)   # (K,K+1)
        sticky = torch.zeros_like(pi_row)
        sticky[torch.arange(self.K), torch.arange(self.K)] = self.kappa
        conc = pi_row + sticky  # (K,K+1)

        term2 = ((conc - 1.0) * ElogA - lgamma(conc)).sum()

        # Third term: ∑_k r̂_{1k} log π_k*
        term3 = (gamma1 * (pi + EPS).log()).sum()

        return term1 + term2 + term3

    def step_optimize_beta(self, gamma_first: torch.Tensor, steps: int = 5):
        for _ in range(steps):
            self.beta_optim.zero_grad()
            L = -self._beta_elbo(gamma_first)  # minimise negative ELBO
            L.backward()
            self.beta_optim.step()
        self._update_pi_from_u()

    # ---------- PUBLIC: one online VI pass on a sequence ----------
    def online_update(self, z_mu: torch.Tensor, z_var: torch.Tensor, n_beta_steps: int = 5):
        """
        Perform one VI pass given a new sequence of encoder posteriors (means & diag vars).
        Args:
            z_mu : (T,D)
            z_var: (T,D)
        """
        z_mu  = z_mu.to(self.device)
        z_var = z_var.to(self.device)

        # E-step
        log_B  = self._emission_loglik(z_mu, z_var)         # (T,K+1)
        log_pi = (self.pi_star + EPS).log()                 # (K+1,)
        log_A  = self._expected_log_A()                     # (K,K+1)

        gamma, xi, _ = self._forward_backward(log_B, log_pi, log_A)

        # M-step (variational updates)
        self._update_rows_dirichlet(xi)
        self._update_emissions_NIW(z_mu, z_var, gamma)

        # optimise β (π*)
        self.step_optimize_beta(gamma_first=gamma[0], steps=n_beta_steps)

        # return posteriors for potential external logging
        return {
            "gamma": gamma.detach(),
            "xi": xi.detach(),
            "pi_star": self.pi_star.detach(),
            "phi_hat": self.phi_hat.detach(),
            "mu_hat": self.mu_hat.detach(),
            "kappa_hat": self.kappa_hat.detach(),
            "nu_hat": self.nu_hat.detach(),
        }

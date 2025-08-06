import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta

class StickyHDPHMM:
    """
    Sticky HDP-HMM variational inference (truncated) using PyTorch tensors.
    """
    def __init__(self, K, D, alpha=1.0, gamma=1.0, kappa=10.0,
                 mu0=None, kappa0=1.0, psi0=None, nu0=None,
                 max_iter=100, tol=1e-4, device=None):
        self.K = K
        self.D = D
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.device = device or torch.device('cpu')

        self.mu0 = torch.zeros(D, device=self.device) if mu0 is None else mu0.to(self.device)
        self.kappa0 = kappa0
        self.psi0 = torch.eye(D, device=self.device) if psi0 is None else psi0.to(self.device)
        self.nu0 = nu0 if nu0 is not None else D + 2

        self.max_iter = max_iter
        self.tol = tol

    def _initialize(self, X):
        T, D = X.shape
        K = self.K
        self.r = torch.full((T, K), 1.0/K, device=self.device)
        self.xi = torch.full((T-1, K, K), 1.0/(K*K), device=self.device)

        self.a = torch.ones(K, device=self.device)
        self.b = self.gamma * torch.ones(K, device=self.device)

        self.phi = torch.ones((K, K+1), device=self.device) * self.alpha

        self.kappa_hat = torch.full((K,), self.kappa0, device=self.device)
        self.nu_hat = torch.full((K,), self.nu0, device=self.device)
        self.mu_hat = self.mu0.unsqueeze(0).repeat(K,1)
        self.psi_hat = self.psi0.unsqueeze(0).repeat(K,1,1)

    def fit(self, X):
        X = X.to(self.device)
        T, D = X.shape
        self._initialize(X)
        elbo_old = -float('inf')
        for i in range(self.max_iter):
            self._e_step(X)
            self._update_transitions()
            self._update_emissions(X)
            self._update_sticks()
            # ELBO tracking can be added
        return self

    def _e_step(self, X):
        T = X.size(0)
        E_log_A = self._expected_log_A()
        E_log_B = self._expected_log_B(X)

        # Forward
        log_alpha = torch.zeros_like(E_log_B)
        log_alpha[0] = self._E_log_pi_k() + E_log_B[0]
        for t in range(1, T):
            prev = log_alpha[t-1].unsqueeze(1) + E_log_A
            log_alpha[t] = E_log_B[t] + torch.logsumexp(prev, dim=0)

        # Backward
        log_beta = torch.zeros_like(log_alpha)
        for t in range(T-2, -1, -1):
            nxt = E_log_A + E_log_B[t+1].unsqueeze(0) + log_beta[t+1].unsqueeze(0)
            log_beta[t] = torch.logsumexp(nxt, dim=1)

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
        self.r = torch.exp(log_gamma)

        # xi
        for t in range(T-1):
            A = (log_alpha[t].unsqueeze(1) + E_log_A +
                 E_log_B[t+1].unsqueeze(0) + log_beta[t+1].unsqueeze(0))
            A = A - torch.logsumexp(A, dim=(0,1))
            self.xi[t] = torch.exp(A)

    def _expected_log_A(self):
        K = self.K
        dig = torch.digamma
        E_log_A = torch.zeros((K,K), device=self.device)
        for i in range(K):
            row = dig(self.phi[i]) - dig(self.phi[i].sum())
            E_log_A[i] = row[:-1]
        return E_log_A

    def _E_log_pi_k(self):
        dig = torch.digamma
        a, b = self.a, self.b
        E1 = dig(a) - dig(a+b)
        E2 = dig(b) - dig(a+b)
        cum = torch.cumsum(F.pad(E2, (1,0))[:-1], dim=0)
        return E1 + cum

    def _expected_log_B(self, X):
        T = X.size(0)
        K, D = self.K, self.D
        E_log_B = torch.zeros((T,K), device=self.device)
        dig = torch.digamma
        for k in range(K):
            nu_k = self.nu_hat[k]
            psi_k = self.psi_hat[k]
            E_log_det = (dig((nu_k + 1 - torch.arange(1,D+1, device=self.device))/2).sum()
                         + D*torch.log(torch.tensor(2.0, device=self.device))
                         + torch.logdet(psi_k))
            mu_k = self.mu_hat[k]
            kappa_k = self.kappa_hat[k]
            inv_psi = torch.inverse(psi_k)
            diff = X - mu_k.unsqueeze(0)
            quad = (D/kappa_k) + nu_k * torch.sum(diff @ inv_psi * diff, dim=1)
            E_log_B[:,k] = 0.5*(E_log_det - quad - D*torch.log(torch.tensor(2*torch.pi, device=self.device)))
        return E_log_B

    def _update_transitions(self):
        K = self.K
        for i in range(K):
            counts = self.xi[:,i,:].sum(0)
            prior = self.alpha * self._pi_prior_row(i)
            sticky = self.kappa * torch.eye(K+1, device=self.device)[i]
            self.phi[i] = prior + sticky + counts

    def _pi_prior_row(self, i):
        K = self.K
        v = self.a/(self.a + self.b)
        pi = torch.zeros(K+1, device=self.device)
        cum = 1
        for k in range(K):
            pi[k] = v[k]*cum
            cum *= (1-v[k])
        pi[K] = cum
        return pi

    def _update_emissions(self, X):
        T = X.size(0)
        N_k = self.r.sum(0)
        x_bar = (self.r.t() @ X) / N_k.unsqueeze(1)
        for k in range(self.K):
            self.kappa_hat[k] = self.kappa0 + N_k[k]
            self.nu_hat[k] = self.nu0 + N_k[k]
            self.mu_hat[k] = (self.kappa0*self.mu0 + N_k[k]*x_bar[k]) / self.kappa_hat[k]
            diff0 = x_bar[k] - self.mu0
            S = ((self.r[:,k].unsqueeze(1)*(X - x_bar[k].unsqueeze(0))).t() @ (X - x_bar[k].unsqueeze(0)))
            self.psi_hat[k] = (self.psi0 + S +
                (self.kappa0*N_k[k])/(self.kappa0+N_k[k]) * torch.ger(diff0, diff0))

    def _update_sticks(self):
        N_k = self.r.sum(0)
        tail = torch.cumsum(N_k.flip(0), dim=0).flip(0)[1:].tolist()+[0]
        for k in range(self.K):
            self.a[k] = 1 + N_k[k]
            self.b[k] = self.gamma + tail[k]

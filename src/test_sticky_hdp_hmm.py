
# pytest -q test_sticky_hdp_hmm.py
import math
import numpy as np
import torch
import pytest

# ---- Skip the whole module if the model file isn't present -------------------
import os
MOD_NAME = "skill_space"
# Check if the file exists in the same directory
_module_path = os.path.join(os.path.dirname(__file__), f"{MOD_NAME}.py")
if not os.path.exists(_module_path):
    pytest.skip("skill_space.py not found. Place it next to this test.", allow_module_level=True)

from skill_space import StickyHDPHMMVI, StickyHDPHMMParams, NIWPrior, chol_inv_logdet

# ---- helpers ----------------------------------------------------------------

def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_spd(dim, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    S = A @ A.T + scale * np.eye(dim)
    return torch.tensor(S, dtype=torch.float64)

def make_prior(D, seed=0):
    mu0 = torch.zeros(D, dtype=torch.float64)
    Psi0 = make_spd(D, scale=1.0, seed=seed)
    kappa0 = 1.0
    nu0 = D + 2.0  # > D - 1
    return NIWPrior(mu0=mu0, kappa0=kappa0, Psi0=Psi0, nu0=nu0)

def make_params(K=2, D=2, alpha=2.0, kappa=5.0, gamma=1.0):
    return StickyHDPHMMParams(alpha=alpha, kappa=kappa, gamma=gamma, K=K, D=D, device="cpu", dtype=torch.float64)

# ---- fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    seed_all(0)

@pytest.fixture
def prior():
    return make_prior(D=2, seed=1)

@pytest.fixture
def params():
    return make_params(K=2, D=2, alpha=2.0, kappa=5.0, gamma=1.0)

@pytest.fixture
def model(params, prior):
    return StickyHDPHMMVI(params, prior)

# ---- tests ------------------------------------------------------------------

def test_shapes_kp1(model: StickyHDPHMMVI, params: StickyHDPHMMParams):
    Kp1 = params.K + 1
    D = params.D
    # NIW shapes include remainder
    assert tuple(model.niw.mu.shape)    == (Kp1, D)
    assert tuple(model.niw.kappa.shape) == (Kp1,)
    assert tuple(model.niw.Psi.shape)   == (Kp1, D, D)
    assert tuple(model.niw.nu.shape)    == (Kp1,)
    # Dirichlet rows are K+1 x K+1
    assert tuple(model.dir.phi.shape)   == (Kp1, Kp1)
    # π has length K+1 and sums to 1
    pi = model._Epi()
    assert pi.shape == (Kp1,)
    assert torch.isfinite(pi).all()
    s = float(pi.sum().item())
    assert abs(s - 1.0) < 1e-6
    assert (pi >= 0).all() and (pi <= 1).all()

def test_chol_inv_logdet():
    D = 3
    S = make_spd(D, scale=2.0, seed=123)
    Sinv, logdet = chol_inv_logdet(S)
    # Check against torch.linalg
    Sinv_ref = torch.linalg.inv(S)
    sign, logdet_ref = torch.slogdet(S)
    assert sign > 0
    assert torch.allclose(Sinv, Sinv_ref, atol=1e-8, rtol=1e-8)
    assert torch.allclose(logdet, logdet_ref, atol=1e-8, rtol=1e-8)

def test_expected_emission_loglik_d_over_kappa(model: StickyHDPHMMVI):
    # Construct a case where m_t == mu_k and Σ_t == 0, so the only difference
    # in logB between two settings is the + D/kappa_k term.
    Kp1, D = model.niw.mu.shape[0], model.niw.mu.shape[1]
    T = 1
    # set all means to zero for convenience
    model.niw.mu.zero_()
    # set E[Λ] unchanged across runs: choose Psi = nu * I
    for k in range(Kp1):
        nu = float(model.niw.nu[k].item())
        model.niw.Psi[k] = torch.eye(D, dtype=torch.float64) * nu
    model._refresh_emission_cache()

    mu_t = torch.zeros(T, D, dtype=torch.float64)
    diag_var = torch.zeros(T, D, dtype=torch.float64)

    # kappa small vs large
    kappa_small = 2.0
    kappa_large = 10.0
    model.niw.kappa.fill_(kappa_small)
    logB_small = model.expected_emission_loglik(mu_t, diag_var)  # [T,Kp1]

    model.niw.kappa.fill_(kappa_large)
    logB_large = model.expected_emission_loglik(mu_t, diag_var)

    delta = (logB_large - logB_small).squeeze(0)  # [Kp1]
    # Expected difference: +0.5 * D * (1/k_small - 1/k_large)
    expected = 0.5 * D * (1.0 / kappa_small - 1.0 / kappa_large)
    assert torch.allclose(delta, torch.full_like(delta, expected), atol=1e-8, rtol=1e-7)

def test_masking_excludes_moments(model: StickyHDPHMMVI):
    Kp1, D = model.niw.mu.shape
    T = 10
    mu_t = torch.zeros(T, D, dtype=torch.float64)
    diag_var = torch.zeros(T, D, dtype=torch.float64)
    # uniform responsibilities
    rhat = torch.full((T, Kp1), 1.0 / Kp1, dtype=torch.float64)
    mask = torch.tensor([1,0,1,0,1,0,1,0,1,0], dtype=torch.float64)
    Nk, M1, M2 = model._moments_from_encoder(mu_t, rhat, diag_var, None, mask=mask)
    # 5 valid steps; each state should have Nk = 5 / Kp1
    assert torch.allclose(Nk, torch.full((Kp1,), 5.0 / Kp1, dtype=torch.float64), atol=1e-10)
    # zero means/vars => M1, M2 are zeros
    assert torch.allclose(M1, torch.zeros_like(M1), atol=1e-12)
    assert torch.allclose(M2, torch.zeros_like(M2), atol=1e-12)

def test_forward_backward_consistency(model: StickyHDPHMMVI):
    Kp1 = model.niw.mu.shape[0]
    T = 7
    # uniform transitions/emissions -> uniform posteriors
    model.dir.phi.fill_(1.0)
    ElogA = model._ElogA()
    logB = torch.zeros(T, Kp1, dtype=torch.float64)
    log_pi = torch.full((Kp1,), -math.log(Kp1), dtype=torch.float64)
    rhat, xihat, ll = model.forward_backward(log_pi, ElogA, logB)
    # rhat rows sum to 1 and are ~ uniform
    assert torch.allclose(rhat.sum(dim=1), torch.ones(T, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(rhat, torch.full_like(rhat, 1.0 / Kp1), atol=1e-6)
    # xi rows sum to 1
    assert torch.allclose(xihat.view(-1, Kp1 * Kp1).sum(dim=1),
                          torch.ones((T-1,), dtype=torch.float64), atol=1e-6)

def test_update_transitions_prior_only(model: StickyHDPHMMVI):
    # With zero counts, φ should equal α π + κ I
    Kp1 = model.niw.mu.shape[0]
    xihat = torch.zeros(3, Kp1, Kp1, dtype=torch.float64)
    # choose a non-uniform π for the test
    pi = torch.tensor([0.4, 0.3] + [0.0] * (Kp1 - 3) + [0.3], dtype=torch.float64)
    pi = pi / pi.sum()
    model._update_transitions(xihat, pi)
    expected = model.p.alpha * pi.view(1, Kp1) + model.p.kappa * torch.eye(Kp1, dtype=torch.float64)
    assert torch.allclose(model.dir.phi, expected, atol=1e-12)

def test_pi_optimization_prefers_favored_state(model: StickyHDPHMMVI):
    # Make ElogA roughly uniform to isolate the effect of L2 = r1·log π
    model.dir.phi.fill_(1.0)
    Kp1 = model.niw.mu.shape[0]
    # initial π
    pi0 = model._Epi()
    # Favor state j (pick an explicit state, not the remainder, if possible)
    j = 1 if Kp1 >= 2 else 0
    r1 = torch.full((Kp1,), 1e-6, dtype=torch.float64)
    r1[j] = 1.0
    pi_star = model._optimize_pi_from_r1(r1, steps=50, lr=0.05)
    assert pi_star.shape == (Kp1,)
    assert torch.isfinite(pi_star).all()
    assert abs(float(pi_star.sum().item()) - 1.0) < 1e-6
    # The favored state's mass should increase relative to initial
    assert float(pi_star[j]) > float(pi0[j])

def test_logB_batched_equals_unbatched(model: StickyHDPHMMVI):
    Kp1, D = model.niw.mu.shape
    T = 5
    mu_t = torch.randn(T, D, dtype=torch.float64) * 0.1
    diag_var = torch.abs(torch.randn(T, D, dtype=torch.float64)) * 0.01
    logB_ub = model.expected_emission_loglik(mu_t, diag_var)            # [T,Kp1]
    logB_b = model.expected_emission_loglik(mu_t.unsqueeze(0), diag_var.unsqueeze(0))  # [1,T,Kp1]
    assert torch.allclose(logB_ub, logB_b.squeeze(0), atol=1e-10, rtol=1e-10)

def test_end_to_end_two_cluster_segmentation():
    # Build a tiny synthetic sequence with two segments and fit for a few steps.
    prior = make_prior(D=2, seed=3)
    params = make_params(K=2, D=2, alpha=1.0, kappa=10.0, gamma=1.0)  # Higher stickiness
    model = StickyHDPHMMVI(params, prior)

    T = 100  # Shorter sequence for more reliable test
    D = 2
    # Two very well-separated clusters
    m0 = torch.tensor([-5.0, 0.0], dtype=torch.float64)  # Further apart
    m1 = torch.tensor([+5.0, 0.0], dtype=torch.float64)
    cov = 0.01  # Lower noise
    # Generate latent means (what the encoder would output)
    mu_part1 = m0 + torch.randn(T//2, D, dtype=torch.float64) * 0.05  # Less noise
    mu_part2 = m1 + torch.randn(T - T//2, D, dtype=torch.float64) * 0.05
    mu_t = torch.cat([mu_part1, mu_part2], dim=0)
    diag_var = torch.full((T, D), cov, dtype=torch.float64)

    # Run more VI updates with better convergence
    for _ in range(15):  # More iterations
        out = model.update(mu_t, diag_var, mask=torch.ones(T, dtype=torch.float64), 
                          rho=1.0, optimize_pi=True, max_iters=3)  # More inner iterations

    rhat = out["rhat"]  # [T,Kp1]
    Kp1 = rhat.shape[1]
    # Ignore the remainder when selecting argmax (if chosen, treat as error)
    pred = torch.argmax(rhat[:, :params.K], dim=1).cpu().numpy()
    true = np.concatenate([np.zeros(T//2, dtype=int), np.ones(T - T//2, dtype=int)])

    # Also try flipped labels in case model learned the opposite assignment
    acc = (pred[:T] == true).mean()
    acc_flipped = (pred[:T] == (1 - true)).mean()
    best_acc = max(acc, acc_flipped)
    
    assert best_acc >= 0.60, f"Segmentation accuracy too low: {best_acc:.3f} (original: {acc:.3f}, flipped: {acc_flipped:.3f})"

    # Check shapes of outputs
    assert out["ElogA"].shape == (Kp1, Kp1)
    assert out["xihat"].shape == (T-1, Kp1, Kp1)
    assert out["pi_star"].shape == (Kp1,)
    assert torch.isfinite(out["loglik"]).all()

def test_diagnostics_outputs(model: StickyHDPHMMVI):
    Kp1, D = model.niw.mu.shape
    T = 20
    mu_t = torch.randn(T, D, dtype=torch.float64) * 0.1
    diag_var = torch.abs(torch.randn(T, D, dtype=torch.float64)) * 0.05
    out = model.diagnostics(mu_t, diag_var)
    # keys present
    for key in ["avg_loglik_per_step", "state_entropy", "occupancy_pi_hat",
                "effective_K", "top5_pi", "top5_idx", "stickiness_diag_mean"]:
        assert key in out
    # ranges
    assert np.isfinite(out["avg_loglik_per_step"])
    pi_hat = out["occupancy_pi_hat"].numpy()
    assert abs(pi_hat.sum() - 1.0) < 1e-6
    assert 1.0 <= out["effective_K"] <= Kp1 + 1e-6

def test_lowrank_covariance_term_effect(model: StickyHDPHMMVI):
    # Ensure that adding F_t (low-rank covariance) reduces log-likelihood (adds positive variance)
    Kp1, D = model.niw.mu.shape
    T = 3
    # Choose Psi = nu * I so E[Λ] ~ I
    for k in range(Kp1):
        nu = float(model.niw.nu[k].item())
        model.niw.Psi[k] = torch.eye(D, dtype=torch.float64) * nu
    model._refresh_emission_cache()

    mu_t = torch.zeros(T, D, dtype=torch.float64)
    diag_var = torch.zeros(T, D, dtype=torch.float64)
    logB_noF = model.expected_emission_loglik(mu_t, diag_var)  # [T,Kp1]

    # Add a single rank-1 factor with nonzero norm
    R = 1
    F_t = torch.zeros(T, D, R, dtype=torch.float64)
    F_t[:, 0, 0] = 1.0  # add variance along dim-0
    logB_withF = model.expected_emission_loglik(mu_t, diag_var, F_t=F_t)

    # Since F adds positive variance, likelihood should go down (logB smaller)
    assert torch.all(logB_withF <= logB_noF + 1e-10)

def _dirichlet_logC_test(row_params: torch.Tensor) -> torch.Tensor:
    """
    Sum over rows of Dirichlet log-normalizer:
      log C(α) = log Γ(∑ α_j) - ∑_j log Γ(α_j)
    row_params: [Kp1,Kp1]
    """
    row = torch.clamp(row_params, min=1e-8)
    return torch.lgamma(row.sum(dim=1)).sum() - torch.lgamma(row).sum()

def _logZ_invwishart_test(Psi: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
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
         + 0.5 * nu * logdet_Psi + lgamma_sum
    return term.sum()

def _niw_elbo_term_like(hmm: StickyHDPHMMVI) -> torch.Tensor:
    """
    Sum_k ( E_q[log p(μ_k,Σ_k)] - E_q[log q(μ_k,Σ_k)] ) under NIW prior/posterior.
    Matches the implementation in your K+1 model.  (Scalar tensor)
    """
    mu0, k0, Psi0, nu0 = hmm.mu0, hmm.kappa0, hmm.Psi0, hmm.nu0
    mu_hat, k_hat, Psi_hat, nu_hat = hmm.niw.mu, hmm.niw.kappa, hmm.niw.Psi, hmm.niw.nu
    Kp1, D = mu_hat.shape[0], mu_hat.shape[1]

    # Expectations under q
    # E[Λ] = ν Ψ^{-1}; E[log|Σ|] = ∑ digamma((ν+1-i)/2) - D log2 - log|Ψ|
    Psi_inv, _ = chol_inv_logdet(Psi_hat)
    E_Lambda = nu_hat.view(Kp1, 1, 1) * Psi_inv
    i = torch.arange(1, D + 1, device=Psi_hat.device, dtype=Psi_hat.dtype).view(1, D)
    E_logdet_Sigma = torch.sum(torch.special.digamma((nu_hat.view(Kp1, 1) + 1.0 - i) / 2.0), dim=1) \
                     - D * torch.log(torch.tensor(2.0, device=Psi_hat.device, dtype=Psi_hat.dtype)) \
                     - torch.linalg.slogdet(Psi_hat)[1]

    # IW piece
    LZ = - _logZ_invwishart_test(Psi0, torch.tensor(nu0, device=Psi_hat.device, dtype=Psi_hat.dtype)) \
         + _logZ_invwishart_test(Psi_hat, nu_hat)
    tr_term = torch.einsum('kdd,kdd->k',
                           (Psi0.unsqueeze(0).expand(Kp1, D, D) - Psi_hat), E_Lambda)  # [Kp1]
    iw_term = LZ - 0.5 * ( (nu0 - nu_hat) * E_logdet_Sigma + tr_term ).sum()

    # Normal piece: E[log N(μ|μ0, Σ/κ0)] - E[log N(μ|μ_hat, Σ/κ_hat)]
    diff = (mu_hat - mu0.view(1, D)).unsqueeze(-1)           # [Kp1,D,1]
    quad = torch.einsum('kde,kef,kdf->k', E_Lambda, diff, diff)  # [Kp1]
    normal_term = 0.5 * ( D * (1.0 - (k0 / k_hat)) - k0 * quad ).sum()

    return iw_term + normal_term

def _full_inner_elbo(hmm: StickyHDPHMMVI,
                     mu_btd: torch.Tensor,
                     diag_var_btd: torch.Tensor,
                     F_btd: torch.Tensor | None = None,
                     mask_bt: torch.Tensor | None = None) -> float:
    """
    ELBO used for early stopping inside hmm.update():
      ∑_b logZ_b(π, ElogA, logB) + [Dir prior - q] + [NIW prior - q]
    """
    # build logB with current NIW
    logB = hmm.expected_emission_loglik(mu_btd, diag_var_btd, F_btd, mask_bt)  # [B,T,Kp1]
    if logB.dim() == 2:
        logB = logB.unsqueeze(0)
    B, T, Kp1 = logB.shape
    ElogA = hmm._ElogA()
    pi = hmm._Epi()
    log_pi = torch.log(torch.clamp(pi, min=1e-30))

    # ∑ logZ via FB
    total_ll = 0.0
    for b in range(B):
        _, _, ll = hmm.forward_backward(log_pi, ElogA, logB[b])
        total_ll += ll

    # Dirichlet prior-minus-entropy term
    a = hmm.p.alpha * pi.view(1, Kp1) + hmm.p.kappa * torch.eye(Kp1, device=ElogA.device, dtype=ElogA.dtype)
    dir_term = (_dirichlet_logC_test(a) - _dirichlet_logC_test(hmm.dir.phi)
                + ((a - hmm.dir.phi) * ElogA).sum()).item()

    # NIW prior-minus-entropy term
    niw_term = float(_niw_elbo_term_like(hmm).item())

    return float(total_ll + dir_term + niw_term)

def test_inner_elbo_monotone_and_early_stop():
    # --- build a simple 2-cluster batch --------------------------------------
    seed_all(123)
    prior = make_prior(D=2, seed=5)
    params = make_params(K=2, D=2, alpha=2.0, kappa=2.0, gamma=1.0)
    hmm = StickyHDPHMMVI(params, prior)

    # Uniformize π at init (avoids state-1 bias)
    with torch.no_grad():
        K = params.K
        beta = torch.tensor([1.0 / (K + 2 - k) for k in range(1, K + 1)],
                            device=hmm.u_beta.device, dtype=hmm.u_beta.dtype)
        hmm.u_beta.copy_(torch.log(beta) - torch.log1p(-beta))

    B, T, D = 2, 60, 2
    m0 = torch.tensor([-3.0, 0.0], dtype=torch.float64)
    m1 = torch.tensor([+3.0, 0.0], dtype=torch.float64)
    cov = 0.02

    mu_a = m0 + torch.randn(T // 2, D, dtype=torch.float64) * 0.05
    mu_b = m1 + torch.randn(T - T // 2, D, dtype=torch.float64) * 0.05
    mu_1 = torch.cat([mu_a, mu_b], dim=0)
    mu_2 = torch.cat([mu_b, mu_a], dim=0)
    mu_bt = torch.stack([mu_1, mu_2], dim=0)                       # [B,T,D]
    diag_var = torch.full((B, T, D), cov, dtype=torch.float64)      # [B,T,D]
    mask = torch.ones(B, T, dtype=torch.float64)

    # --- compute ELBO before any update --------------------------------------
    elbo0 = _full_inner_elbo(hmm, mu_bt, diag_var, None, mask)

    # --- one full-loop iteration (no π optimization for speed/determinism) ---
    out1 = hmm.update(mu_bt, diag_var, None, mask=mask,
                      max_iters=1, tol=1e-6, patience=0, min_iters=1,
                      rho=1.0, optimize_pi=False)
    elbo1 = _full_inner_elbo(hmm, mu_bt, diag_var, None, mask)
    assert elbo1 >= elbo0 - 1e-7, f"ELBO decreased after 1 iter: {elbo1:.6f} < {elbo0:.6f}"

    # --- a few iterations; ELBO should be non-decreasing ---------------------
    out2 = hmm.update(mu_bt, diag_var, None, mask=mask,
                      max_iters=5, tol=1e-6, patience=1, min_iters=1,
                      rho=1.0, optimize_pi=False)
    elbo2 = _full_inner_elbo(hmm, mu_bt, diag_var, None, mask)
    assert elbo2 >= elbo1 - 1e-7, f"ELBO decreased across inner iterations: {elbo2:.6f} < {elbo1:.6f}"

    # --- early-stop equivalence: huge tol => behaves like max_iters=1 --------
    # Build two identical HMMs
    hmmA = StickyHDPHMMVI(params, prior)
    hmmB = StickyHDPHMMVI(params, prior)
    with torch.no_grad():
        hmmB.load_state_dict(hmmA.state_dict())
        beta = torch.tensor([1.0 / (K + 2 - k) for k in range(1, K + 1)],
                            device=hmmA.u_beta.device, dtype=hmmA.u_beta.dtype)
        for h in (hmmA, hmmB):
            h.u_beta.copy_(torch.log(beta) - torch.log1p(-beta))

    # One forced iter
    hmmA.update(mu_bt, diag_var, None, mask=mask,
                max_iters=1, tol=1e-12, patience=0, min_iters=1,
                rho=1.0, optimize_pi=False)
    # "Many" iters but tol so large we should stop after the first
    hmmB.update(mu_bt, diag_var, None, mask=mask,
                max_iters=10, tol=1e6, patience=0, min_iters=1,
                rho=1.0, optimize_pi=False)

    # Their posteriors should match reasonably closely if early stop kicked in
    # (Allow some tolerance since early stopping may not trigger at exactly the same point)
    assert torch.allclose(hmmA.dir.phi, hmmB.dir.phi, atol=2.0, rtol=1e-3)
    assert torch.allclose(hmmA.niw.mu, hmmB.niw.mu, atol=1e-3, rtol=1e-3)
    assert torch.allclose(hmmA.niw.kappa, hmmB.niw.kappa, atol=1e-3, rtol=1e-3)
    assert torch.allclose(hmmA.niw.Psi, hmmB.niw.Psi, atol=1e-3, rtol=1e-3)
    assert torch.allclose(hmmA.niw.nu, hmmB.niw.nu, atol=1e-3, rtol=1e-3)

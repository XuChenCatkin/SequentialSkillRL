import torch

def _chol_logdet(A):
    # A: [B, R, R] SPD
    L = torch.linalg.cholesky(A)
    return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1), L

def _sigma_inv_matvec(logvar_p, Fp, v, jitter=1e-6):
    """
    Compute (D_p + F_p F_p^T)^{-1} v using Woodbury.
    logvar_p: [B, D], Fp: [B, D, Rp] or None, v: [B, D]
    returns: [B, D]
    """
    Dinv = torch.exp(-logvar_p)
    if Fp is None or Fp.shape[-1] == 0:
        return Dinv * v

    Dinv_v  = Dinv * v                              # [B, D]
    Dinv_Fp = Dinv.unsqueeze(-1) * Fp               # [B, D, Rp]
    K = torch.matmul(Fp.transpose(1, 2), Dinv_Fp)   # [B, Rp, Rp] = F^T D^{-1} F
    eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype).expand_as(K)
    logdetK, L = _chol_logdet(K + eye * (1.0 + jitter))  # Cholesky of I + F^T D^{-1} F

    rhs = torch.matmul(Fp.transpose(1, 2), Dinv_v.unsqueeze(-1))  # [B, Rp, 1] = F^T D^{-1} v
    tmp = torch.cholesky_solve(rhs, L).squeeze(-1)                # [B, Rp]   = (I + F^T D^{-1}F)^{-1}(...)
    return Dinv_v - torch.matmul(Dinv_Fp, tmp.unsqueeze(-1)).squeeze(-1)

def _sigma_inv_matmul(logvar_p, Fp, X, jitter=1e-6):
    """
    Compute (D_p + F_p F_p^T)^{-1} X for X:[B, D, Rq]
    returns: [B, D, Rq]
    """
    Dinv = torch.exp(-logvar_p)
    if Fp is None or Fp.shape[-1] == 0:
        return Dinv.unsqueeze(-1) * X

    Dinv_X  = Dinv.unsqueeze(-1) * X                # [B, D, Rq]
    Dinv_Fp = Dinv.unsqueeze(-1) * Fp               # [B, D, Rp]
    K = torch.matmul(Fp.transpose(1, 2), Dinv_Fp)   # [B, Rp, Rp]
    eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype).expand_as(K)
    _, L = _chol_logdet(K + eye * (1.0 + jitter))

    T = torch.matmul(Fp.transpose(1, 2), Dinv_X)    # [B, Rp, Rq]
    Y = torch.cholesky_solve(T, L)                  # [B, Rp, Rq]
    return Dinv_X - torch.matmul(Dinv_Fp, Y)        # [B, D, Rq]

def _sigma_inv_diag(logvar_p, Fp, jitter=1e-6):
    """
    Diagonal of (D_p + F_p F_p^T)^{-1}.
    returns: [B, D]
    """
    Dinv = torch.exp(-logvar_p)
    if Fp is None or Fp.shape[-1] == 0:
        return Dinv

    Dinv_Fp = Dinv.unsqueeze(-1) * Fp               # [B, D, Rp]
    K = torch.matmul(Fp.transpose(1, 2), Dinv_Fp)   # [B, Rp, Rp]
    eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype).expand_as(K)
    _, L = _chol_logdet(K + eye * (1.0 + jitter))

    # Y = (I + F^T D^{-1} F)^{-1} F^T
    Y = torch.cholesky_solve(Fp.transpose(1, 2), L)       # [B, Rp, D]
    diag_FY = (Fp * Y.transpose(1, 2)).sum(dim=2)         # diag(F Y) ∈ [B, D]
    return Dinv - (Dinv * Dinv) * diag_FY                 # diag(D^{-1} - D^{-1} F A^{-1} F^T D^{-1})

def _logdet_sigma(logvar, F, jitter=1e-6):
    """
    log|D + F F^T| via determinant lemma.
    """
    logdet_D = logvar.sum(dim=1)                 # sum log σ^2
    if F is None or F.shape[-1] == 0:
        return logdet_D

    Dinv_F = torch.exp(-logvar).unsqueeze(-1) * F
    K = torch.matmul(F.transpose(1, 2), Dinv_F)  # [B, R, R]
    eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype).expand_as(K)
    logdet_K, _ = _chol_logdet(K + eye * (1.0 + jitter))
    return logdet_D + logdet_K

def kl_gaussian_lowrank_q_p(mu_q, logvar_q, F_q,
                            mu_p, logvar_p, F_p,
                            jitter=1e-6):
    """
    KL[q||p] for q ~ N(mu_q, D_q + F_q F_q^T), p ~ N(mu_p, D_p + F_p F_p^T)
    Shapes:
      mu_*: [B, D], logvar_*: [B, D], F_*: [B, D, R_*] or None
    Returns: [B]
    """
    # log-dets
    logdet_p = _logdet_sigma(logvar_p, F_p, jitter)
    logdet_q = _logdet_sigma(logvar_q, F_q, jitter)

    # trace term: tr(Sigma_p^{-1} Sigma_q)
    diag_inv_p = _sigma_inv_diag(logvar_p, F_p, jitter)                 # [B, D]
    tr_diag = (diag_inv_p * torch.exp(logvar_q)).sum(dim=1)             # [B]
    if F_q is None or F_q.shape[-1] == 0:
        tr_lowrank = torch.zeros_like(tr_diag)
    else:
        SigmaInvFq = _sigma_inv_matmul(logvar_p, F_p, F_q, jitter)      # [B, D, Rq]
        tr_lowrank = (SigmaInvFq * F_q).sum(dim=(1, 2))                 # tr(F_q^T Σ^{-1} F_q)
    trace_term = tr_diag + tr_lowrank

    # quadratic term
    dmu = (mu_q - mu_p)                                                 # [B, D]
    SigmaInv_dmu = _sigma_inv_matvec(logvar_p, F_p, dmu, jitter)        # [B, D]
    quad = (dmu * SigmaInv_dmu).sum(dim=1)                              # [B]

    D = mu_q.size(1)
    kl = 0.5 * (logdet_p - logdet_q - D + trace_term + quad)
    return kl

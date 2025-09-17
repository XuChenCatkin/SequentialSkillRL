from __future__ import annotations
import os, time, math, json, random, dataclasses, logging
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# MiniHack / Gym
import gymnasium as gym
from minihack import MiniHackNavigation, MiniHackSkill, MiniHack
from nle import nethack  # action id space (615)

# --- your code ---
from src.model import MultiModalHackVAE, VAEConfig, vae_loss
from src.skill_space import StickyHDPHMMVI, StickyHDPHMMParams, NIWPrior
from src.data_collection import NetHackDataCollector, one_hot, compute_passability_and_safety
from utils.action_utils import ACTION_DIM, KEYPRESS_INDEX_MAPPING

# KL helper already used by your VAE loss
from utils.math_utils import kl_gaussian_lowrank_q_p

# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # rollout / optimisation
    num_envs: int = 8
    rollout_len: int = 128       # T per PPO update
    total_updates: int = 20000
    minibatch_size: int = 2048
    epochs_per_update: int = 3
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    vf_learning_rate: Optional[float] = None  # if None, use learning_rate
    policy_uses_skill: bool = True            # concatenate q(h_t) to the policy input
    deterministic_eval: bool = True

@dataclass
class CuriosityConfig:
    use_dyn_kl: bool = True       # (A) dynamics surprise
    use_skill_entropy: bool = True # (B) skill-posterior entropy (will be boundary-gated if enabled)
    use_skill_boundary_gate: bool = True     # multiply H_t by 1[ΔH_t > gate_delta_eps]
    gate_delta_eps: float = 1e-3             # small positive threshold to de-noise the gate
    use_skill_transition_novelty: bool = True # (C) skill-transition novelty
    use_rnd: bool = False         # RND baseline (set True for baseline run)

    # Annealing: eta(t) = eta0 * exp(-t / tau)
    eta0_dyn: float = 1.0
    tau_dyn: float = 3e6
    eta0_hdp: float = 1.0         # for boundary-gated skill entropy
    tau_hdp: float = 3e6
    eta0_stn: float = 1.0         # anneal multiplier for skill‑transition novelty
    tau_stn: float = 3e6
    eta0_rnd: float = 0.25        # keep smaller by default
    tau_rnd: float = 3e6

    # EMA norm for each raw term
    ema_beta: float = 0.99
    eps: float = 1e-8

@dataclass
class HMMOnlineConfig:
    # HMM update cadence and footprint
    hmm_update_every: int = 10_240          # env steps between HMM refreshes
    hmm_update_growth: float = 1.30         # after each refresh: interval *= growth
    hmm_update_every_cap: int = 60_000      # cap interval to avoid going too sparse
    hmm_fit_window: int = 400_000           # how many most recent steps to re-fit on
    hmm_max_iters: int = 7                  # inner VI iterations
    hmm_tol: float = 1e-2                   # relative ELBO tolerance
    hmm_elbo_drop_tol: float = 1e-2         # relative ELBO drop tolerance for early stopping
    rho_emission: float = 0.05              # streaming blend
    rho_transition: Optional[float] = None  # default to rho_emission if None
    optimise_pi: bool = True                # optimize π using mean r1
    # π optimization parameters
    pi_steps: int = 200                     # π optimization steps
    pi_lr: float = 0.05                     # π optimization learning rate
    pi_early_stopping_patience: int = 10   # early stopping patience for π optimization
    pi_early_stopping_min_delta: float = 1e-5,  # early stopping min delta for π optimization
    emission_mode: str = "sample",          # "sample" or "mean" or "expected" or "student_t"
    student_t_use_sample: bool = False      # if using student_t, use sampled z for logB (else mean)

@dataclass
class VAEOnlineConfig:
    # Full VAE online update - synchronized with HMM updates
    vae_update_every: int = 10_240          # env steps between VAE refreshes (same as HMM)
    vae_update_growth: float = 1.30         # after each refresh: interval *= growth (same as HMM)
    vae_update_every_cap: int = 60_000      # cap interval to avoid going too sparse (same as HMM)
    vae_lr: float = 1e-4                    # learning rate for full VAE update
    vae_steps_per_call: int = 8             # More mini-updates per call for better adaptation
    span_len: int = 64                      # random short spans for stability/memory
    mini_batch_B: int = 32                  # number of random spans per step
    # VAE loss coefficients for online training (lighter than offline)
    training_config: VAEConfig = VAEConfig()  # use default VAEConfig
    mi_beta: float = 0.1                    # Mutual information beta (lighter online)
    tc_beta: float = 0.1                    # Total correlation beta
    dw_beta: float = 0.1                    # Dimension-wise KL beta

@dataclass
class RNDConfig:
    proj_dim: int = 128
    hidden: int = 256
    lr: float = 1e-3
    update_per_rollout: int = 2

@dataclass
class TrainConfig:
    env_id: str = "MiniHack-Quest-Hard-v0"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "./runs/minihack_ppo"
    save_every: int = 50_000  # env steps
    eval_every: int = 50_000  # env steps
    eval_episodes: int = 10

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class EMANormalizer:
    """
    Track var with EMA and normalise new values.
    If center=False, we divide by running std only (keeps non‑negative signals non‑negative).
    """
    def __init__(self, beta=0.99, eps=1e-8, device="cpu", center: bool = True):
        self.beta = beta; self.eps = eps; self.center = center
        self.mean = torch.zeros((), device=device)
        self.var  = torch.ones((), device=device)
        self.initialised = False
    @torch.no_grad()
    def update(self, x: torch.Tensor) -> torch.Tensor:
        val = x.detach()
        m = val.mean(); v = val.var(unbiased=False) + self.eps
        if not self.initialised:
            self.mean.copy_(m); self.var.copy_(v); self.initialised = True
        else:
            self.mean = self.beta * self.mean + (1 - self.beta) * m
            self.var  = self.beta * self.var  + (1 - self.beta) * v
        y = (x - self.mean) if self.center else x
        return y / (self.var.sqrt() + self.eps)

def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = torch.zeros(*indices.shape, num_classes, device=indices.device)
    return y.scatter_(-1, indices.long().unsqueeze(-1), 1.0)

def obs_to_device(obs: dict, device, hero_info):
    # NLE/MiniHack observation keys; robust to dict/np/tensors
    def to_t(x):
        t = torch.as_tensor(x)
        return t.to(device, dtype=torch.long)
    
    out = {
        "game_chars": to_t(obs.get("chars") if "chars" in obs else obs["glyphs_char"]),
        "game_colors": to_t(obs.get("colors") if "colors" in obs else obs["glyphs_color"]),
        "blstats": to_t(obs["blstats"]),
        "message_chars": to_t(obs.get("message", np.zeros((obs["blstats"].shape[0], 256), np.int64))),
        "hero_info": to_t(hero_info)
    }
    # ensure batch dimension for vectorized envs
    for k, v in out.items():
        if v.dim() == 2 and k in ["game_chars", "game_colors"]:
            # single env -> [21,79], add batch
            out[k] = v.unsqueeze(0)
        elif v.dim() == 1 and k == "blstats":
            out[k] = v.unsqueeze(0)
        elif v.dim() == 1 and k in ["message_chars", "hero_info"]:
            out[k] = v.unsqueeze(0)
    return out

def message_ascii_to_string(message_chars: np.ndarray) -> str:
    """Convert ASCII codes from message observation to string."""
    # Handle different input types
    if hasattr(message_chars, 'cpu'):  # torch tensor
        message_chars = message_chars.cpu().numpy()
    elif hasattr(message_chars, 'numpy'):  # numpy array
        message_chars = message_chars
    
    # Filter out padding (0) and convert to characters
    valid_chars = message_chars[message_chars > 0]
    return ''.join(chr(code) for code in valid_chars if 32 <= code <= 126)

# --------------------------------------------------------------------------------------
# Policy & RND
# --------------------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, z_dim: int, n_actions: int, skill_dim: int = 0, hidden: int = 256):
        super().__init__()
        in_dim = z_dim + (skill_dim if skill_dim > 0 else 0)
        self.policy = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, z: torch.Tensor, skill_feat: Optional[torch.Tensor] = None):
        x = z if skill_feat is None else torch.cat([z, skill_feat], dim=-1)
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

class RNDModule(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, proj_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, proj_dim)
        )
        # freeze target
        for p in self.target.parameters():
            p.requires_grad = False
    def forward(self, z):
        with torch.no_grad():
            t = self.target(z)
        p = self.predictor(z)
        return (p - t).pow(2).sum(-1)

# --------------------------------------------------------------------------------------
# Curiosity computer
# --------------------------------------------------------------------------------------

class CuriosityComputer:
    """
    Compute (A) dynamics surprise, (B) skill entropy, and (C) optional RND.
    All terms are normalised with EMA and annealed with time.
    """
    def __init__(
        self,
        vae: MultiModalHackVAE,
        hmm: StickyHDPHMMVI | None,
        device,
        cur_cfg: CuriosityConfig,
        hmm_cfg: HMMOnlineConfig,
        rnd_cfg: RNDConfig,
        z_dim: int,
        skill_K: int,
    ):
        self.vae = vae
        self.hmm = hmm
        self.has_hmm = hmm is not None
        self.device = device
        self.cfg = cur_cfg
        self.hmm_cfg = hmm_cfg

        if not self.has_hmm:
            if cur_cfg.use_skill_entropy or cur_cfg.use_skill_transition_novelty:
                raise ValueError("Skill-based curiosity requires an HMM; disable those terms in the config when no HMM is provided.")
            if cur_cfg.use_skill_boundary_gate:
                # Boundary gate only matters when skill entropy is used, but be explicit.
                self.cfg.use_skill_boundary_gate = False

        self.norm_dyn = EMANormalizer(cur_cfg.ema_beta, cur_cfg.eps, device, center=False)
        self.norm_hdp = EMANormalizer(cur_cfg.ema_beta, cur_cfg.eps, device, center=False)
        self.norm_trans = EMANormalizer(cur_cfg.ema_beta, cur_cfg.eps, device, center=False)
        self.norm_rnd = EMANormalizer(cur_cfg.ema_beta, cur_cfg.eps, device, center=False)

        self.global_step = 0

        self.use_rnd = cur_cfg.use_rnd
        self.rnd = None
        self.rnd_opt = None
        if self.use_rnd:
            self.rnd = RNDModule(z_dim, proj_dim=rnd_cfg.proj_dim, hidden=rnd_cfg.hidden).to(device)
            self.rnd_opt = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_cfg.lr)
            self.rnd_updates_per_rollout = rnd_cfg.update_per_rollout
        if self.has_hmm:
            self.skill_K = skill_K
        else:
            # Fall back to the world-model skill dimension so we can feed zeros.
            self.skill_K = getattr(self.vae.world_model, "skill_num", 0)

    @torch.no_grad()
    def _eta(self, eta0: float, tau: float) -> float:
        return float(eta0 * math.exp(- self.global_step / max(1.0, tau)))


    @torch.no_grad()
    def compute_skill_filtered(
        self,
        mu_seq: torch.Tensor,                 # [B,T,D]
        diagvar_seq: torch.Tensor,            # [B,T,D]
        F_seq: Optional[torch.Tensor],        # [B,T,D,R] or None
        mask: torch.Tensor,                   # [B,T]  (1=valid)
        dones: Optional[torch.Tensor] = None  # [B,T] episode termination flags
    ) -> Dict[str, torch.Tensor]:
        """
        Causal per-step HMM filtering using only z_{1:t}.
        Returns:
            alpha : [B,T,Kp1]   filtered marginals α_t
            xi    : [B,T-1,Kp1,Kp1] causal pair posteriors ξ_{t-1,t} (uses z_t)
            H     : [B,T]       entropy H(α_t)
            p_change : [B,T]    boundary prob 1 - ∑_k ξ_{t-1,t}(k,k); first step NaN
        """
        if not self.has_hmm:
            raise RuntimeError("compute_skill_filtered called without an HMM present")
        B, T, _ = mu_seq.shape
        ElogA = self.hmm._ElogA()
        Kp1 = self.hmm.niw.mu.size(0)
        alpha = torch.zeros(B, T, Kp1, device=self.device, dtype=mu_seq.dtype)
        xi = torch.zeros(B, max(T-1,0), Kp1, Kp1, device=self.device, dtype=mu_seq.dtype)
        H = torch.zeros(B, T, device=self.device, dtype=mu_seq.dtype)
        pchg = torch.zeros(B, T, device=self.device, dtype=mu_seq.dtype)
        
        logB = self.hmm.make_logB_for_filter(
                    mu_seq, diagvar_seq, F_seq, None, self.hmm_cfg.emission_mode, self.hmm_cfg.student_t_use_sample
                )

        for b in range(B):
            st = None
            prev_valid = False
            for t in range(T):
                # Check if we need to reset due to episode boundary first
                if dones is not None and t > 0 and dones[b, t-1]:
                    # Previous step was terminal, start new HMM chain
                    st = None
                    prev_valid = False
                
                if mask is not None and not bool(mask[b, t].item()):
                    # Invalid observation: transition through without emission update
                    if st is not None and prev_valid:
                        # Apply transition-only step to maintain chain continuity
                        # Use prior transition probabilities to update state
                        log_alpha_pred = st.log_alpha.unsqueeze(1) + ElogA  # [Kp1, Kp1]
                        st.log_alpha = torch.logsumexp(log_alpha_pred, dim=0)  # [Kp1]
                        # Set outputs to NaN/zero for invalid timesteps
                        alpha[b, t] = torch.full((Kp1,), float('nan'), device=self.device)
                        H[b, t] = float('nan')
                        pchg[b, t] = float('nan')
                    else:
                        # No previous state to transition from
                        alpha[b, t] = torch.full((Kp1,), float('nan'), device=self.device)
                        H[b, t] = float('nan')
                        pchg[b, t] = float('nan')
                    continue
                
                logB_t = logB[b, t]  # [Kp1]
                
                # Diagnostic: Check for extreme logB_t values that might cause uniform posterior
                if t % 1000 == 0 and b == 0:  # Log occasionally
                    logB_range = logB_t.max() - logB_t.min()
                    unused_skills = (self.hmm.niw.nu <= 106.1).float().sum().item()  # Count near-prior skills
                    used_skills = (self.hmm.niw.nu > 106.1).float().sum().item()
                    print(f"🔍 logB_t at t={t}: range={logB_range:.2f}, "
                          f"unused_skills={unused_skills}, used_skills={used_skills}")
                    print(f"   logB_t[unused] ≈ {logB_t[self.hmm.niw.nu <= 106.1].mean():.2f}, "
                          f"logB_t[used] ≈ {logB_t[self.hmm.niw.nu > 106.1].mean():.2f}")
                
                if not prev_valid or st is None:
                    # Initialize new HMM chain (start of episode or after invalid step)
                    st = self.hmm.filter_init_from_logB(logB_t)
                    a_t = torch.exp(st.log_alpha.to(torch.float32))
                    alpha[b, t] = a_t
                    H[b, t] = (-(a_t.clamp_min(1e-12).log() * a_t).sum())
                    pchg[b, t] = float('nan')
                    prev_valid = True
                else:
                    # Continue existing HMM chain
                    st, a_t, xi_t, boundary_prob, H_t = self.hmm.filter_step(st, logB_t, ElogA)
                    alpha[b, t] = a_t
                    if t > 0:  # Only store xi if we have a previous timestep
                        xi[b, t-1] = xi_t
                    H[b, t] = H_t
                    pchg[b, t] = boundary_prob
                    
        return {"alpha": alpha, "xi": xi, "H": H, "p_change": pchg}


    def compute_intrinsic(
        self,
        mu: torch.Tensor, logvar: torch.Tensor, F: Optional[torch.Tensor],       # [B,T,D], [B,T,D], [B,T,D,R] or None
        actions: torch.Tensor,                                                   # [B,T] (nle codes)
        mask: torch.Tensor,                                                      # [B,T]
        dones: Optional[torch.Tensor] = None,                                    # [B,T] episode termination flags
        next_obs_mu: Optional[torch.Tensor] = None,                              # [B,D] next observation mu for final timestep dynamics
        next_obs_logvar: Optional[torch.Tensor] = None,                          # [B,D] next observation logvar for final timestep dynamics
        next_obs_F: Optional[torch.Tensor] = None                                # [B,D,R] next observation F for final timestep dynamics
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with per-step intrinsic bonuses (aligned to timesteps), and per-term summaries.
        """
        device = mu.device
        B, T, D = mu.shape
        A = ACTION_DIM

        # Create episode-aware mask for HMM filtering
        if dones is not None:
            # Build a mask that resets HMM chains at episode boundaries
            hmm_mask = torch.ones_like(mask)
            for b in range(B):
                for t in range(T):
                    if t > 0 and dones[b, t-1]:
                        # This is the first step of a new episode, so reset is needed
                        # The HMM filtering will handle this by checking prev_valid
                        pass
                    hmm_mask[b, t] = mask[b, t]
        else:
            hmm_mask = mask

        h_boundary_gated = torch.zeros(B, T, device=device, dtype=mu.dtype)

        if self.has_hmm:
            # ---- HMM responsibilities rhat and (B) skill entropy ----
            filted = self.compute_skill_filtered(mu, logvar.exp().clamp_min(1e-6), F, hmm_mask, dones)  # [B,T,Kp1]
            alpha = filted["alpha"]  # [B,T,Kp1]
            xi = filted["xi"]      # [B,T-1,Kp1,Kp1]
            h_entropy = filted["H"]  # [B,T]
            Kp1 = alpha.shape[-1]
            skill_soft = alpha[...,:(Kp1-1)]

            # ---- (B) Skill-boundary entropy gated + (C) Skill-transition novelty ----
            with torch.no_grad():
                ElogA = self.hmm._ElogA()  # [Kp1,Kp1]
                
                # Compute boundary gate: 1{ΔH_t > 0} with episode boundary awareness
                gate_bool = torch.zeros(B, T, device=device)
                for b in range(B):
                    prev_H = None
                    for t in range(T):
                        if mask[b, t] < 0.5:
                            prev_H = None
                            continue
                        # Reset at episode boundaries
                        if dones is not None and t > 0 and dones[b, t-1]:
                            prev_H = None
                        
                        if prev_H is None:
                            gate_bool[b, t] = 0.0  # no previous step to compare to
                        else:
                            dH = h_entropy[b, t] - prev_H
                            gate_bool[b, t] = 1.0 if dH > self.cfg.gate_delta_eps else 0.0
                        prev_H = float(h_entropy[b, t].item())

                # Compute transition novelty using already computed xi
                trans_novel = torch.zeros(B, T, device=device)
                # xi is [B, T-1, Kp1, Kp1] - pairwise posteriors ξ_{t-1,t}
                neg_logA = (-ElogA).clamp_min(0.0)            # [Kp1,Kp1]
                neg_logA.fill_diagonal_(0.0)                  # ignore self-transitions
                for t in range(1, T):  # start from t=1 since xi[t-1] exists
                    # Don't compute transition novelty across episode boundaries
                    for b in range(B):
                        if dones is not None and dones[b, t-1]:
                            # Episode terminated at t-1, so no transition to t
                            trans_novel[b, t] = 0.0
                        else:
                            trans_novel[b, t] = (xi[b, t-1] * neg_logA).sum()

                # gated boundary entropy: H(α_t) * 1{ΔH_t > 0}
                h_boundary_gated = (h_entropy * gate_bool) * mask  # [B,T]
                trans_novel = trans_novel * mask                # [B,T]
        else:
            skill_soft = torch.zeros(B, T, self.skill_K, device=device, dtype=mu.dtype)
            h_entropy = torch.zeros(B, T, device=device, dtype=mu.dtype)
            gate_bool = torch.zeros(B, T, device=device, dtype=mu.dtype)
            trans_novel = torch.zeros(B, T, device=device, dtype=mu.dtype)
            h_boundary_gated = h_boundary_gated * 0.0

        # ---- (A) dynamics KL surprise using world-model prior ----
        dyn = torch.zeros(B, T, device=device)
        if self.cfg.use_dyn_kl and self.vae.world_model.enabled:
            # Prepare action one-hot and skill features; compute priors for t+1
            a_onehot = one_hot(actions.clamp(0, A-1), A)  # [B,T,A]
            # initial world state zero for each batch element
            s = self.vae.world_model.initial_state(B, device=device)
            mu_p_list, logvar_p_list, Fp_list = [], [], []
            
            # Compute predictions for t=0 to t=T-1 (need T predictions for T timesteps)
            num_predictions = T if (next_obs_mu is not None) else T-1
            for t in range(num_predictions):
                s, mu_p, logvar_p, F_p = self.vae.world_model_step(
                    s, mu[:,t,:], a_onehot[:,t,:], skill_soft[:,t,:]
                )
                mu_p_list.append(mu_p)
                logvar_p_list.append(logvar_p)
                Fp_list.append(F_p)
            
            mu_p = torch.stack(mu_p_list, dim=1)                     # [B,num_predictions,D]
            logvar_p = torch.stack(logvar_p_list, dim=1)             # [B,num_predictions,D]
            F_p = torch.stack(Fp_list, dim=1) if Fp_list[0] is not None else None  # [B,num_predictions,D,R] or None

            # Prepare actual next observations for KL computation
            if next_obs_mu is not None:
                # Include final timestep using provided next observation
                mu_q = torch.cat([mu[:,1:,:], next_obs_mu.unsqueeze(1)], dim=1)     # [B,T,D]
                logvar_q = torch.cat([logvar[:,1:,:], next_obs_logvar.unsqueeze(1)], dim=1)  # [B,T,D]
                if F is not None and next_obs_F is not None:
                    F_q = torch.cat([F[:,1:,:,:], next_obs_F.unsqueeze(1)], dim=1)  # [B,T,D,R]
                else:
                    F_q = None
                kl_timesteps = T  # All timesteps have dynamics reward
            else:
                # Original behavior - only t=1 to t=T-1
                mu_q = mu[:,1:,:]; logvar_q = logvar[:,1:,:]
                F_q = F[:,1:,:,:] if F is not None else None
                kl_timesteps = T-1  # Only T-1 timesteps have dynamics reward
            
            # Compute KL divergences
            kl = kl_gaussian_lowrank_q_p(
                mu_q=mu_q.reshape(-1, D), logvar_q=logvar_q.reshape(-1, D), 
                F_q=None if F_q is None else F_q.reshape(-1, D, F_q.size(-1)),
                mu_p=mu_p.reshape(-1, D), logvar_p=logvar_p.reshape(-1, D), 
                F_p=None if F_p is None else F_p.reshape(-1, D, F_p.size(-1))
            ).view(B, kl_timesteps)
            
            # Assign KL values to appropriate timesteps
            if next_obs_mu is not None:
                dyn = kl  # [B, T] - all timesteps covered
            else:
                dyn[:,:-1] = kl  # [B, T-1] - last timestep remains zero
            dyn = dyn * mask

        # ---- RND novelty on z_t ----
        rnd = torch.zeros(B, T, device=device)
        if self.use_rnd:
            with torch.no_grad():
                rnd = self.rnd(mu.reshape(B*T, D)).view(B, T)
                rnd = rnd * mask

        # ---- Normalise and anneal ----
        out = {}
        if self.cfg.use_dyn_kl:
            dyn_n = self.norm_dyn.update(dyn[mask.bool()])
            dyn_scaled = self._eta(self.cfg.eta0_dyn, self.cfg.tau_dyn) * dyn
            dyn_scaled[mask.bool()] = self._eta(self.cfg.eta0_dyn, self.cfg.tau_dyn) * dyn_n
            out["dyn_raw"] = dyn; out["dyn"] = dyn_scaled
        else:
            out["dyn_raw"] = dyn; out["dyn"] = torch.zeros_like(dyn)

        # (B) boundary‑gated skill entropy
        if self.cfg.use_skill_entropy:
            eta_hdp = self._eta(self.cfg.eta0_hdp, self.cfg.tau_hdp)
            # normalise each sub‑term on valid steps, then sum
            ent_n = self.norm_hdp.update(h_boundary_gated[mask.bool()])
            ent_scaled = eta_hdp * h_boundary_gated
            ent_scaled[mask.bool()] = eta_hdp * ent_n
            out["hdp_raw"] = h_entropy * mask    # plain entropy for diagnostics
            out["hdp"] = ent_scaled
        else:
            out["hdp_raw"] = h_entropy * mask    # plain entropy for diagnostics
            out["hdp"] = torch.zeros_like(h_entropy)
            
        # (C) Skill-transition novelty
        if self.cfg.use_skill_transition_novelty:
            trans_n = self.norm_trans.update(trans_novel[mask.bool()])
            eta_trans = self._eta(self.cfg.eta0_stn, self.cfg.tau_stn)
            trans_scaled = eta_trans * trans_novel
            trans_scaled[mask.bool()] = eta_trans * trans_n
            out["trans_raw"]= trans_novel
            out["trans"]    = trans_scaled
        else:
            out["trans_raw"] = trans_novel
            out["trans"] = torch.zeros_like(trans_novel)

        if self.use_rnd:
            rnd_n = self.norm_rnd.update(rnd[mask.bool()])
            rnd_scaled = self._eta(self.cfg.eta0_rnd, self.cfg.tau_rnd) * rnd
            rnd_scaled[mask.bool()] = self._eta(self.cfg.eta0_rnd, self.cfg.tau_rnd) * rnd_n
            out["rnd_raw"] = rnd; out["rnd"] = rnd_scaled
        else:
            out["rnd_raw"] = rnd; out["rnd"] = torch.zeros_like(rnd)

        out["rhat_skill"] = skill_soft  # [B,T,K]
        return out

    def train_rnd(self, mu: torch.Tensor):
        if not self.use_rnd: return
        z = mu.detach()
        loss = self.rnd(z).mean()
        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()

# --------------------------------------------------------------------------------------
# Replay buffer (on-policy rollouts for PPO)
# --------------------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, num_envs, T, z_dim, skill_dim, device):
        self.num_envs = num_envs; self.T = T; self.device = device; self.z_dim = z_dim
        self.ptr = 0
        self.z      = torch.zeros(T, num_envs, z_dim, device=device)
        self.mu     = torch.zeros(T, num_envs, z_dim, device=device)
        self.logvar = torch.zeros(T, num_envs, z_dim, device=device)
        self.lowrank_factors = None  # low-rank optional - allocated dynamically when first F is added
        self.actions= torch.zeros(T, num_envs, dtype=torch.long, device=device)
        self.rews_e = torch.zeros(T, num_envs, device=device)  # extrinsic
        self.dones  = torch.zeros(T, num_envs, dtype=torch.bool, device=device)
        self.mask   = torch.zeros(T, num_envs, device=device)  # validity mask
        self.val    = torch.zeros(T, num_envs, device=device)
        self.logp   = torch.zeros(T, num_envs, device=device)
        self.skill  = torch.zeros(T, num_envs, skill_dim, device=device) if skill_dim>0 else None
        
        # Store raw observations for full VAE training
        self.obs_chars = None    # [T, num_envs, 21, 79] - allocated when first obs is added
        self.obs_colors = None   # [T, num_envs, 21, 79] - allocated when first obs is added  
        self.obs_blstats = None  # [T, num_envs, blstats_dim] - allocated when first obs is added
        self.obs_message = None  # [T, num_envs, 256] - allocated when first obs is added
        self.obs_hero_info = None  # [T, num_envs, 4] - allocated when first obs is added
        
        # Store last timestep from previous rollout for continuity
        self.prev_mu = torch.zeros(num_envs, z_dim, device=device)
        self.prev_logvar = torch.zeros(num_envs, z_dim, device=device)
        self.prev_lowrank_factors = None
        self.prev_actions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_mask = torch.zeros(num_envs, device=device)
        self.prev_dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Store previous observations too
        self.prev_obs_chars = None
        self.prev_obs_colors = None  
        self.prev_obs_blstats = None
        self.prev_obs_message = None
        self.prev_obs_hero_info = None
        self.has_prev = False  # Flag to indicate if we have valid previous data

    def _maybe_allocate_lowrank_factors(self, lowrank_factors_tensor):
        """Allocate lowrank_factors buffer when first lowrank_factors tensor is provided."""
        if self.lowrank_factors is None and lowrank_factors_tensor is not None:
            # Get dimensions from the provided tensor
            D, R = lowrank_factors_tensor.shape[-2], lowrank_factors_tensor.shape[-1]
            self.lowrank_factors = torch.zeros(self.T, self.num_envs, D, R, device=self.device)
            # Also allocate prev_lowrank_factors buffer
            self.prev_lowrank_factors = torch.zeros(self.num_envs, D, R, device=self.device)
            return True
        return False
        
    def _maybe_allocate_obs_buffers(self, obs_batch):
        """Allocate observation buffers when first observation batch is provided."""
        if self.obs_chars is None and 'game_chars' in obs_batch:
            chars_shape = obs_batch['game_chars'].shape[1:]  # Remove batch dim
            colors_shape = obs_batch['game_colors'].shape[1:]
            blstats_shape = obs_batch['blstats'].shape[1:]
            message_shape = obs_batch['message_chars'].shape[1:]
            hero_info_shape = obs_batch['hero_info'].shape[1:]
            
            # Allocate current rollout buffers
            self.obs_chars = torch.zeros(self.T, self.num_envs, *chars_shape, device=self.device, dtype=torch.long)
            self.obs_colors = torch.zeros(self.T, self.num_envs, *colors_shape, device=self.device, dtype=torch.long)
            self.obs_blstats = torch.zeros(self.T, self.num_envs, *blstats_shape, device=self.device, dtype=torch.long)
            self.obs_message = torch.zeros(self.T, self.num_envs, *message_shape, device=self.device, dtype=torch.long)
            self.obs_hero_info = torch.zeros(self.T, self.num_envs, *hero_info_shape, device=self.device, dtype=torch.long)
            
            # Allocate previous timestep buffers
            self.prev_obs_chars = torch.zeros(self.num_envs, *chars_shape, device=self.device, dtype=torch.long)
            self.prev_obs_colors = torch.zeros(self.num_envs, *colors_shape, device=self.device, dtype=torch.long)
            self.prev_obs_blstats = torch.zeros(self.num_envs, *blstats_shape, device=self.device, dtype=torch.long)
            self.prev_obs_message = torch.zeros(self.num_envs, *message_shape, device=self.device, dtype=torch.long)
            self.prev_obs_hero_info = torch.zeros(self.num_envs, *hero_info_shape, device=self.device, dtype=torch.long)
            return True
        return False
        
    def add(self, **kw):
        t = self.ptr

        # Special handling for lowrank_factors - allocate buffer if needed
        if 'lowrank_factors' in kw and kw['lowrank_factors'] is not None:
            self._maybe_allocate_lowrank_factors(kw['lowrank_factors'])
            
        # Special handling for observations - allocate buffers if needed
        if 'obs_batch' in kw and kw['obs_batch'] is not None:
            self._maybe_allocate_obs_buffers(kw['obs_batch'])
            # Store observation components
            obs_batch = kw['obs_batch']
            if self.obs_chars is not None:
                self.obs_chars[t].copy_(obs_batch['game_chars'])
                self.obs_colors[t].copy_(obs_batch['game_colors'])
                self.obs_blstats[t].copy_(obs_batch['blstats'])
                self.obs_message[t].copy_(obs_batch['message_chars'])
                self.obs_hero_info[t].copy_(obs_batch['hero_info'])

        for k,v in kw.items():
            if k == 'obs_batch':  # Skip obs_batch as it's handled above
                continue
            buffer = getattr(self, k)
            if buffer is None: 
                continue
            buffer[t].copy_(v)
        self.ptr += 1
        
    def full(self): return self.ptr >= self.T
    
    def reset(self):
        """Reset buffer for new rollout, but preserve last timestep as prev_* for continuity."""
        if self.ptr > 0:  # Only store if we have collected some data
            # Store last timestep data for next rollout's continuity
            self.prev_mu.copy_(self.mu[self.ptr - 1])
            self.prev_logvar.copy_(self.logvar[self.ptr - 1])
            if self.lowrank_factors is not None and self.prev_lowrank_factors is not None:
                self.prev_lowrank_factors.copy_(self.lowrank_factors[self.ptr - 1])
            self.prev_actions.copy_(self.actions[self.ptr - 1])
            self.prev_mask.copy_(self.mask[self.ptr - 1])
            self.prev_dones.copy_(self.dones[self.ptr - 1])
            
            # Store previous observations for continuity
            if self.obs_chars is not None and self.prev_obs_chars is not None:
                self.prev_obs_chars.copy_(self.obs_chars[self.ptr - 1])
                self.prev_obs_colors.copy_(self.obs_colors[self.ptr - 1])
                self.prev_obs_blstats.copy_(self.obs_blstats[self.ptr - 1])
                self.prev_obs_message.copy_(self.obs_message[self.ptr - 1])
                self.prev_obs_hero_info.copy_(self.obs_hero_info[self.ptr - 1])
            
            self.has_prev = True
        self.ptr = 0
        
    def get_extended_data_for_transitions(self):
        """Get data extended with previous timestep for transition computations."""
        if not self.has_prev:
            # First rollout - no previous data available
            return {
                "mu": self.mu,
                "logvar": self.logvar,
                "lowrank_factors": self.lowrank_factors,
                "actions": self.actions,
                "mask": self.mask,
                "dones": self.dones,
                "has_prev": False
            }
        
        # Extend current rollout with previous timestep
        # Shape: [T+1, B, ...] where index 0 is previous timestep, 1:T+1 is current rollout
        extended_mu = torch.cat([self.prev_mu.unsqueeze(0), self.mu], dim=0)
        extended_logvar = torch.cat([self.prev_logvar.unsqueeze(0), self.logvar], dim=0)
        extended_actions = torch.cat([self.prev_actions.unsqueeze(0), self.actions], dim=0)
        extended_mask = torch.cat([self.prev_mask.unsqueeze(0), self.mask], dim=0)
        extended_dones = torch.cat([self.prev_dones.unsqueeze(0), self.dones], dim=0)
        
        extended_lowrank_factors = None
        if self.lowrank_factors is not None and self.prev_lowrank_factors is not None:
            extended_lowrank_factors = torch.cat([self.prev_lowrank_factors.unsqueeze(0), self.lowrank_factors], dim=0)
        
        return {
            "mu": extended_mu,
            "logvar": extended_logvar,
            "lowrank_factors": extended_lowrank_factors,
            "actions": extended_actions,
            "mask": extended_mask,
            "dones": extended_dones,
            "has_prev": True
        }
    def get(self):
        # flatten T*B
        T,B = self.T, self.num_envs
        data = { "mu": self.mu, "logvar": self.logvar, "actions": self.actions, "extrinsic": self.rews_e, "dones": self.dones, "mask": self.mask, "values": self.val, "logp": self.logp }
        if self.skill is not None: data["skill"] = self.skill
        if self.lowrank_factors is not None: data["lowrank_factors"] = self.lowrank_factors
        for k in ["mu","logvar","actions","extrinsic","dones","mask","values","logp","skill","lowrank_factors"]:
            if k in data: data[k] = data[k].reshape(T*B, *data[k].shape[2:])
        return data

# --------------------------------------------------------------------------------------
# PPO trainer
# --------------------------------------------------------------------------------------

class PPOTrainer:
    def __init__(self, env_id: str, ppo_cfg: PPOConfig, cur_cfg: CuriosityConfig, hmm_cfg: HMMOnlineConfig, vae_cfg: VAEOnlineConfig, rnd_cfg: RNDConfig, run_cfg: TrainConfig,
                 vae: MultiModalHackVAE, hmm: StickyHDPHMMVI, wandb_run=None):
        self.env_id = env_id
        self.ppo_cfg = ppo_cfg
        self.cur_cfg = cur_cfg
        self.hmm_cfg = hmm_cfg
        self.vae_cfg = vae_cfg
        self.rnd_cfg = rnd_cfg
        self.run_cfg = run_cfg
        self.device = run_cfg.device
        self.wandb_run = wandb_run

        # Vec env
        def make_env():
            return gym.make(env_id)
        self.envs = gym.vector.SyncVectorEnv([make_env for _ in range(ppo_cfg.num_envs)])
        obs_space = self.envs.single_observation_space
        # Use the FULL NLE action set so one policy transfers across MiniHack tasks
        self.n_actions = ACTION_DIM
        # Build per-env mask and global->local index mapping
        self._init_action_adapter()

        # Models
        self.vae = vae.to(self.device).eval()  # encoder used in no-grad mode during rollouts
        self.hmm = hmm
        self.has_hmm = hmm is not None

        # Policy
        z_dim = vae.latent_dim
        if self.has_hmm and ppo_cfg.policy_uses_skill:
            skill_dim = hmm.niw.mu.size(0) - 1  # exclude remainder
        else:
            skill_dim = 0
        self.actor_critic = ActorCritic(z_dim, self.n_actions, skill_dim=skill_dim).to(self.device)
        self.opt = torch.optim.Adam(self.actor_critic.parameters(), lr=ppo_cfg.learning_rate)

        # Curiosity computer
        curiosity_skill_dim = (hmm.niw.mu.size(0) - 1) if self.has_hmm else getattr(self.vae.world_model, "skill_num", 0)
        self.curiosity = CuriosityComputer(self.vae, self.hmm, self.device, cur_cfg, hmm_cfg, rnd_cfg, z_dim, curiosity_skill_dim)

        # Storage
        self.buf = RolloutBuffer(ppo_cfg.num_envs, ppo_cfg.rollout_len, z_dim, skill_dim, self.device)

        # Bookkeeping
        self.global_steps = 0
        self.last_hmm_refresh = 0
        self._hmm_interval = self.hmm_cfg.hmm_update_every if self.has_hmm else float('inf')
        self.last_vae_refresh = 0
        self._vae_interval = self.vae_cfg.vae_update_every  # Track VAE update interval
        self.vae_opt = None
        # Enable full VAE training instead of just world model
        if self.vae is not None:
            # Unfreeze all VAE parameters for full training
            for p in self.vae.parameters():
                p.requires_grad_(True)
            self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.vae_cfg.vae_lr)
        os.makedirs(run_cfg.log_dir, exist_ok=True)
        self._log_file = os.path.join(run_cfg.log_dir, "metrics.jsonl")
        # --- HMM filtering state (per-env) and cached E[log A] ---
        self._filt_state = [None for _ in range(ppo_cfg.num_envs)] if (self.has_hmm and self.ppo_cfg.policy_uses_skill) else None
        # cache ElogA; refresh whenever HMM is updated
        self._ElogA = self.hmm._ElogA() if self.has_hmm else None
        
        # --- World model state for continuity across rollouts ---
        self._world_model_state = None  # Store final world model state from previous rollout
        
        # --- Hero info tracking for each environment ---
        self.data_collector = NetHackDataCollector()
        self._hero_info = [None for _ in range(ppo_cfg.num_envs)]  # Current hero info per env [role, race, gender, alignment]
        self._episode_start = [True for _ in range(ppo_cfg.num_envs)]  # Track if we need to parse hero info

        # --- Replay buffers for model updates ---
        self.replay_mu = []
        self.replay_logvar = []
        self.replay_lowrank_factors = []
        self.replay_actions = []
        self.replay_obs_chars = []
        self.replay_obs_colors = []
        self.replay_obs_blstats = []
        self.replay_obs_message = []
        self.replay_obs_hero_info = []
        self.replay_rewards = []
        self.replay_dones = []

    # --------------------------- rollout -------------------------------------

    @torch.no_grad()
    def _encode_obs(self, obs_dict: dict, hero_info_list: list) -> Dict[str, torch.Tensor]:
        # Stack current hero info for all environments
        obs_shape = obs_dict['chars'].shape
        if len(obs_shape) == 2:
            # Single env case: obs_dict values are [H,W] - wrap in batch dim
            for k in obs_dict:
                if hasattr(obs_dict[k], 'unsqueeze'):
                    # PyTorch tensor
                    obs_dict[k] = obs_dict[k].unsqueeze(0)  # [1,H,W] or [1,256] etc
                else:
                    # Numpy array
                    obs_dict[k] = np.expand_dims(obs_dict[k], axis=0)  # [1,H,W] or [1,256] etc
        assert len(hero_info_list) == obs_dict[list(obs_dict.keys())[0]].shape[0], "hero_info_list length must match number of envs"
        hero_info_batch = torch.stack([
            hero_info if hero_info is not None else torch.zeros(4, dtype=torch.int32)
            for hero_info in hero_info_list
        ], dim=0)  # [num_envs, 4]
        
        b = obs_to_device(obs_dict, self.device, hero_info=hero_info_batch)
        enc = self.vae.encode(b["game_chars"], b["game_colors"], b["blstats"], b["message_chars"], b["hero_info"])
        mu = enc["mu"]; logvar = enc["logvar"]; F = enc["lowrank_factors"]  # tensors [B,D], [B,D], [B,D,R] or None
        z  = mu  # use mean latent for policy input
        return {"z": z, "mu": mu, "logvar": logvar, "F": F}

    def _update_hero_info_from_obs(self, obs_dict: dict, num_envs: int, episode_start_flags: list, hero_info_list: list, logger: Optional[logging.Logger] = None):
        """Extract hero info from the first observation of each new episode."""
        assert len(episode_start_flags) == num_envs, "episode_start_flags length must match num_envs"
        assert len(hero_info_list) == num_envs, "hero_info_list length must match num_envs"
        
        obs_shape = obs_dict['chars'].shape
        if len(obs_shape) == 2:
            # Single env case: obs_dict values are [H,W] - wrap in batch dim
            for k in obs_dict:
                if hasattr(obs_dict[k], 'unsqueeze'):
                    # PyTorch tensor
                    obs_dict[k] = obs_dict[k].unsqueeze(0)  # [1,H,W] or [1,256] etc
                else:
                    # Numpy array
                    obs_dict[k] = np.expand_dims(obs_dict[k], axis=0)  # [1,H,W] or [1,256] etc
        
        assert len(obs_dict[list(obs_dict.keys())[0]]) == num_envs, "obs_dict batch size must match num_envs"
        
        for env_idx in range(num_envs):
            if episode_start_flags[env_idx]:
                try:
                    # Extract message from observation for this environment
                    message_key = "message" if "message" in obs_dict else None
                    if message_key and hasattr(obs_dict[message_key], 'shape') and len(obs_dict[message_key].shape) > 1:
                        # Vectorized env case: obs_dict[message_key] is [num_envs, 256]
                        message_ascii = obs_dict[message_key][env_idx]
                    elif message_key and env_idx == 0:
                        # Single env case: obs_dict[message_key] is [256]
                        message_ascii = obs_dict[message_key]
                    else:
                        # No message found
                        continue
                    
                    # Convert ASCII codes to string
                    message_str = message_ascii_to_string(message_ascii)
                    
                    # Parse hero info using data collector
                    # Use a unique game_id per environment (could be improved with actual game tracking)
                    game_id = env_idx  # Simple approach: use env index as game_id
                    hero_info = self.data_collector.parse_hero_info_from_message(game_id, message_str)
                    
                    if hero_info is not None:
                        hero_info_list[env_idx] = hero_info
                        #print(f"✅ Environment {env_idx}: Parsed hero info {hero_info.tolist()} from message: '{message_str[:100]}...'")
                    else:
                        # Keep previous hero info or use zeros as fallback
                        if hero_info_list[env_idx] is None:
                            hero_info_list[env_idx] = torch.zeros(4, dtype=torch.int32)
                            if logger: logger.warning(f"⚠️ Environment {env_idx}: Could not parse hero info from message: '{message_str[:100]}...'")
                    
                except Exception as e:
                    # Fallback to zeros if anything goes wrong
                    if hero_info_list[env_idx] is None:
                        hero_info_list[env_idx] = torch.zeros(4, dtype=torch.int32)
                    if logger: logger.warning(f"⚠️ Environment {env_idx}: Error parsing hero info: {e}")

                # Mark that we've processed the episode start
                episode_start_flags[env_idx] = False

    def _init_action_adapter(self):
        """
        Build (a) valid-action mask over the global NLE action space and
        (b) a mapping from global action id -> local env index expected by each env.
        Assumes all vectorized envs are the same MiniHack task (usual case).
        """
        G = ACTION_DIM
        # Grab allowed actions list from the first underlying env
        base_env = self.envs.envs[0]
        allowed = getattr(base_env.unwrapped, "actions", None)
        if allowed is None:
            allowed = list(range(G))  # full action set
        allowed = [int(a) for a in allowed]

        # mask over global ids
        mask = torch.zeros(G, dtype=torch.bool, device=self.device)
        allowed_indices = [KEYPRESS_INDEX_MAPPING[k] for k in allowed]
        mask[allowed_indices] = True
        # global->local index (for stepping env)
        g2l = torch.full((G,), -1, dtype=torch.long, device=self.device)
        for li, gid in enumerate(allowed_indices):
            g2l[gid] = li
        # Broadcast to all env slots
        B = self.ppo_cfg.num_envs
        self.action_mask = mask.view(1, G).expand(B, G).contiguous()      # [B,G] bool
        self.global2local = g2l.view(1, G).expand(B, G).contiguous()      # [B,G] long

    def _masked_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the per-env action mask to logits.
        logits: [B,G]
        """
        # Extremely negative for invalid actions to remove them from Categorical
        return logits.masked_fill(~self.action_mask, -1e9)

    @torch.no_grad()
    def collect_rollout(self, logger: Optional[logging.Logger] = None):
        self.buf.reset()
        # Continue from previous observation state (set in train() method)
        obs = self._obs

        # Restore world model state from previous rollout or initialize
        if self._world_model_state is not None and self.vae.world_model.enabled:
            s_wm = self._world_model_state
        else:
            s_wm = self.vae.world_model.initial_state(self.ppo_cfg.num_envs, device=self.device) if self.vae.world_model.enabled else None

        # Track episode statistics for logging
        episode_returns = [0.0] * self.ppo_cfg.num_envs
        episode_lengths = [0] * self.ppo_cfg.num_envs

        for t in range(self.ppo_cfg.rollout_len):
            enc = self._encode_obs(obs, self._hero_info)  # dict of tensors [B,D], etc
            z = enc["z"]  # [B,D]

            skill_feat = self._compute_skill_features(enc)  # [B,K] or None

            logits, value = self.actor_critic(z, skill_feat)        # logits: [B,G]
            masked = self._masked_logits(logits)
            dist = torch.distributions.Categorical(logits=masked)
            a_global = dist.sample()               # [B] global ids in NLE space
            logp = dist.log_prob(a_global)

            # Map to per-env local indices for stepping vectorized env
            a_local = self.global2local.gather(1, a_global.view(-1,1)).squeeze(1)  # [B]
            # Safety: if something is -1 (shouldn't happen with mask), map to 0
            a_local = torch.where(a_local < 0, torch.zeros_like(a_local), a_local)

            # step
            next_obs, rew, terminated, truncated, info = self.envs.step(a_local.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # Update episode statistics and log completions
            for env_idx in range(self.ppo_cfg.num_envs):
                episode_returns[env_idx] += rew[env_idx]
                episode_lengths[env_idx] += 1
                
                if done[env_idx]:
                    # Log episode completion
                    #if logger is not None:
                        #logger.info(f"🎮 Episode completed (env {env_idx}): return={episode_returns[env_idx]:.2f}, length={episode_lengths[env_idx]}")
                        
                    # Extract game message if available
                    message = ""
                    if hasattr(next_obs, '__getitem__') and 'message' in next_obs:
                        try:
                            message = message_ascii_to_string(next_obs['message'][env_idx])
                        except:
                            message = ""
                    if logger is not None and len(message) > 0:
                        logger.info(f"💬 Final game message (env {env_idx}): {message[:200]}{'...' if len(message)>200 else ''}")
                    
                    # Reset episode tracking for this environment
                    episode_returns[env_idx] = 0.0
                    episode_lengths[env_idx] = 0
            
            # Create mask: all collected steps are valid (no padding in rollout)
            step_mask = torch.ones(self.ppo_cfg.num_envs, dtype=torch.float32, device=self.device)
            
            # Prepare observation batch for storage
            hero_info_batch = torch.stack([
                hero_info if hero_info is not None else torch.zeros(4, dtype=torch.int32)
                for hero_info in self._hero_info
            ], dim=0)  # [num_envs, 4]
            obs_batch = obs_to_device(obs, self.device, hero_info=hero_info_batch)
            
            # store
            self.buf.add(z=z, mu=enc["mu"], logvar=enc["logvar"], lowrank_factors=enc.get('lowrank_factors', None),
                         actions=torch.as_tensor(a_global, device=self.device),
                         rews_e=torch.as_tensor(rew, dtype=torch.float32, device=self.device),
                         dones=torch.as_tensor(done, dtype=torch.bool, device=self.device),
                         mask=step_mask,
                         val=value, logp=logp,
                         skill=skill_feat if (self.buf.skill is not None) else None,
                         obs_batch=obs_batch)
            # reset filter state for envs that terminated; next loop will re-init on new obs
            if self.ppo_cfg.policy_uses_skill:
                for b, d in enumerate(done):
                    if d:
                        self._filt_state[b] = None
            
            # Mark episode start for environments that terminated (for hero info parsing)
            for b, d in enumerate(done):
                if d:
                    self._episode_start[b] = True

            obs = next_obs
            
            # Update hero info for any new episodes that started
            self._update_hero_info_from_obs(obs, self.ppo_cfg.num_envs, self._episode_start, self._hero_info, logger=logger)
            
            self.global_steps += self.ppo_cfg.num_envs
        
        # Store final world model state for next rollout's continuity
        if s_wm is not None:
            self._world_model_state = s_wm
        
        self._obs = obs
        
        # Append rollout data to replay buffers for model updates
        with torch.no_grad():
            # Get observations and latents from current rollout
            mu_bt = self.buf.mu.transpose(0,1)          # [B,T,D]
            logvar_bt = self.buf.logvar.transpose(0,1)  # [B,T,D]
            lowrank_bt = self.buf.lowrank_factors.transpose(0,1) if self.buf.lowrank_factors is not None else None  # [B,T,D,R] or None
            acts_bt = self.buf.actions.transpose(0,1)   # [B,T]
            
            obs_chars_bt = self.buf.obs_chars.transpose(0,1) if self.buf.obs_chars is not None else None      # [B,T,21,79]
            obs_colors_bt = self.buf.obs_colors.transpose(0,1) if self.buf.obs_colors is not None else None  # [B,T,21,79]
            obs_blstats_bt = self.buf.obs_blstats.transpose(0,1) if self.buf.obs_blstats is not None else None  # [B,T,blstats_dim]
            obs_message_bt = self.buf.obs_message.transpose(0,1) if self.buf.obs_message is not None else None  # [B,T,256]
            obs_hero_info_bt = self.buf.obs_hero_info.transpose(0,1) if self.buf.obs_hero_info is not None else None  # [B,T,4]
            
            # Get extrinsic rewards and done flags for VAE training
            rewards_bt = self.buf.rews_e.transpose(0,1)  # [B,T] - extrinsic rewards only
            dones_bt = self.buf.dones.transpose(0,1)     # [B,T] - done flags
            
            # Build replay buffers using class attributes
            if self.has_hmm:
                if len(self.replay_mu) == 0:
                    self.replay_mu = [mu_bt]
                    self.replay_logvar = [logvar_bt]
                    self.replay_lowrank_factors = [lowrank_bt] if lowrank_bt is not None else []
                else:
                    self.replay_mu.append(mu_bt)
                    self.replay_logvar.append(logvar_bt)
                    if lowrank_bt is not None:
                        self.replay_lowrank_factors.append(lowrank_bt)
            else:
                # Keep HMM-specific buffers empty to avoid unbounded growth when no HMM is present
                self.replay_mu = []
                self.replay_logvar = []
                self.replay_lowrank_factors = []

            if len(self.replay_actions) == 0:
                # First rollout - initialize VAE replay buffers
                self.replay_actions = [acts_bt]
                self.replay_rewards = [rewards_bt]
                self.replay_dones = [dones_bt]

                if obs_chars_bt is not None:
                    self.replay_obs_chars = [obs_chars_bt]
                    self.replay_obs_colors = [obs_colors_bt]
                    self.replay_obs_blstats = [obs_blstats_bt]
                    self.replay_obs_message = [obs_message_bt]
                    self.replay_obs_hero_info = [obs_hero_info_bt]
            else:
                # Append new rollout data for VAE buffers
                self.replay_actions.append(acts_bt)
                self.replay_rewards.append(rewards_bt)
                self.replay_dones.append(dones_bt)

                if obs_chars_bt is not None:
                    self.replay_obs_chars.append(obs_chars_bt)
                    self.replay_obs_colors.append(obs_colors_bt)
                    self.replay_obs_blstats.append(obs_blstats_bt)
                    self.replay_obs_message.append(obs_message_bt)
                    self.replay_obs_hero_info.append(obs_hero_info_bt)

    # --------------------------- compute advantages --------------------------

    @torch.no_grad()
    def _compute_intrinsic_for_buffer(self) -> Dict[str, torch.Tensor]:
        T, B = self.ppo_cfg.rollout_len, self.ppo_cfg.num_envs
        
        # Encode next observation for final timestep dynamics calculation
        next_obs_enc = self._encode_obs(self._obs, self._hero_info)
        next_obs_mu = next_obs_enc["mu"]
        next_obs_logvar = next_obs_enc["logvar"]  
        next_obs_F = next_obs_enc.get("F", None)
        
        # Get extended data that includes previous rollout's last timestep
        extended_data = self.buf.get_extended_data_for_transitions()
        
        self.curiosity.global_step = self.global_steps

        if extended_data["has_prev"]:
            # Use extended data for transition computations
            # extended_data has shape [T+1, B, ...] where 0 is prev timestep
            mu_ext = extended_data["mu"].transpose(0,1)      # [B, T+1, D]
            logvar_ext = extended_data["logvar"].transpose(0,1)  # [B, T+1, D]
            lowrank_factors_ext = extended_data["lowrank_factors"].transpose(0,1) if extended_data["lowrank_factors"] is not None else None
            actions_ext = extended_data["actions"].transpose(0,1)  # [B, T+1]
            dones_ext = extended_data["dones"].transpose(0,1)    # [B, T+1]
            mask_ext = extended_data["mask"].transpose(0,1)      # [B, T+1]
            
            # Current rollout data (for returning rewards aligned to timesteps)
            mu_curr = mu_ext[:, 1:, :]      # [B, T, D] - current rollout only
            logvar_curr = logvar_ext[:, 1:, :]  # [B, T, D]
            lowrank_factors_curr = lowrank_factors_ext[:, 1:, :, :] if lowrank_factors_ext is not None else None
            actions_curr = actions_ext[:, 1:]   # [B, T]
            dones_curr = dones_ext[:, 1:]     # [B, T]
            mask_curr = mask_ext[:, 1:]       # [B, T]
            
            # Compute intrinsic rewards using extended data for transition computations
            bonuses = self.curiosity.compute_intrinsic(
                mu_ext, logvar_ext, lowrank_factors_ext, actions_ext, mask_ext, dones_ext,
                next_obs_mu=next_obs_mu, next_obs_logvar=next_obs_logvar, next_obs_F=next_obs_F
            )
            
            # Extract rewards for current rollout timesteps only (skip the prepended previous timestep)
            result_bonuses = {}
            for key, value in bonuses.items():
                if value.dim() > 1:
                    # Per-timestep outputs (rewards, skill probabilities, etc.) - extract current rollout portion
                    result_bonuses[key] = value[:, 1:]  # [B, T, ...] - skip first timestep which is from prev rollout
                else:
                    # Scalar or batch-level outputs - keep as is
                    result_bonuses[key] = value
            
        else:
            # First rollout - use current data only
            mu_curr = self.buf.mu.transpose(0,1)      # [B,T,D]
            logvar_curr = self.buf.logvar.transpose(0,1)  # [B,T,D]
            lowrank_factors_curr = self.buf.lowrank_factors.transpose(0,1) if self.buf.lowrank_factors is not None else None
            actions_curr = self.buf.actions.transpose(0,1) # [B,T]
            dones_curr = self.buf.dones.transpose(0,1)   # [B,T]
            mask_curr = self.buf.mask.transpose(0,1)    # [B,T]
            
            result_bonuses = self.curiosity.compute_intrinsic(
                mu_curr, logvar_curr, lowrank_factors_curr, actions_curr, mask_curr, dones_curr,
                next_obs_mu=next_obs_mu, next_obs_logvar=next_obs_logvar, next_obs_F=next_obs_F
            )
        return result_bonuses

    @torch.no_grad()
    def _compute_skill_features(self, enc):
        """Helper to compute skill features for a single batch of observations."""
        if not self.ppo_cfg.policy_uses_skill:
            return None
            
        B = enc["z"].size(0)
        mu = enc["mu"]
        dv = enc["logvar"].exp()
        F = enc.get("lowrank_factors", None)
        logB = self.hmm.make_logB_for_filter(
            mu, dv, F, None, self.hmm_cfg.emission_mode, self.hmm_cfg.student_t_use_sample
        )  # [B, Kp1]
        Kp1 = self.hmm.niw.mu.size(0)
        skill_list = []
        
        for b in range(B):
            st = self._filt_state[b]
            if st is None:
                # initialise at episode start
                st = self.hmm.filter_init_from_logB(logB[b])
                alpha_b = torch.exp(st.log_alpha.to(torch.float32))  # [Kp1]
            else:
                # one causal update (uses cached ElogA)
                st, alpha_b, _, _, _ = self.hmm.filter_step(st, logB[b], self._ElogA)

            self._filt_state[b] = st
            skill_list.append(alpha_b[:Kp1-1])  # Drop remainder state
        
        return torch.stack(skill_list, dim=0)

    @torch.no_grad()
    def _advantages(self, rews_total: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
        """
        Compute GAE advantages with proper episode boundary handling.
        
        Args:
            rews_total: [T,B] total rewards (extrinsic + intrinsic)
            values: [T,B] value estimates for each timestep
            dones: [T,B] episode termination flags
        """
        T, B = rews_total.size()
        
        # Get bootstrap values (value of observation after rollout)
        enc = self._encode_obs(self._obs, self._hero_info)
        skill_feat = self._compute_skill_features(enc)
        _, bootstrap_values = self.actor_critic(enc["z"], skill_feat)
        
        # Extend values with bootstrap
        extended_values = torch.cat([values, bootstrap_values.unsqueeze(0)], dim=0)  # [T+1, B]
        
        # Compute advantages
        advantages = torch.zeros_like(rews_total)
        gae = torch.zeros(B, device=self.device)
        
        for t in reversed(range(T)):
            # Episode continues if current step is not done
            nextnonterminal = (~dones[t]).float()
            
            # TD error
            delta = rews_total[t] + self.ppo_cfg.gamma * extended_values[t + 1] * nextnonterminal - values[t]
            
            # GAE (resets to delta when episode ends)
            gae = delta + self.ppo_cfg.gamma * self.ppo_cfg.gae_lambda * nextnonterminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns

    def _ppo_update(self, advantages, returns, skills_for_policy):
        data = self.buf.get()
        Btotal = advantages.numel()
        inds = torch.randperm(Btotal, device=self.device)
        mb = self.ppo_cfg.minibatch_size
        for epoch in range(self.ppo_cfg.epochs_per_update):
            for start in range(0, Btotal, mb):
                idx = inds[start:start+mb]
                mu = data["mu"][idx]
                skill = None if skills_for_policy is None else skills_for_policy.reshape(-1, skills_for_policy.shape[-1])[idx]
                logits, value = self.actor_critic(mu, skill) # [N,G]
                # Same env for all samples -> same mask row; broadcast to batch
                masked = logits.masked_fill(~self.action_mask[0].unsqueeze(0), -1e9)
                dist = torch.distributions.Categorical(logits=masked)
                logp = dist.log_prob(data["actions"][idx])
                ratio = torch.exp(logp - data["logp"][idx])
                adv = advantages.reshape(-1)[idx]
                # normalise advantages
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                # policy loss
                unclipped = -adv * ratio
                clipped = -adv * torch.clamp(ratio, 1 - self.ppo_cfg.clip_coef, 1 + self.ppo_cfg.clip_coef)
                pg_loss = torch.max(unclipped, clipped).mean()
                # value loss
                v_loss = 0.5 * (returns.reshape(-1)[idx] - value).pow(2).mean()
                # entropy
                ent = dist.entropy().mean()
                loss = pg_loss + self.ppo_cfg.vf_coef * v_loss - self.ppo_cfg.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.ppo_cfg.max_grad_norm)
                self.opt.step()

    # --------------------------- HMM / VAE refresh ---------------------------

    def maybe_refresh_models(self, logger: Optional[logging.Logger] = None):
        """
        Synchronized HMM and VAE updates. Updates happen together when either interval is reached.
        This ensures coordinated learning between the skill discovery (HMM) and representation (VAE).
        """
        hmm_ready = self.has_hmm and ((self.global_steps - self.last_hmm_refresh) >= self._hmm_interval)
        vae_ready = (self.global_steps - self.last_vae_refresh) >= self._vae_interval and self.vae_opt is not None
        
        # Update both models if either is ready (synchronized updates)
        if hmm_ready or vae_ready:
            if logger is not None:
                logger.info(f"🔄 Synchronized model update at step {self.global_steps:,}")
                logger.info(f"   HMM interval: {self._hmm_interval:,}, VAE interval: {self._vae_interval:,}")
            
            # Check if we have replay data
            if len(self.replay_actions) == 0:
                if logger is not None:
                    logger.warning("No replay data available for model updates, skipping")
                return

            # Combine all replay data
            combined_mu = torch.cat(self.replay_mu, dim=1) if len(self.replay_mu) > 0 else None
            combined_logvar = torch.cat(self.replay_logvar, dim=1) if len(self.replay_logvar) > 0 else None
            combined_lowrank_factors = torch.cat(self.replay_lowrank_factors, dim=1) if len(self.replay_lowrank_factors) > 0 else None
            combined_actions = torch.cat(self.replay_actions, dim=1)
            combined_rewards = torch.cat(self.replay_rewards, dim=1)  # Extrinsic rewards
            combined_dones = torch.cat(self.replay_dones, dim=1)      # Done flags
            
            combined_obs_chars = torch.cat(self.replay_obs_chars, dim=1) if len(self.replay_obs_chars) > 0 else None
            combined_obs_colors = torch.cat(self.replay_obs_colors, dim=1) if len(self.replay_obs_colors) > 0 else None
            combined_obs_blstats = torch.cat(self.replay_obs_blstats, dim=1) if len(self.replay_obs_blstats) > 0 else None
            combined_obs_message = torch.cat(self.replay_obs_message, dim=1) if len(self.replay_obs_message) > 0 else None
            combined_obs_hero_info = torch.cat(self.replay_obs_hero_info, dim=1) if len(self.replay_obs_hero_info) > 0 else None
            
            # Crop to window size for all replay buffers (no mask needed - all steps are valid)
            current_window_size = combined_actions.size(1)
            if self.has_hmm and current_window_size > self.hmm_cfg.hmm_fit_window:
                s = current_window_size - self.hmm_cfg.hmm_fit_window
                combined_mu = combined_mu[:, s:, :] if combined_mu is not None else None
                combined_logvar = combined_logvar[:, s:, :] if combined_logvar is not None else None
                combined_lowrank_factors = combined_lowrank_factors[:, s:, :, :] if combined_lowrank_factors is not None else None
                combined_actions = combined_actions[:, s:]
                combined_rewards = combined_rewards[:, s:] 
                combined_dones = combined_dones[:, s:]
                
                # Crop observations too
                if combined_obs_chars is not None:
                    combined_obs_chars = combined_obs_chars[:, s:, :, :]
                    combined_obs_colors = combined_obs_colors[:, s:, :, :]
                    combined_obs_blstats = combined_obs_blstats[:, s:, :]
                    combined_obs_message = combined_obs_message[:, s:, :]
                    combined_obs_hero_info = combined_obs_hero_info[:, s:, :]
            
            # Create a valid mask for the entire window (all steps are valid in PPO online training)
            B, T = combined_actions.shape
            valid_mask = torch.ones(B, T, device=self.device, dtype=torch.bool)

            # HMM update (uses latent representations that we already have)
            if hmm_ready and combined_mu is not None:
                self._refresh_hmm(
                    combined_mu,
                    combined_logvar,
                    valid_mask,
                    combined_lowrank_factors,
                    combined_dones,
                    logger,
                )
                if logger is not None:
                    logger.info(f"   ✅ HMM updated (next in {self._hmm_interval:,} steps)")
                # Reset only HMM-related replay buffers
                self._reset_hmm_replay_buffers()
            
            # VAE update (uses raw observations)  
            if vae_ready and combined_obs_chars is not None:
                self._refresh_vae(combined_actions, valid_mask,
                                 combined_obs_chars, combined_obs_colors, combined_obs_blstats, 
                                 combined_obs_message, combined_obs_hero_info,
                                 combined_rewards, combined_dones, logger)
                if logger is not None:
                    logger.info(f"   ✅ VAE updated (next in {self._vae_interval:,} steps)")
                # Reset only VAE-related replay buffers
                self._reset_vae_replay_buffers()

            if not self.has_hmm:
                # Ensure HMM buffers stay clear when we are not training an HMM
                self._reset_hmm_replay_buffers()

    def _reset_replay_buffers(self):
        """Reset replay buffers to prevent memory accumulation."""
        # This method is kept for backward compatibility but now calls separate resets
        self._reset_hmm_replay_buffers()
        self._reset_vae_replay_buffers()
    
    def _reset_hmm_replay_buffers(self):
        """Reset HMM-related replay buffers after HMM refresh."""
        self.replay_mu = []
        self.replay_logvar = []
        self.replay_lowrank_factors = []

    def _reset_vae_replay_buffers(self):
        """Reset VAE-related replay buffers after VAE refresh."""
        self.replay_actions = []
        self.replay_rewards = []
        self.replay_dones = []
        self.replay_obs_chars = []
        self.replay_obs_colors = []
        self.replay_obs_blstats = []
        self.replay_obs_message = []
        self.replay_obs_hero_info = []

    def _split_rollouts_by_episode(
        self,
        mu: torch.Tensor,
        diag_var: torch.Tensor,
        lowrank: torch.Tensor | None,
        mask: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Split [B,T,...] sequences into per-episode segments using done flags."""
        if dones is None:
            return mu, diag_var, lowrank, mask

        B, T, D = mu.shape
        device = mu.device
        mask_dtype = mask.dtype if mask is not None else torch.bool
        segments_mu: list[torch.Tensor] = []
        segments_diag: list[torch.Tensor] = []
        segments_lowrank: list[torch.Tensor] | None = [] if lowrank is not None else None
        lengths: list[int] = []

        for b in range(B):
            start_idx = None
            for t in range(T):
                is_valid = bool(mask[b, t].item()) if mask is not None else True
                if not is_valid:
                    if start_idx is not None and t > start_idx:
                        end = t
                        segments_mu.append(mu[b, start_idx:end].clone())
                        segments_diag.append(diag_var[b, start_idx:end].clone())
                        if segments_lowrank is not None:
                            segments_lowrank.append(lowrank[b, start_idx:end].clone())
                        lengths.append(end - start_idx)
                        start_idx = None
                    continue

                if start_idx is None:
                    start_idx = t

                if bool(dones[b, t].item()):
                    end = t + 1
                    if start_idx is not None and end > start_idx:
                        segments_mu.append(mu[b, start_idx:end].clone())
                        segments_diag.append(diag_var[b, start_idx:end].clone())
                        if segments_lowrank is not None:
                            segments_lowrank.append(lowrank[b, start_idx:end].clone())
                        lengths.append(end - start_idx)
                    start_idx = None

            if start_idx is not None and start_idx < T:
                segments_mu.append(mu[b, start_idx:T].clone())
                segments_diag.append(diag_var[b, start_idx:T].clone())
                if segments_lowrank is not None:
                    segments_lowrank.append(lowrank[b, start_idx:T].clone())
                lengths.append(T - start_idx)

        if not segments_mu:
            return mu, diag_var, lowrank, mask

        max_len = max(lengths)
        new_B = len(segments_mu)
        new_mu = torch.zeros(new_B, max_len, D, device=device, dtype=mu.dtype)
        new_diag = torch.zeros(new_B, max_len, D, device=device, dtype=diag_var.dtype)
        new_mask = torch.zeros(new_B, max_len, device=device, dtype=mask_dtype)
        new_lowrank = None
        if segments_lowrank is not None and len(segments_lowrank) > 0:
            R = segments_lowrank[0].shape[-1]
            new_lowrank = torch.zeros(new_B, max_len, D, R, device=device, dtype=segments_lowrank[0].dtype)

        for idx, (seg_mu, seg_diag) in enumerate(zip(segments_mu, segments_diag)):
            L = seg_mu.shape[0]
            new_mu[idx, :L] = seg_mu
            new_diag[idx, :L] = seg_diag
            if new_lowrank is not None:
                new_lowrank[idx, :L] = segments_lowrank[idx]
            new_mask[idx, :L] = True

        return new_mu, new_diag, new_lowrank, new_mask

    @torch.no_grad()
    def _refresh_hmm(
        self,
        replay_mu: torch.Tensor,
        replay_logvar: torch.Tensor,
        replay_mask: torch.Tensor,
        replay_lowrank_factors: torch.Tensor | None = None,
        replay_dones: torch.Tensor | None = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Internal HMM update method."""
        if not self.has_hmm:
            return

        diag_var = replay_logvar.exp().clamp_min(1e-6)
        mu = replay_mu
        mask = replay_mask
        lowrank = replay_lowrank_factors

        if replay_dones is not None:
            mu, diag_var, lowrank, mask = self._split_rollouts_by_episode(mu, diag_var, lowrank, mask, replay_dones)

        B, T, D = mu.shape
        
        # Log HMM refresh start
        if logger is not None:
            logger.info(f"🔄 HMM refresh starting at step {self.global_steps:,} with replay size {B}x{T}")
        
        # run full VI update on the window
        out = self.hmm.update(
            mu_t=mu,
            diag_var_t=diag_var,
            F_t=lowrank,
            mask=mask,
            max_iters=self.hmm_cfg.hmm_max_iters, tol=self.hmm_cfg.hmm_tol, elbo_drop_tol=self.hmm_cfg.hmm_elbo_drop_tol,
            optimize_pi=self.hmm_cfg.optimise_pi,
            pi_steps=self.hmm_cfg.pi_steps, pi_lr=self.hmm_cfg.pi_lr,
            pi_early_stopping_patience=self.hmm_cfg.pi_early_stopping_patience,
            pi_early_stopping_min_delta=self.hmm_cfg.pi_early_stopping_min_delta, logger=logger
        )
        # refresh cached ElogA used by the online filter
        self._ElogA = self.hmm._ElogA()
        
        self.last_hmm_refresh = self.global_steps
        # relax cadence
        self._hmm_interval = min(int(self._hmm_interval * self.hmm_cfg.hmm_update_growth),
                                self.hmm_cfg.hmm_update_every_cap)
        
        # Log HMM update diagnostics
        self._log_scalar({
            "hmm_update/final_elbo": float(out.get("loglik", 0.0)),
            "hmm_update/iterations_used": int(out.get("n_iters", 0)),
            "hmm_update/converged": bool(out.get("converged", False)),
            "hmm_update/pi_optimization_iterations": int(out.get("pi_n_iters", 0)),
            "hmm_update/pi_converged": bool(out.get("pi_converged", False)),
            "hmm_update/effective_batch_size": int(B),
            "hmm_update/sequence_length": int(T),
            "hmm_update/rho_emission": float(self.hmm_cfg.rho_emission),
            "hmm_update/rho_transition": float(self.hmm_cfg.rho_transition if self.hmm_cfg.rho_transition is not None else self.hmm_cfg.rho_emission),
            "hmm_update/max_iters": int(self.hmm_cfg.hmm_max_iters),
            "hmm_update/tolerance": float(self.hmm_cfg.hmm_tol),
            "hmm_update/elbo_drop_tolerance": float(self.hmm_cfg.hmm_elbo_drop_tol),
            "hmm_update/pi_steps": int(self.hmm_cfg.pi_steps),
            "hmm_update/pi_lr": float(self.hmm_cfg.pi_lr),
            "hmm_update/pi_early_stopping_patience": int(self.hmm_cfg.pi_early_stopping_patience),
            "hmm_update/pi_early_stopping_min_delta": float(self.hmm_cfg.pi_early_stopping_min_delta),
            "steps": self.global_steps
        })
        
        # Log HMM refresh completion
        if logger is not None:
            num_states = self.hmm.niw.mu.size(0) - 1  # Exclude remainder state
            status = "converged" if bool(out.get("converged", False)) else "max iters"
            logger.info(f"   HMM refresh complete: ELBO={float(out.get('loglik', 0.0)):.4f}, iters={int(out.get('n_iters', 0))}, {status}, states={num_states}")

    def _refresh_vae(self, replay_actions: torch.Tensor, replay_mask: torch.Tensor,
                     replay_obs_chars: torch.Tensor = None, replay_obs_colors: torch.Tensor = None,
                     replay_obs_blstats: torch.Tensor = None, replay_obs_message: torch.Tensor = None,
                     replay_obs_hero_info: torch.Tensor = None, 
                     replay_rewards: torch.Tensor = None, replay_dones: torch.Tensor = None,
                     logger: Optional[logging.Logger] = None):
        """Internal VAE update method."""
        # Check if we have stored observations in the replay buffer
        if replay_obs_chars is None:
            if logger is not None:
                logger.warning("⚠️ No observations stored in replay buffer for VAE training, skipping...")
            return
            
        B, T = replay_obs_chars.shape[:2]
        
        # Log VAE refresh start
        if logger is not None:
            logger.info(f"🔄 VAE refresh starting at step {self.global_steps:,} with replay size {B}x{T}")

        device = self.device
        span = min(self.vae_cfg.span_len, T)
        if span < 2:
            if logger is not None:
                logger.warning(f"⚠️ Span length {span} too short for VAE training, skipping...")
            return

        # Set VAE to training mode
        self.vae.train()
        
        final_loss = 0.0
        
        # Choose between span sampling and full buffer usage based on buffer size
        total_valid_timesteps = B * T  # All timesteps are valid in PPO online training
        use_full_buffer = total_valid_timesteps <= 2048  # Use full buffer if reasonable size
        
        if use_full_buffer:
            # Use entire buffer for more complete training
            if logger is not None:
                logger.info(f"   Using full buffer ({total_valid_timesteps} timesteps)")
            
            # Gather all valid timesteps from replay buffer
            all_chars = replay_obs_chars.view(-1, 21, 79)      # [B*T, 21, 79]
            all_colors = replay_obs_colors.view(-1, 21, 79)    # [B*T, 21, 79]
            all_blstats = replay_obs_blstats.view(-1, replay_obs_blstats.shape[-1])  # [B*T, blstats_dim]
            all_message = replay_obs_message.view(-1, 256)     # [B*T, 256]
            all_hero_info = replay_obs_hero_info.view(-1, 4)   # [B*T, 4]
            all_actions = replay_actions.view(-1)              # [B*T]
            
            # Create action_onehot using one_hot encoding
            all_action_onehot = one_hot(all_actions, ACTION_DIM)  # [B*T, ACTION_DIM]
            
            # Create has_next: True for timesteps that are not the last AND not done
            all_has_next = torch.zeros(B*T, dtype=torch.bool, device=device)
            all_dones_flat = replay_dones.view(-1) if replay_dones is not None else torch.zeros(B*T, dtype=torch.bool, device=device)
            for b in range(B):
                for t in range(T-1):  # All except last timestep in sequence
                    idx = b*T + t
                    # has_next is True if not done at current timestep
                    all_has_next[idx] = not all_dones_flat[idx]
            
            # Use real extrinsic rewards and done targets (not dummy zeros)
            all_reward_target = replay_rewards.view(-1).float() if replay_rewards is not None else torch.zeros(B*T, dtype=torch.float32, device=device)
            all_done_target = replay_dones.view(-1).float() if replay_dones is not None else torch.zeros(B*T, dtype=torch.float32, device=device)
            
            # Use replay_mask for valid_screen instead of creating new mask
            all_valid_screen = replay_mask.view(-1).bool() if replay_mask is not None else torch.ones(B*T, dtype=torch.bool, device=device)
            
            # Calculate passability and safety targets
            all_passability_target = torch.zeros(B*T, 8, dtype=torch.float32, device=device)
            all_safety_target = torch.zeros(B*T, 8, dtype=torch.float32, device=device)
            all_hard_mask = torch.zeros(B*T, 8, dtype=torch.float32, device=device)
            all_weight = torch.zeros(B*T, 8, dtype=torch.float32, device=device)
            
            # Calculate passability and safety for each timestep where hero is visible
            for b in range(B):
                for t in range(T):
                    idx = b*T + t
                    if all_valid_screen[idx]:  # Only calculate for valid timesteps
                        chars_map = all_chars[idx]  # [21, 79]
                        colors_map = all_colors[idx]  # [21, 79]
                        
                        # Find hero position (prefer '@' character)
                        ys, xs = (chars_map == ord('@')).nonzero(as_tuple=True)
                        if ys.numel() > 0:
                            hy, hx = int(ys[0].item()), int(xs[0].item())
                            p8, s8, hm8, w8 = compute_passability_and_safety(chars_map, colors_map, hy, hx)
                            all_passability_target[idx] = p8
                            all_safety_target[idx] = s8
                            all_hard_mask[idx] = hm8
                            all_weight[idx] = w8
            
            combined_batch = {
                'game_chars': all_chars,
                'game_colors': all_colors,
                'blstats': all_blstats,
                'message_chars': all_message,
                'hero_info': all_hero_info,
                'valid_screen': all_valid_screen,
                'action_onehot': all_action_onehot,
                'has_next': all_has_next,
                'reward_target': all_reward_target,
                'done_target': all_done_target,
                'passability_target': all_passability_target,
                'safety_target': all_safety_target,
                'hard_mask': all_hard_mask,
                'weight': all_weight,
                'original_batch_shape': (B, T)  # For sequential processing in VAE
            }
            
            # Single training step with full buffer
            vae_steps_to_run = 1
        else:
            # Use span sampling for large buffers
            if logger is not None:
                logger.info(f"   Using span sampling ({total_valid_timesteps} timesteps, too large for full buffer)")
            vae_steps_to_run = self.vae_cfg.vae_steps_per_call
        
        for step in range(vae_steps_to_run):
            if not use_full_buffer:
                # Sample random spans from replay buffer
                b_idx = torch.randint(low=0, high=B, size=(self.vae_cfg.mini_batch_B,), device=device)
                t0 = torch.randint(low=0, high=max(1, T - span), size=(self.vae_cfg.mini_batch_B,), device=device)
                
                # Gather observation spans and create batches from replay buffer
                batch_list = []
                for bi, t in zip(b_idx.tolist(), t0.tolist()):
                    if t + span > T:
                        continue  # Skip invalid spans
                    
                    # Extract span data
                    span_chars = replay_obs_chars[bi, t:t+span]      # [span, 21, 79]
                    span_colors = replay_obs_colors[bi, t:t+span]    # [span, 21, 79]
                    span_blstats = replay_obs_blstats[bi, t:t+span]  # [span, blstats_dim]
                    span_message = replay_obs_message[bi, t:t+span]  # [span, 256]
                    span_hero_info = replay_obs_hero_info[bi, t:t+span]  # [span, 4]
                    span_actions = replay_actions[bi, t:t+span]      # [span]
                    
                    # Extract real reward and done data for this span
                    span_rewards = replay_rewards[bi, t:t+span] if replay_rewards is not None else torch.zeros(span, dtype=torch.float32, device=device)
                    span_dones = replay_dones[bi, t:t+span] if replay_dones is not None else torch.zeros(span, dtype=torch.bool, device=device)
                    span_mask = replay_mask[bi, t:t+span].bool() if replay_mask is not None else torch.ones(span, dtype=torch.bool, device=device)
                    
                    # Create action_onehot using one_hot encoding
                    span_action_onehot = one_hot(span_actions, ACTION_DIM)  # [span, ACTION_DIM]
                    
                    # Create has_next: True for timesteps that are not the last AND not done
                    span_has_next = torch.zeros(span, dtype=torch.bool, device=device)
                    for s in range(span-1):  # All except last timestep in span
                        # has_next is True if not done at current timestep
                        span_has_next[s] = not span_dones[s]
                    
                    # Use real extrinsic rewards and done targets (not dummy zeros)
                    span_reward_target = span_rewards.float()
                    span_done_target = span_dones.float()
                    
                    # Calculate passability and safety targets for this span
                    span_passability_target = torch.zeros(span, 8, dtype=torch.float32, device=device)
                    span_safety_target = torch.zeros(span, 8, dtype=torch.float32, device=device)
                    span_hard_mask = torch.zeros(span, 8, dtype=torch.float32, device=device)
                    span_weight = torch.zeros(span, 8, dtype=torch.float32, device=device)
                    
                    # Calculate passability and safety for each timestep in span where hero is visible
                    for s in range(span):
                        if span_mask[s]:  # Only calculate for valid timesteps
                            chars_map = span_chars[s]  # [21, 79]
                            colors_map = span_colors[s]  # [21, 79]
                            
                            # Find hero position (prefer '@' character)
                            ys, xs = (chars_map == ord('@')).nonzero(as_tuple=True)
                            if ys.numel() > 0:
                                hy, hx = int(ys[0].item()), int(xs[0].item())
                                p8, s8, hm8, w8 = compute_passability_and_safety(chars_map, colors_map, hy, hx)
                                span_passability_target[s] = p8
                                span_safety_target[s] = s8
                                span_hard_mask[s] = hm8
                                span_weight[s] = w8
                    
                    batch_list.append({
                        'game_chars': span_chars,
                        'game_colors': span_colors,
                        'blstats': span_blstats,
                        'message_chars': span_message,
                        'hero_info': span_hero_info,
                        'valid_screen': span_mask,
                        'action_onehot': span_action_onehot,
                        'has_next': span_has_next,
                        'reward_target': span_reward_target,
                        'done_target': span_done_target,
                        'passability_target': span_passability_target,
                        'safety_target': span_safety_target,
                        'hard_mask': span_hard_mask,
                        'weight': span_weight
                    })
                
                if not batch_list:
                    continue  # No valid spans found
                
                # Concatenate all spans into a single batch
                # Reshape from [num_spans, span_len, ...] to [num_spans * span_len, ...]
                all_chars = torch.cat([b['game_chars'] for b in batch_list], dim=0)
                all_colors = torch.cat([b['game_colors'] for b in batch_list], dim=0)
                all_blstats = torch.cat([b['blstats'] for b in batch_list], dim=0)
                all_message = torch.cat([b['message_chars'] for b in batch_list], dim=0)
                all_hero_info = torch.cat([b['hero_info'] for b in batch_list], dim=0)
                all_valid = torch.cat([b['valid_screen'] for b in batch_list], dim=0)
                all_action_onehot = torch.cat([b['action_onehot'] for b in batch_list], dim=0)
                all_has_next = torch.cat([b['has_next'] for b in batch_list], dim=0)
                all_reward_target = torch.cat([b['reward_target'] for b in batch_list], dim=0)
                all_done_target = torch.cat([b['done_target'] for b in batch_list], dim=0)
                all_passability_target = torch.cat([b['passability_target'] for b in batch_list], dim=0)
                all_safety_target = torch.cat([b['safety_target'] for b in batch_list], dim=0)
                all_hard_mask = torch.cat([b['hard_mask'] for b in batch_list], dim=0)
                all_weight = torch.cat([b['weight'] for b in batch_list], dim=0)
                
                # Calculate combined batch shape for sequential processing
                num_spans = len(batch_list)
                span_len = batch_list[0]['game_chars'].shape[0]
                
                combined_batch = {
                    'game_chars': all_chars,
                    'game_colors': all_colors,
                    'blstats': all_blstats,
                    'message_chars': all_message,
                    'hero_info': all_hero_info,
                    'valid_screen': all_valid,
                    'action_onehot': all_action_onehot,
                    'has_next': all_has_next,
                    'reward_target': all_reward_target,
                    'done_target': all_done_target,
                    'passability_target': all_passability_target,
                    'safety_target': all_safety_target,
                    'hard_mask': all_hard_mask,
                    'weight': all_weight,
                    'original_batch_shape': (num_spans, span_len)  # For sequential processing in VAE
                }
            combined_batch['sticky_hmm'] = self.hmm if self.has_hmm else None
            # Forward pass through VAE to get complete model output (like train.py)
            self.vae_opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=False):
                model_output = self.vae(combined_batch)
                
                # Calculate VAE loss with lighter coefficients for online training
                vae_loss_dict = vae_loss(
                    model_output=model_output,
                    batch=combined_batch,
                    config=self.vae_cfg.training_config,
                    mi_beta=self.vae_cfg.mi_beta,
                    tc_beta=self.vae_cfg.tc_beta,
                    dw_beta=self.vae_cfg.dw_beta
                )
                loss = vae_loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            self.vae_opt.step()
            
            # Track final loss
            final_loss = float(loss.detach().item())
            
            # Log VAE training metrics (only on first step to avoid spam)
            if step == 0:
                self._log_scalar({
                    "vae/total_loss": float(loss.detach().item()),
                    "vae/raw_loss": float(vae_loss_dict['total_raw_loss'].item()),
                    "vae/mi_beta": self.vae_cfg.mi_beta,
                    "vae/tc_beta": self.vae_cfg.tc_beta,
                    "vae/dw_beta": self.vae_cfg.dw_beta,
                    "vae/training_triggered": 1.0,  # Flag that VAE training occurred
                    "vae/timesteps_processed": float(total_valid_timesteps),
                    "steps": self.global_steps
                })
        
        self.last_vae_refresh = self.global_steps
        # Relax VAE update cadence (same as HMM)
        self._vae_interval = min(int(self._vae_interval * self.vae_cfg.vae_update_growth),
                                 self.vae_cfg.vae_update_every_cap)
        
        # Set VAE back to eval mode for inference
        self.vae.eval()
        
        # Log VAE refresh completion
        if logger is not None:
            logger.info(f"   VAE refresh complete: loss={final_loss:.4f}, training_steps={vae_steps_to_run}, replay_timesteps={total_valid_timesteps}")

    # --------------------------- main train loop -----------------------------

    def train(self, logger: Optional[logging.Logger] = None):
        set_seed(self.run_cfg.seed)
        # Reset environments only once at the beginning of training
        # After this, collect_rollout() will continue from the current state
        obs, _ = self.envs.reset(seed=self.run_cfg.seed)
        
        # Parse hero info from initial observations (episode start)
        self._episode_start = [True for _ in range(self.ppo_cfg.num_envs)]
        self._update_hero_info_from_obs(obs, self.ppo_cfg.num_envs, self._episode_start, self._hero_info, logger=logger)
        
        self._obs = obs

        # Calculate total training parameters for progress tracking
        total_env_steps = self.ppo_cfg.total_updates * self.ppo_cfg.rollout_len * self.ppo_cfg.num_envs
        steps_per_update = self.ppo_cfg.rollout_len * self.ppo_cfg.num_envs
        
        print(f"🚀 Starting PPO Training")
        print(f"   Total updates: {self.ppo_cfg.total_updates}")
        print(f"   Steps per update: {steps_per_update}")
        print(f"   Total environment steps: {total_env_steps:,}")
        print(f"   Environment: {self.env_id}")
        print(f"   Seed: {self.run_cfg.seed}")
        print()
        
        # Create progress bar for training updates
        pbar = tqdm(
            total=self.ppo_cfg.total_updates, 
            desc="Training Progress",
            unit="update",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}"
        )
        
        update_count = 0
        start_time = time.time()
        
        while self.global_steps < total_env_steps:
            update_start_time = time.time()
            
            self.collect_rollout(logger)
            bonuses = self._compute_intrinsic_for_buffer()
            # skills for policy (concat to z) during PPO update: use the SAME features we acted with
            skills_for_policy = self.buf.skill.transpose(0,1) if (self.ppo_cfg.policy_uses_skill and self.buf.skill is not None) else None  # [T,B,K]
            # total reward
            intrinsic = torch.relu(bonuses["dyn"]) + torch.relu(bonuses["hdp"]) + torch.relu(bonuses["trans"]) + torch.relu(bonuses["rnd"])  # [B, T]
            ext = self.buf.rews_e  # [T, B]
            rews_total = ext + intrinsic.transpose(0, 1)  # [T, B]
            adv, ret = self._advantages(rews_total, self.buf.val, self.buf.dones)
            self._ppo_update(adv, ret, skills_for_policy)

            # optional RND predictor update to keep error scale meaningful
            if self.curiosity.use_rnd:
                mu_flat = self.buf.mu.reshape(-1, self.buf.mu.size(-1)).detach()
                for _ in range(self.rnd_cfg.update_per_rollout):
                    self.curiosity.train_rnd(mu_flat)

            # Use synchronized model updates for coordinated learning
            self.maybe_refresh_models(logger)

            # logging
            intrinsic_mean = intrinsic.mean()
            cur_eff = float((ext.mean() / (intrinsic_mean + 1e-8)).item())
            
            # Update progress bar and add metrics
            update_count += 1
            update_time = time.time() - update_start_time
            elapsed_time = time.time() - start_time
            updates_per_sec = update_count / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = (self.ppo_cfg.total_updates - update_count) / updates_per_sec if updates_per_sec > 0 else 0
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
            
            # Update progress bar with rich information
            pbar.set_postfix({
                'Steps': f"{self.global_steps:,}",
                'ExtRet': f"{ext.mean().item():.2f}",
                'CurEff': f"{cur_eff:.2f}",
                'UPS': f"{updates_per_sec:.1f}",
                'ETA': eta_str
            })
            pbar.update(1)
            
            metrics = {
                "steps": self.global_steps,
                "return/mean_ext": float(ext.mean().item()),
                "int/dyn_mean": float(bonuses["dyn"].mean().item()),
                "int/hdp_mean": float(bonuses["hdp"].mean().item()),
                "int/trans_mean": float(bonuses["trans"].mean().item()),
                "int/rnd_mean": float(bonuses["rnd"].mean().item()),
                "curiosity/efficiency": cur_eff,
                # Add raw (always positive) vs normalized comparisons
                "int/dyn_raw_mean": float(bonuses["dyn_raw"].mean().item()),
                "int/hdp_raw_mean": float(bonuses["hdp_raw"].mean().item()),
                "int/trans_raw_mean": float(bonuses["trans_raw"].mean().item()),
                "int/rnd_raw_mean": float(bonuses["rnd_raw"].mean().item()),
                # Track negative ratio to monitor normalization impact
                "int/negative_ratio": float((intrinsic < 0).float().mean().item()),
                "int/total_mean": float(intrinsic.mean().item()),
                "int/total_std": float(intrinsic.std().item()),
                # Synchronized model training diagnostics
                "learning_cycle/hmm_training_gap": float(self.global_steps - self.last_hmm_refresh),
                "learning_cycle/vae_training_gap": float(self.global_steps - self.last_vae_refresh),
                "learning_cycle/hmm_interval": float(self._hmm_interval),
                "learning_cycle/vae_interval": float(self._vae_interval),
            }

            if self.has_hmm:
                nu = self.hmm.niw.nu
                unused_mask = (nu[:-1] <= 106.1)
                used_mask = (nu[:-1] > 106.1)
                metrics.update({
                    "hmm/hdp_raw_std": float(bonuses["hdp_raw"].std().item()),
                    "hmm/hdp_raw_min": float(bonuses["hdp_raw"].min().item()),
                    "hmm/hdp_raw_max": float(bonuses["hdp_raw"].max().item()),
                    "hmm/skill_posterior_entropy_var": float(bonuses["hdp_raw"].var().item()),
                    "hmm/constant_entropy_ratio": float((torch.abs(bonuses["hdp_raw"] - bonuses["hdp_raw"].mean()) < 0.01).float().mean().item()),
                    "hmm/unused_skills": float(unused_mask.float().sum().item()),
                    "hmm/used_skills": float(used_mask.float().sum().item()),
                    "hmm/max_nu": float(nu.max().item()),
                    "hmm/mean_nu": float(nu.mean().item()),
                    "learning_cycle/unused_skill_posterior_mass": float(bonuses["rhat_skill"][:, :, unused_mask].sum().item()),
                    "learning_cycle/used_skill_posterior_mass": float(bonuses["rhat_skill"][:, :, used_mask].sum().item()),
                })
            else:
                metrics.update({
                    "hmm/hdp_raw_std": 0.0,
                    "hmm/hdp_raw_min": 0.0,
                    "hmm/hdp_raw_max": 0.0,
                    "hmm/skill_posterior_entropy_var": 0.0,
                    "hmm/constant_entropy_ratio": 0.0,
                    "hmm/unused_skills": 0.0,
                    "hmm/used_skills": 0.0,
                    "hmm/max_nu": 0.0,
                    "hmm/mean_nu": 0.0,
                    "learning_cycle/unused_skill_posterior_mass": 0.0,
                    "learning_cycle/used_skill_posterior_mass": 0.0,
                })

            self._log_scalar(metrics)
            if (self.global_steps % self.run_cfg.eval_every) < (self.ppo_cfg.num_envs * self.ppo_cfg.rollout_len):
                pbar.write(f"🧪 Running evaluation at step {self.global_steps:,}...")
                self.evaluate(self.run_cfg.eval_episodes)
            if (self.global_steps % self.run_cfg.save_every) < (self.ppo_cfg.num_envs * self.ppo_cfg.rollout_len):
                pbar.write(f"💾 Saving checkpoint at step {self.global_steps:,}...")
                self._save_ckpt()
        
        # Training completed
        pbar.close()
        final_time = time.time() - start_time
        print(f"\n✅ Training completed!")
        print(f"   Total time: {final_time/3600:.2f} hours")
        print(f"   Final steps: {self.global_steps:,}")
        print(f"   Updates completed: {update_count}")
        print(f"   Average updates/sec: {update_count/final_time:.2f}")
        print()

    @torch.no_grad()
    def evaluate(self, episodes: int):
        """
        Evaluation with richer diagnostics to support:
          (1) VAE+HMM+PPO vs VAE+PPO (task returns, success, sample-efficiency)
          (2) Curiosity vs RND (decomposition & exploration proxies)
        """
        print(f"🧪 Starting evaluation with {episodes} episodes...")
        start_time = time.time()
        
        env = gym.make(self.env_id)
        episode_start = [True]
        hero_info = [None]  # single env
        eval_filt_state = None  # single env HMM filter state
        ret_list, len_list, succ_list = [], [], []
        cover_list = []    # unique (x,y) cells visited per episode (if blstats available)
        len_succ_list = []  # length of successful episodes
        # HMM‑centric diagnostics
        bound_mass_list, bound_bool_list, ent_list = [], [], []
        used_skills_list, effK_list = [], []

        # Create progress bar for evaluation episodes
        eval_pbar = tqdm(
            range(episodes), 
            desc="Evaluation", 
            unit="ep",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}"
        )

        for ep_idx in eval_pbar:
            ep_start_time = time.time()
            o, _ = env.reset()
            episode_start = [True]
            eval_filt_state = None  # Reset filter state for new episode
            self._update_hero_info_from_obs(o, 1, episode_start, hero_info)
            done = False; ret = 0.0; ep_len = 0
            max_episode_steps = 10000  # Maximum steps per episode to prevent hanging
            # buffers to run HMM causal filter post‑episode
            mu_seq, logvar_seq, F_seq = [], [], []
            visited = set()

            while not done and ep_len < max_episode_steps:
                # coverage proxy from blstats (x,y)
                if isinstance(o, dict) and "blstats" in o:
                    bl = o["blstats"]
                    if getattr(bl, "ndim", 0) == 1:
                        x, y = int(bl[0]), int(bl[1])
                    else:
                        x, y = int(bl[0][0]), int(bl[0][1])  # [B,dim]
                    visited.add((x, y))

                enc = self._encode_obs(o, hero_info)
                
                # ---- Causal skill filtering (per-env, current frame) ----
                skill_feat = None
                if self.ppo_cfg.policy_uses_skill:
                    # Use single-environment version of the training logic
                    Kp1 = self.hmm.niw.mu.size(0)
                    mu_b = enc["mu"]  # [1, D] since we have single env batch
                    dv_b = enc["logvar"].exp().clamp_min(1e-6)  # [1, D]
                    F_b = enc.get('lowrank_factors', None)  # [1, D, R] or None
                    
                    logB_b = self.hmm.make_logB_for_filter(
                        mu_b, dv_b, F_b, None, ('mean' if self.hmm_cfg.emission_mode != 'student_t' else 'student_t'), False
                    ).squeeze(0)  # [Kp1] - always use deterministic mean for eval

                    if eval_filt_state is None:
                        # Initialize at episode start (same as training)
                        eval_filt_state = self.hmm.filter_init_from_logB(logB_b)
                        alpha_b = torch.exp(eval_filt_state.log_alpha.to(torch.float32))  # [Kp1]
                    else:
                        # One causal update (same as training)
                        eval_filt_state, alpha_b, _xi, _bound, _sent = self.hmm.filter_step(eval_filt_state, logB_b, self._ElogA)
                    
                    # Drop remainder state for policy features (same as training)
                    skill_feat = alpha_b[:Kp1-1].unsqueeze(0)  # [1, K] - add batch dim for consistency
                
                mu_seq.append(enc["mu"].squeeze(0).cpu())
                logvar_seq.append(enc["logvar"].squeeze(0).cpu())
                if "lowrank_factors" in enc and enc["lowrank_factors"] is not None:
                    F_seq.append(enc["lowrank_factors"].squeeze(0).cpu())
                else:
                    F_seq.append(None)

                logits, value = self.actor_critic(enc["z"], skill_feat)
                # For evaluation with single env, use only first row of action mask
                eval_action_mask = self.action_mask[0]  # [G] 
                masked = logits.masked_fill(~eval_action_mask, -1e9)
                a_global = (torch.argmax(masked, dim=-1)
                     if self.ppo_cfg.deterministic_eval
                     else torch.distributions.Categorical(logits=masked).sample())
                
                # Convert global action to local action for single environment
                # Use first row of global2local mapping
                eval_global2local = self.global2local[0]  # [G]
                a_local = eval_global2local[a_global.item()]
                
                o, r, term, trunc, _ = env.step(int(a_local.item()))
                done = term or trunc
                episode_start = [done]
                if done:
                    eval_filt_state = None  # Reset HMM filter state at episode end (same as training)
                self._update_hero_info_from_obs(o, 1, episode_start, hero_info)
                ret += float(r); ep_len += 1

            # Check if episode was terminated due to step limit
            if ep_len >= max_episode_steps and not done:
                print(f"⚠️ Episode {ep_idx + 1} terminated due to step limit ({max_episode_steps} steps)")

            # episode‑level tallies
            ret_list.append(ret)
            len_list.append(ep_len)
            succ = 1.0 if ret > 0.0 else 0.0
            succ_list.append(succ)
            if succ > 0.5:
                len_succ_list.append(ep_len)
            cover_list.append(len(visited))
            
            # Update progress bar with episode stats
            ep_time = time.time() - ep_start_time
            current_avg_ret = np.mean(ret_list) if ret_list else 0.0
            current_success_rate = np.mean(succ_list) if succ_list else 0.0
            
            eval_pbar.set_postfix({
                'Ret': f"{ret:.1f}",
                'AvgRet': f"{current_avg_ret:.2f}",
                'Success': f"{current_success_rate:.1%}",
                'Len': f"{ep_len}",
                'Time': f"{ep_time:.1f}s"
            })

            # ---------- HMM filter over the episode ----------
            if self.ppo_cfg.policy_uses_skill:
                mu_t = torch.stack(mu_seq, dim=0).to(self.device)           # [T,D]
                logvar_t = torch.stack(logvar_seq, dim=0).to(self.device)   # [T,D]
                diag_var_t = torch.exp(logvar_t)
                F_t = torch.stack(F_seq, dim=0).to(self.device) if F_seq[0] is not None else None  # [T,D,R] or None

                # Emission potentials and causal filter
                logB = self.hmm.make_logB_for_filter(
                    mu_t.unsqueeze(0), 
                    diag_var_t.unsqueeze(0), 
                    F_t.unsqueeze(0) if F_t is not None else None, 
                    None, 
                    ('mean' if self.hmm_cfg.emission_mode != 'student_t' else 'student_t'), 
                    False
                ).squeeze(0) # [T,Kp1], always use deterministic mean for eval

                fs = self.hmm.filter_sequence(logB)                     # dict with alpha, xi, boundary_prob, skill_entropy
                alpha = fs["alpha"]                                     # [T,Kp1]
                boundary_prob = fs["boundary_prob"]                     # [T]
                skill_entropy = fs["skill_entropy"]                     # [T]

                # boolean gate 1[ΔH_t > 0]
                dH = torch.nan_to_num(skill_entropy[1:] - skill_entropy[:-1])
                gate_bool = (dH > self.cur_cfg.gate_delta_eps).float()
                bound_bool_rate = float(gate_bool.mean().item()) if gate_bool.numel() > 0 else 0.0
                bound_mass_rate = float(torch.nan_to_num(boundary_prob[1:]).mean().item()) if boundary_prob.numel() > 1 else 0.0
                ent_mean = float(torch.nan_to_num(skill_entropy).mean().item())

                # used skills (exclude remainder state = last index)
                if alpha.numel() > 0:
                    Kp1 = alpha.size(-1)
                    alpha_no_rest = alpha[:, :Kp1-1]
                    used = torch.unique(torch.argmax(alpha_no_rest, dim=-1)).numel()
                    occ = alpha_no_rest.sum(dim=0); occ = occ / (occ.sum() + 1e-12)
                    effK = float(torch.exp(-(occ.clamp_min(1e-12) * occ.clamp_min(1e-12).log()).sum()).item())
                else:
                    used, effK = 0, 0.0

                bound_mass_list.append(bound_mass_rate)
                bound_bool_list.append(bound_bool_rate)
                ent_list.append(ent_mean)
                used_skills_list.append(int(used))
                effK_list.append(effK)

        eval_pbar.close()
        env.close()
        
        # Calculate evaluation summary
        eval_time = time.time() - start_time
        final_avg_ret = np.mean(ret_list) if ret_list else 0.0
        final_success_rate = np.mean(succ_list) if succ_list else 0.0
        final_avg_len = np.mean(len_list) if len_list else 0.0
        final_avg_coverage = np.mean(cover_list) if cover_list else 0.0
        
        print(f"📊 Evaluation Results:")
        print(f"   Episodes: {episodes}")
        print(f"   Time: {eval_time:.1f}s ({eval_time/episodes:.1f}s/ep)")
        print(f"   Average Return: {final_avg_ret:.3f} ± {np.std(ret_list):.3f}")
        print(f"   Success Rate: {final_success_rate:.1%}")
        print(f"   Average Length: {final_avg_len:.1f}")
        print(f"   Average Coverage: {final_avg_coverage:.1f} positions")
        if bound_mass_list:
            print(f"   Skill Entropy: {np.mean(ent_list):.3f}")
            print(f"   Used Skills: {np.mean(used_skills_list):.1f}")
            print(f"   Effective K: {np.mean(effK_list):.2f}")
        print()

        # Aggregate & log
        def _p(arr, q): return float(np.percentile(arr, q)) if len(arr) else 0.0
        log_dict = {
            "eval/return_mean": float(final_avg_ret),
            "eval/return_median": float(_p(ret_list, 50)),
            "eval/return_10th": float(_p(ret_list, 10)),
            "eval/return_90th": float(_p(ret_list, 90)),
            "eval/return_std": float(np.std(ret_list) if len(ret_list) else 0.0),
            "eval/success_rate": float(final_success_rate),
            "eval/ep_len_mean": float(final_avg_len),
            "eval/ep_len_success_mean": float(np.mean(len_succ_list) if len(len_succ_list) else 0.0),
            "eval/coverage_pos_mean": float(final_avg_coverage),
        }
        if bound_mass_list:
            log_dict.update({
                "eval/skill_boundary_mass_rate": float(np.mean(bound_mass_list)),
                "eval/skill_boundary_bool_rate": float(np.mean(bound_bool_list)),
                "eval/skill_entropy_mean": float(np.mean(ent_list)),
                "eval/used_skills_mean": float(np.mean(used_skills_list)),
                "eval/effective_K_mean": float(np.mean(effK_list)),
            })
        self._log_scalar(log_dict)

    def _log_scalar(self, d: Dict[str, float]):
        # Always log to JSON file
        with open(self._log_file, "a") as f:
            f.write(json.dumps(d) + "\n")
        
        # Also log to W&B if available
        if self.wandb_run is not None:
            try:
                # Extract step for W&B logging
                step = d.get("steps", self.global_steps)
                # Log all metrics to W&B
                wandb_dict = {k: v for k, v in d.items() if k != "steps"}
                self.wandb_run.log(wandb_dict, step=step)
            except Exception as e:
                # Don't fail training if W&B logging fails
                print(f"Warning: W&B logging failed: {e}")

    def _save_ckpt(self):
        path = os.path.join(self.run_cfg.log_dir, f"ckpt_{self.global_steps}.pt")
        payload = {
            "actor_critic": self.actor_critic.state_dict(),
            "opt": self.opt.state_dict(),
        }
        if self.has_hmm:
            payload["hmm"] = self.hmm.get_posterior_params()
        torch.save(payload, path)

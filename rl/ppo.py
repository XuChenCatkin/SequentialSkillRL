from __future__ import annotations
import os, time, math, json, random, dataclasses
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MiniHack / Gym
import gymnasium as gym
from minihack import MiniHackNavigation, MiniHackSkill, MiniHack
from nle import nethack  # action id space (615)

# --- your code ---
from src.model import MultiModalHackVAE, VAEConfig
from src.skill_space import StickyHDPHMMVI, StickyHDPHMMParams, NIWPrior
from src.data_collection import NetHackDataCollector
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
    use_skill_entropy: bool = True # (B) skill-posterior entropy
    use_rnd: bool = False         # (C) RND baseline (set True for baseline run)

    # Annealing: eta(t) = eta0 * exp(-t / tau)
    eta0_dyn: float = 1.0
    tau_dyn: float = 3e6
    eta0_hdp: float = 1.0
    tau_hdp: float = 3e6
    eta0_rnd: float = 0.25        # keep smaller by default
    tau_rnd: float = 3e6

    # EMA norm for each raw term
    ema_beta: float = 0.99
    eps: float = 1e-8

@dataclass
class HMMOnlineConfig:
    # HMM update cadence and footprint
    hmm_update_every: int = 50_000          # env steps between HMM refreshes
    hmm_fit_window: int = 400_000           # how many most recent steps to re-fit on
    hmm_max_iters: int = 7                  # inner VI iterations
    hmm_tol: float = 1e-2                   # relative ELBO tolerance
    hmm_elbo_drop_tol: float = 1e-2
    rho_emission: float = 0.05              # streaming blend
    rho_transition: Optional[float] = None  # default to rho_emission if None
    optimise_pi: bool = True
    reset_low_count_states: Optional[float] = 5e-4  # if occupancy below this, reset NIW to prior

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
    """Track mean/var with EMA and normalise new values."""
    def __init__(self, beta=0.99, eps=1e-8, device="cpu"):
        self.beta = beta; self.eps = eps
        self.mean = torch.zeros((), device=device)
        self.var  = torch.ones((), device=device)
        self.initialised = False
    @torch.no_grad()
    def update(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N] or [N, ...] flattened later
        val = x.detach()
        m = val.mean()
        v = val.var(unbiased=False) + self.eps
        if not self.initialised:
            self.mean.copy_(m); self.var.copy_(v); self.initialised = True
        else:
            self.mean = self.beta * self.mean + (1 - self.beta) * m
            self.var  = self.beta * self.var  + (1 - self.beta) * v
        return (x - self.mean) / (self.var.sqrt() + self.eps)

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
    def __init__(self, vae: MultiModalHackVAE, hmm: StickyHDPHMMVI, device, cur_cfg: CuriosityConfig, rnd_cfg: RNDConfig, z_dim: int, skill_K: int):
        self.vae = vae
        self.hmm = hmm
        self.device = device
        self.cfg = cur_cfg

        self.norm_dyn = EMANormalizer(cur_cfg.ema_beta, cur_cfg.eps, device)
        self.norm_hdp = EMANormalizer(cur_cfg.ema_beta, cur_cfg.eps, device)
        self.norm_rnd = EMANormalizer(cur_cfg.ema_beta, cur_cfg.eps, device)

        self.global_step = 0

        self.use_rnd = cur_cfg.use_rnd
        self.rnd = None
        self.rnd_opt = None
        if self.use_rnd:
            self.rnd = RNDModule(z_dim, proj_dim=rnd_cfg.proj_dim, hidden=rnd_cfg.hidden).to(device)
            self.rnd_opt = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_cfg.lr)
            self.rnd_updates_per_rollout = rnd_cfg.update_per_rollout
        self.skill_K = skill_K

    @torch.no_grad()
    def _eta(self, eta0: float, tau: float) -> float:
        return float(eta0 * math.exp(- self.global_step / max(1.0, tau)))

    @torch.no_grad()
    def compute_skill_rhat(self, mu_seq: torch.Tensor, diagvar_seq: torch.Tensor, F_seq: Optional[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        Sequence responsibilities via HMM FB.
        Args:
          mu_seq     : [B,T,D]
          diagvar_seq: [B,T,D]
          F_seq      : [B,T,D,R] or None
          mask       : [B,T] (1=valid)
        Returns:
          rhat: [B,T,Kp1]
        """
        logB = StickyHDPHMMVI.expected_emission_loglik(
            self.hmm.niw.mu, self.hmm.niw.kappa, self.hmm.niw.Psi, self.hmm.niw.nu,
            mu_seq, diagvar_seq, F_seq, mask
        )  # [B,T,Kp1]
        ElogA = self.hmm._ElogA()
        log_pi = torch.log(torch.clamp(self.hmm._Epi(), min=1e-30))
        B, T, Kp1 = logB.shape
        r_list = []
        for b in range(B):
            r, _, _ = StickyHDPHMMVI.forward_backward(log_pi, ElogA, logB[b])
            if mask is not None:
                m = mask[b].bool()
                r = r * m.unsqueeze(-1)
            r_list.append(r)
        rhat = torch.stack(r_list, dim=0)  # [B,T,Kp1]
        return rhat

    def compute_intrinsic(
        self,
        mu: torch.Tensor, logvar: torch.Tensor, F: Optional[torch.Tensor],       # [B,T,D], [B,T,D], [B,T,D,R] or None
        actions: torch.Tensor,                                                   # [B,T] (nle codes)
        mask: torch.Tensor                                                       # [B,T]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with per-step intrinsic bonuses (aligned to timesteps), and per-term summaries.
        """
        device = mu.device
        B, T, D = mu.shape
        A = ACTION_DIM

        # ---- HMM responsibilities rhat and (B) skill entropy ----
        rhat = self.compute_skill_rhat(mu, logvar.exp().clamp_min(1e-6), F, mask)  # [B,T,Kp1]
        Kp1 = rhat.shape[-1]
        # Entropy H[q(h_t)] only for valid steps
        with torch.no_grad():
            r_safe = torch.clamp(rhat, min=1e-8)
            h_entropy = -(r_safe * r_safe.log()).sum(-1)  # [B,T]
            h_entropy = h_entropy * mask
            # drop last residual state from features (world model uses K skills)
            skill_soft = rhat[...,:(Kp1-1)]

        # ---- (A) dynamics KL surprise using world-model prior ----
        dyn = torch.zeros(B, T, device=device)
        if self.cfg.use_dyn_kl and self.vae.world_model.enabled:
            # Prepare action one-hot and skill features; compute priors for t+1
            a_onehot = one_hot(actions.clamp(0, A-1), A)  # [B,T,A]
            # initial world state zero for each batch element
            s = self.vae.world_model.initial_state(B, device=device)
            mu_p_list, logvar_p_list, Fp_list = [], [], []
            for t in range(T-1):
                s, mu_p, logvar_p, F_p = self.vae.world_model_step(
                    s, mu[:,t,:], a_onehot[:,t,:], skill_soft[:,t,:]
                )
                mu_p_list.append(mu_p)
                logvar_p_list.append(logvar_p)
                Fp_list.append(F_p)
            # pad last step with zeros (no next obs)
            mu_p = torch.stack(mu_p_list, dim=1)                     # [B,T-1,D]
            logvar_p = torch.stack(logvar_p_list, dim=1)             # [B,T-1,D]
            F_p = torch.stack(Fp_list, dim=1) if Fp_list[0] is not None else None  # [B,T-1,D,R] or None

            # KL at t+1 between q(z_{t+1}|x_{t+1}) and p(z_{t+1}|s_t,a_t,h_t)
            mu_q = mu[:,1:,:]; logvar_q = logvar[:,1:,:]
            F_q = F[:,1:,:,:] if F is not None else None
            kl = kl_gaussian_lowrank_q_p(
                mu_q=mu_q.reshape(-1, D), logvar_q=logvar_q.reshape(-1, D), F_q=None if F_q is None else F_q.reshape(-1, D, F_q.size(-1)),
                mu_p=mu_p.reshape(-1, D), logvar_p=logvar_p.reshape(-1, D), F_p=None if F_p is None else F_p.reshape(-1, D, F_p.size(-1))
            ).view(B, T-1)
            dyn[:,:-1] = kl
            dyn = dyn * mask

        # ---- (C) RND novelty on z_t ----
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

        if self.cfg.use_skill_entropy:
            hdp_n = self.norm_hdp.update(h_entropy[mask.bool()])
            hdp_scaled = self._eta(self.cfg.eta0_hdp, self.cfg.tau_hdp) * h_entropy
            hdp_scaled[mask.bool()] = self._eta(self.cfg.eta0_hdp, self.cfg.tau_hdp) * hdp_n
            out["hdp_raw"] = h_entropy; out["hdp"] = hdp_scaled
        else:
            out["hdp_raw"] = h_entropy; out["hdp"] = torch.zeros_like(h_entropy)

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
        self.val    = torch.zeros(T, num_envs, device=device)
        self.logp   = torch.zeros(T, num_envs, device=device)
        self.skill  = torch.zeros(T, num_envs, skill_dim, device=device) if skill_dim>0 else None

    def _maybe_allocate_lowrank_factors(self, lowrank_factors_tensor):
        """Allocate lowrank_factors buffer when first lowrank_factors tensor is provided."""
        if self.lowrank_factors is None and lowrank_factors_tensor is not None:
            # Get dimensions from the provided tensor
            D, R = lowrank_factors_tensor.shape[-2], lowrank_factors_tensor.shape[-1]
            self.lowrank_factors = torch.zeros(self.T, self.num_envs, D, R, device=self.device)
            return True
        return False
        
    def add(self, **kw):
        t = self.ptr

        # Special handling for lowrank_factors - allocate buffer if needed
        if 'lowrank_factors' in kw and kw['lowrank_factors'] is not None:
            self._maybe_allocate_lowrank_factors(kw['lowrank_factors'])

        for k,v in kw.items():
            buffer = getattr(self, k)
            if buffer is None: 
                continue
            buffer[t].copy_(v)
        self.ptr += 1
        
    def full(self): return self.ptr >= self.T
    def reset(self): self.ptr = 0
    def get(self):
        # flatten T*B
        T,B = self.T, self.num_envs
        data = { "mu": self.mu, "logvar": self.logvar, "actions": self.actions, "extrinsic": self.rews_e, "dones": self.dones, "values": self.val, "logp": self.logp }
        if self.skill is not None: data["skill"] = self.skill
        if self.lowrank_factors is not None: data["lowrank_factors"] = self.lowrank_factors
        for k in ["mu","logvar","actions","extrinsic","dones","values","logp","skill","lowrank_factors"]:
            if k in data: data[k] = data[k].reshape(T*B, *data[k].shape[2:])
        return data

# --------------------------------------------------------------------------------------
# PPO trainer
# --------------------------------------------------------------------------------------

class PPOTrainer:
    def __init__(self, env_id: str, ppo_cfg: PPOConfig, cur_cfg: CuriosityConfig, hmm_cfg: HMMOnlineConfig, rnd_cfg: RNDConfig, run_cfg: TrainConfig,
                 vae: MultiModalHackVAE, hmm: StickyHDPHMMVI):
        self.env_id = env_id
        self.ppo_cfg = ppo_cfg
        self.cur_cfg = cur_cfg
        self.hmm_cfg = hmm_cfg
        self.rnd_cfg = rnd_cfg
        self.run_cfg = run_cfg
        self.device = run_cfg.device

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

        # Policy
        z_dim = vae.latent_dim
        skill_dim = hmm.niw.mu.size(0)-1 if ppo_cfg.policy_uses_skill else 0  # exclude remainder
        self.actor_critic = ActorCritic(z_dim, self.n_actions, skill_dim=skill_dim).to(self.device)
        self.opt = torch.optim.Adam(self.actor_critic.parameters(), lr=ppo_cfg.learning_rate)

        # Curiosity computer
        self.curiosity = CuriosityComputer(self.vae, self.hmm, self.device, cur_cfg, rnd_cfg, z_dim, skill_dim)

        # Storage
        self.buf = RolloutBuffer(ppo_cfg.num_envs, ppo_cfg.rollout_len, z_dim, skill_dim, self.device)

        # Bookkeeping
        self.global_steps = 0
        self.last_hmm_refresh = 0
        os.makedirs(run_cfg.log_dir, exist_ok=True)
        self._log_file = os.path.join(run_cfg.log_dir, "metrics.jsonl")
        # --- HMM filtering state (per-env) and cached E[log A] ---
        self._filt_state = [None for _ in range(ppo_cfg.num_envs)]  # StickyHDPHMMVI.FilterState or None per env
        # cache ElogA; refresh whenever HMM is updated
        self._ElogA = self.hmm._ElogA()
        
        # --- Hero info tracking for each environment ---
        self.data_collector = NetHackDataCollector()
        self._hero_info = [None for _ in range(ppo_cfg.num_envs)]  # Current hero info per env [role, race, gender, alignment]
        self._episode_start = [True for _ in range(ppo_cfg.num_envs)]  # Track if we need to parse hero info

    # --------------------------- rollout -------------------------------------

    @torch.no_grad()
    def _encode_obs(self, obs_dict: dict) -> Dict[str, torch.Tensor]:
        # Stack current hero info for all environments
        hero_info_batch = torch.stack([
            hero_info if hero_info is not None else torch.zeros(4, dtype=torch.int32)
            for hero_info in self._hero_info
        ], dim=0)  # [num_envs, 4]
        
        b = obs_to_device(obs_dict, self.device, hero_info=hero_info_batch)
        enc = self.vae.encode(b["game_chars"], b["game_colors"], b["blstats"], b["message_chars"], b["hero_info"])
        mu = enc["mu"]; logvar = enc["logvar"]; F = enc["lowrank_factors"]  # tensors [B,D], [B,D], [B,D,R] or None
        z  = mu  # use mean latent for policy input
        return {"z": z, "mu": mu, "logvar": logvar, "F": F}

    def _update_hero_info_from_obs(self, obs_dict: dict):
        """Extract hero info from the first observation of each new episode."""
        for env_idx in range(self.ppo_cfg.num_envs):
            if self._episode_start[env_idx]:
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
                        self._hero_info[env_idx] = hero_info
                        print(f"✅ Environment {env_idx}: Parsed hero info {hero_info.tolist()} from message: '{message_str[:100]}...'")
                    else:
                        # Keep previous hero info or use zeros as fallback
                        if self._hero_info[env_idx] is None:
                            self._hero_info[env_idx] = torch.zeros(4, dtype=torch.int32)
                            print(f"⚠️ Environment {env_idx}: Could not parse hero info from message: '{message_str[:100]}...'")
                    
                except Exception as e:
                    # Fallback to zeros if anything goes wrong
                    if self._hero_info[env_idx] is None:
                        self._hero_info[env_idx] = torch.zeros(4, dtype=torch.int32)
                    print(f"⚠️ Environment {env_idx}: Error parsing hero info: {e}")
                
                # Mark that we've processed the episode start
                self._episode_start[env_idx] = False

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
    def collect_rollout(self):
        self.buf.reset()
        if not hasattr(self, "_obs"):
            self._obs, _ = self.envs.reset(seed=self.run_cfg.seed)
        obs = self._obs

        # hidden world state for curiosity prior
        s_wm = self.vae.world_model.initial_state(self.ppo_cfg.num_envs, device=self.device) if self.vae.world_model.enabled else None

        for t in range(self.ppo_cfg.rollout_len):
            enc = self._encode_obs(obs)
            z = enc["z"]  # [B,D]

            # ---- Causal skill filtering (per-env, current frame) ----
            skill_feat = None
            if self.ppo_cfg.policy_uses_skill:
                B = z.size(0)
                Kp1 = int(self.hmm.niw.mu.size(0))  # includes remainder
                skill_list = []
                for b in range(B):
                    mu_b = enc["mu"][b]                                     # [D]
                    dv_b = enc["logvar"][b].exp().clamp_min(1e-6)           # [D]
                    F    = enc.get('lowrank_factors', None)                 # [D,R] or None
                    # logB_t[k] = E_q[log p(z_t | h_t=k)]
                    logB_b = StickyHDPHMMVI.expected_emission_loglik(
                        self.hmm.niw.mu, self.hmm.niw.kappa, self.hmm.niw.Psi, self.hmm.niw.nu,
                        mu_b.unsqueeze(0), dv_b.unsqueeze(0), F.unsqueeze(0) if F is not None else None, mask=None
                    ).squeeze(0)  # [Kp1]

                    st = self._filt_state[b]
                    if st is None:
                        # initialise at episode start
                        st = self.hmm.filter_init_from_logB(logB_b)
                        alpha_b = torch.exp(st.log_alpha.to(torch.float32))  # [Kp1]
                    else:
                        # one causal update (uses cached ElogA)
                        st, alpha_b, _xi, _bound, _sent = self.hmm.filter_step(st, logB_b, self._ElogA)
                    self._filt_state[b] = st
                    # drop the remainder state for policy features
                    skill_list.append(alpha_b[:Kp1-1])

                skill_feat = torch.stack(skill_list, dim=0)  # [B, K]

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
            # store
            self.buf.add(z=z, mu=enc["mu"], logvar=enc["logvar"], lowrank_factors=enc.get('lowrank_factors', None),
                         actions=torch.as_tensor(a_global, device=self.device),
                         rews_e=torch.as_tensor(rew, dtype=torch.float32, device=self.device),
                         dones=torch.as_tensor(done, dtype=torch.bool, device=self.device),
                         val=value, logp=logp,
                         skill=skill_feat if (self.buf.skill is not None) else None)
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
            self._update_hero_info_from_obs(obs)
            
            self.global_steps += self.ppo_cfg.num_envs
        self._obs = obs

    # --------------------------- compute advantages --------------------------

    @torch.no_grad()
    def _compute_intrinsic_for_buffer(self) -> Dict[str, torch.Tensor]:
        T, B = self.ppo_cfg.rollout_len, self.ppo_cfg.num_envs
        mu     = self.buf.mu.transpose(0,1)      # [B,T,D]
        logvar = self.buf.logvar.transpose(0,1)  # [B,T,D]
        lowrank_factors = self.buf.lowrank_factors.transpose(0,1) if self.buf.lowrank_factors is not None else None  # [B,T,D,R] or None
        actions= self.buf.actions.transpose(0,1) # [B,T]
        mask   = (~self.buf.dones).transpose(0,1).float() # [B,T], 1 for valid
        # curvature: last step after done is invalid; mask handled
        bonuses = self.curiosity.compute_intrinsic(mu, logvar, lowrank_factors, actions, mask)
        self.curiosity.global_step = self.global_steps
        return bonuses

    @torch.no_grad()
    def _compute_skill_features(self, enc):
        """Helper to compute skill features for a single batch of observations."""
        if not self.ppo_cfg.policy_uses_skill:
            return None
            
        B = enc["z"].size(0)
        Kp1 = self.hmm.niw.mu.size(0)
        skill_list = []
        
        for b in range(B):
            mu_b = enc["mu"][b]
            dv_b = enc["logvar"][b].exp()
            
            if hasattr(self.hmm, "emission_logB_one"):
                logB_b = self.hmm.emission_logB_one(mu_b, dv_b, None)
            else:
                logB_b = StickyHDPHMMVI.expected_emission_loglik(
                    self.hmm.niw.mu, self.hmm.niw.kappa, self.hmm.niw.Psi, self.hmm.niw.nu,
                    mu_b.unsqueeze(0), dv_b.unsqueeze(0), None, mask=None
                ).squeeze(0)
            
            st = self._filt_state[b]
            if st is None:
                if hasattr(self.hmm, "filter_init_from_logB"):
                    st = self.hmm.filter_init_from_logB(logB_b)
                    alpha_b = torch.exp(st.log_alpha.to(torch.float32))
                else:
                    # Fallback for older HMM classes
                    alpha_b = torch.softmax(logB_b, dim=-1)
            else:
                st, alpha_b, _, _, _ = self.hmm.filter_step(st, logB_b, self._ElogA)
            
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
        enc = self._encode_obs(self._obs)
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

    @torch.no_grad()
    def maybe_refresh_hmm(self, replay_mu: torch.Tensor, replay_logvar: torch.Tensor, replay_mask: torch.Tensor):
        if (self.global_steps - self.last_hmm_refresh) < self.hmm_cfg.hmm_update_every:
            return
        B,T,D = replay_mu.shape
        # run full VI update on the window
        out = self.hmm.update(
            mu_t=replay_mu, diag_var_t=replay_logvar.exp(), F_t=None, mask=replay_mask,
            max_iters=self.hmm_cfg.hmm_max_iters, tol=self.hmm_cfg.hmm_tol, elbo_drop_tol=self.hmm_cfg.hmm_elbo_drop_tol,
            rho=self.hmm_cfg.rho_emission, optimize_pi=self.hmm_cfg.optimise_pi, offline=False
        )
        # refresh cached ElogA used by the online filter
        self._ElogA = self.hmm._ElogA()
        if self.hmm_cfg.reset_low_count_states is not None:
            self.hmm.reset_low_count_states(self.hmm_cfg.reset_low_count_states)
        self.last_hmm_refresh = self.global_steps

    # --------------------------- main train loop -----------------------------

    def train(self):
        set_seed(self.run_cfg.seed)
        obs, _ = self.envs.reset(seed=self.run_cfg.seed)
        
        # Parse hero info from initial observations (episode start)
        self._episode_start = [True for _ in range(self.ppo_cfg.num_envs)]
        self._update_hero_info_from_obs(obs)
        
        self._obs = obs

        # maintain a growing replay window of latents for HMM refresh
        replay_mu, replay_logvar, replay_mask = None, None, None

        while self.global_steps < self.ppo_cfg.total_updates * self.ppo_cfg.rollout_len * self.ppo_cfg.num_envs:
            self.collect_rollout()
            bonuses = self._compute_intrinsic_for_buffer()
            # skills for policy (concat to z) during PPO update: use the SAME features we acted with
            skills_for_policy = self.buf.skill.transpose(0,1) if (self.ppo_cfg.policy_uses_skill and self.buf.skill is not None) else None  # [T,B,K]
            # total reward
            intrinsic = bonuses["dyn"] + bonuses["hdp"] + bonuses["rnd"]
            ext = self.buf.rews_e
            rews_total = (ext + intrinsic).transpose(0,1)  # [T,B]
            adv, ret = self._advantages(rews_total, self.buf.val.transpose(0,1), self.buf.dones.transpose(0,1))
            self._ppo_update(adv, ret, skills_for_policy)

            # optional RND predictor update to keep error scale meaningful
            if self.curiosity.use_rnd:
                mu_flat = self.buf.mu.reshape(-1, self.buf.mu.size(-1)).detach()
                for _ in range(self.rnd_cfg.update_per_rollout):
                    self.curiosity.train_rnd(mu_flat)

            # append to replay window for HMM refresh
            with torch.no_grad():
                mu_bt     = self.buf.mu.transpose(0,1)     # [B,T,D]
                logvar_bt = self.buf.logvar.transpose(0,1) # [B,T,D]
                mask_bt   = (~self.buf.dones).transpose(0,1).float()  # [B,T]
                if replay_mu is None:
                    replay_mu, replay_logvar, replay_mask = mu_bt, logvar_bt, mask_bt
                else:
                    # concatenate, then crop to window
                    replay_mu = torch.cat([replay_mu, mu_bt], dim=1)
                    replay_logvar = torch.cat([replay_logvar, logvar_bt], dim=1)
                    replay_mask = torch.cat([replay_mask, mask_bt], dim=1)
                    if replay_mu.size(1) > self.hmm_cfg.hmm_fit_window:
                        s = replay_mu.size(1) - self.hmm_cfg.hmm_fit_window
                        replay_mu = replay_mu[:, s:, :]
                        replay_logvar = replay_logvar[:, s:, :]
                        replay_mask = replay_mask[:, s:]

            self.maybe_refresh_hmm(replay_mu, replay_logvar, replay_mask)

            # logging
            self._log_scalar({
                "steps": self.global_steps,
                "return/mean_ext": float(ext.mean().item()),
                "int/dyn_mean": float(bonuses["dyn"].mean().item()),
                "int/hdp_mean": float(bonuses["hdp"].mean().item()),
                "int/rnd_mean": float(bonuses["rnd"].mean().item()),
            })
            if (self.global_steps % self.run_cfg.eval_every) < (self.ppo_cfg.num_envs * self.ppo_cfg.rollout_len):
                self.evaluate(self.run_cfg.eval_episodes)
            if (self.global_steps % self.run_cfg.save_every) < (self.ppo_cfg.num_envs * self.ppo_cfg.rollout_len):
                self._save_ckpt()

    @torch.no_grad()
    def evaluate(self, episodes: int):
        env = gym.make(self.env_id)
        total = []
        for _ in range(episodes):
            o, _ = env.reset()
            # local filter state for single env evaluation
            fstate = None
            done = False; ret = 0.0
            while not done:
                enc = self._encode_obs(o)
                skill_feat = None
                if self.ppo_cfg.policy_uses_skill:
                    mu = enc["mu"].squeeze(0)             # [D]
                    dv = enc["logvar"].squeeze(0).exp()   # [D]
                    if hasattr(self.hmm, "emission_logB_one"):
                        logB = self.hmm.emission_logB_one(mu, dv, None)
                    else:
                        logB = StickyHDPHMMVI.expected_emission_loglik(
                            self.hmm.niw.mu, self.hmm.niw.kappa, self.hmm.niw.Psi, self.hmm.niw.nu,
                            mu.unsqueeze(0), dv.unsqueeze(0), None, mask=None
                        ).squeeze(0)
                    if fstate is None:
                        fstate = self.hmm.filter_init_from_logB(logB) if hasattr(self.hmm, "filter_init_from_logB") else None
                        alpha = torch.exp(fstate.log_alpha.to(torch.float32)) if fstate is not None else torch.softmax(logB, dim=-1)
                    else:
                        fstate, alpha, _xi, _bound, _sent = self.hmm.filter_step(fstate, logB, self._ElogA)
                    # drop remainder state
                    skill_feat = alpha[: self.hmm.niw.mu.size(0)-1].unsqueeze(0)

                logits, value = self.actor_critic(enc["z"], skill_feat)
                a = torch.argmax(logits, dim=-1) if self.ppo_cfg.deterministic_eval else torch.distributions.Categorical(logits=logits).sample()
                o, r, term, trunc, _ = env.step(int(a.item()))
                done = term or trunc
                ret += r
                if done:
                    fstate = None
            total.append(ret)
        self._log_scalar({"eval/return_mean": float(np.mean(total)), "eval/return_std": float(np.std(total))})

    def _log_scalar(self, d: Dict[str, float]):
        with open(self._log_file, "a") as f:
            f.write(json.dumps(d) + "\n")

    def _save_ckpt(self):
        path = os.path.join(self.run_cfg.log_dir, f"ckpt_{self.global_steps}.pt")
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "opt": self.opt.state_dict(),
            "hmm": self.hmm.get_posterior_params()
        }, path)

# --------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------

def main():
    run = TrainConfig()
    set_seed(run.seed)

    # Build VAE (frozen during RL) and HMM
    vae_cfg = VAEConfig()
    vae = MultiModalHackVAE(vae_cfg).to(run.device).eval()

    # HMM params aligned to VAE latent dim and desired K
    K = 16
    assert vae_cfg.skill_num in (0, K), f"Set VAEConfig.skill_num={K} to feed skills to world model"
    D = vae_cfg.latent_dim
    niw = NIWPrior(
        mu0=torch.zeros(D, device=run.device),
        kappa0=1.0,
        Psi0=torch.eye(D, device=run.device),
        nu0=D + 2.0
    )
    hmm = StickyHDPHMMVI(
        StickyHDPHMMParams(alpha=4.0, kappa=4.0, gamma=1.0, K=K, D=D, device=run.device, dtype=torch.float32),
        niw_prior=niw, rho_emission=0.05, rho_transition=None
    ).to(run.device)

    # Configs
    ppo_cfg = PPOConfig()
    cur_cfg = CuriosityConfig(use_dyn_kl=True, use_skill_entropy=True, use_rnd=False)
    hmm_cfg = HMMOnlineConfig()
    rnd_cfg = RNDConfig()

    trainer = PPOTrainer(run.env_id, ppo_cfg, cur_cfg, hmm_cfg, rnd_cfg, run, vae, hmm)
    trainer.train()

if __name__ == "__main__":
    main()

# model.py — Decision‑sufficient, predictive VAE for NetHack screens
# ---------------------------------------------------------------
# This rewrite focuses on:
#   • A control‑useful representation z (decision‑sufficient, predictive)
#   • Ego‑centric semantic map head (k×k around the hero)
#   • 8‑direction passability/legality head
#   • Reward / done / short‑horizon value heads
#   • Optional bag head (with proper gradient flow + normalized loss)
#   • BL stats head
#   • Action‑conditional latent forward model z_{t+1} ≈ f(z_t, a_t)
#   • Flexible KL prior: standard normal, (placeholder) sticky HDP‑HMM mixture, or a blend
#
# NOT included on purpose (can be added back later if needed):
#   • Heavy global map reconstruction heads (common/rare char/color)
#   • Huge hero_loc supervision (we prefer ego‑centric supervision instead)
#   • TV/Dice regularizers that fight sharp edges in maps
#   • Actual HDP‑HMM implementation (left as a pluggable prior interface)
#
# Expected batch keys (you can adapt your train.py to supply any subset):
#   inputs:
#     - obs: Tensor[B, C, H, W]      # encoded screen input (e.g., glyph/color embeddings or RGB)
#     - blstats: Tensor[B, S]        # numeric hero stats (optionally preprocessed)
#     - hero_xy: Tensor[B, 2]        # hero position in map coords, normalized to [-1, 1]
#     - action_onehot: Tensor[B, A]  # (optional) for forward model
#     - z_next_detached: Tensor[B, Z]# (optional) target latent for forward model (no grad)
#     - skills: Tensor[B, K]         # (optional) soft skill posteriors q(h_t) for HMM prior
#   targets:
#     - ego_sem_target: Long[B, k, k]          # ego‑centric semantic classes
#     - passability_target: Float[B, 8]        # 8‑dir legality/safety bits
#     - reward_target: Float[B]                # scalar reward
#     - done_target: Float[B]                  # episode termination (0/1)
#     - value_k_target: Float[B, M]            # M horizons (e.g., k∈{5,10})
#     - bag_target: Float[B, P]                # optional bag of pairs/classes (0/1)
#     - stats_target: Float[B, S]              # normalized stats target
#
# All heads are optional at training time; the loss only uses provided targets.
#
# Author: (rewritten by ChatGPT)

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Small building blocks
# ----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupNormConv(nn.Module):
    """Conv2d + GroupNorm + ReLU. Safer than BatchNorm for small batch RL.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, groups: int = 8):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


# ----------------------------
# Config
# ----------------------------

@dataclass
class VAEConfig:
    input_channels: int = 8           # channels of preprocessed screen
    blstats_dim: int = 0              # optional concatenated dims; 0 to disable
    latent_dim: int = 128

    # Encoder/decoder
    enc_channels: Tuple[int, int, int] = (64, 128, 256)
    dec_channels: Tuple[int, int] = (128, 64)

    # Ego‑centric semantics
    ego_window: int = 11              # k×k grid; should be odd
    ego_classes: int = 10             # number of semantic classes

    # Passability / legality
    passability_dirs: int = 8         # N, S, E, W, NE, NW, SE, SW

    # Value heads
    value_horizons: Tuple[int, ...] = (5, 10)  # used only for naming/logging

    # Bag head
    bag_dim: int = 0                  # 0 to disable bag head

    # Stats head
    stats_dim: int = 0                # 0 to disable stats head

    # Forward model
    action_dim: int = 0               # one‑hot action size; 0 to disable
    forward_hidden: int = 256

    # KL prior mode
    prior_mode: str = "standard"      # "standard" | "hmm" | "blend"
    prior_blend_alpha: float = 0.5    # for blend: alpha * standard + (1‑alpha) * hmm

    # Loss weights (can be overridden in train.py)
    raw_modality_weights: Dict[str, float] = field(default_factory=lambda: {
        'ego_sem': 3.0,
        'passability': 1.5,
        'reward': 1.0,
        'done': 1.0,
        'value': 1.0,
        'bag': 8.0,
        'stats': 0.5,
        'forward': 1.0,
    })

    # KL settings
    beta: float = 1.0                 # treat as cap or scalar; trainer can do capacity scheduling


# ----------------------------
# Prior modules
# ----------------------------

class StandardNormalPrior(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x)=N(mu,diag(exp(logvar))) || N(0,I)). Returns per‑sample KL [B]."""
        # KL = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
        kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
        return kl.sum(dim=1)


class StickyHMMPriorPlaceholder(nn.Module):
    """Placeholder mixture prior for future sticky HDP‑HMM.
    Assumes we are given soft skill posteriors s = q(h_t) of shape [B, K], and
    per‑skill Gaussian params (mu_k, logvar_k). In this placeholder we expose a
    simple mixture‑of‑Gaussians prior for z.
    """
    def __init__(self, latent_dim: int, num_skills: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_skills = num_skills
        if num_skills > 0:
            self.mu = nn.Parameter(torch.zeros(num_skills, latent_dim))
            self.logvar = nn.Parameter(torch.zeros(num_skills, latent_dim))
        else:
            self.register_parameter('mu', None)
            self.register_parameter('logvar', None)

    def forward(self, mu_q: torch.Tensor, logvar_q: torch.Tensor, skill_probs: Optional[torch.Tensor]) -> torch.Tensor:
        """KL(q(z) || p(z|h)) marginalized over h with q(h)=skill_probs.
        If skill_probs is None or num_skills==0, returns zeros (caller should fall back).
        Returns per‑sample KL [B].
        """
        B, D = mu_q.shape
        if self.mu is None or skill_probs is None:
            return torch.zeros(B, device=mu_q.device, dtype=mu_q.dtype)
        # Compute per‑skill KLs: KL(q || p_k)
        # KL(N(m1,S1)||N(m2,S2)) = 0.5 * [ tr(S2^{-1} S1) + (m2-m1)^T S2^{-1} (m2-m1) - D + log|S2| - log|S1| ]
        # Here S are diagonal.
        q_var = torch.exp(logvar_q)          # [B,D]
        P_var = torch.exp(self.logvar)       # [K,D]
        inv_P_var = torch.exp(-self.logvar)  # [K,D]
        # term1: tr(S2^{-1} S1)
        term1 = torch.einsum('kd,bd->bk', inv_P_var, q_var)  # [B,K]
        # term2: (m2-m1)^T S2^{-1} (m2-m1)
        diff2 = (self.mu.unsqueeze(0) - mu_q.unsqueeze(1)).pow(2)    # [B,K,D]
        term2 = torch.einsum('bkd,kd->bk', diff2, inv_P_var)         # [B,K]
        # term3: log|S2| - log|S1|
        term3 = torch.sum(self.logvar, dim=1).unsqueeze(0) - torch.sum(logvar_q, dim=1, keepdim=True)  # [B,1]
        kl_bk = 0.5 * (term1 + term2 - D + term3)  # [B,K]
        # Marginalize over q(h)
        kl = (skill_probs * kl_bk).sum(dim=1)      # [B]
        return kl


# ----------------------------
# Encoder / Decoder
# ----------------------------

class ScreenEncoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        C = cfg.input_channels
        ch = cfg.enc_channels
        self.backbone = nn.Sequential(
            GroupNormConv(C, ch[0], 5, 2),
            GroupNormConv(ch[0], ch[1], 3, 2),
            GroupNormConv(ch[1], ch[2], 3, 2),
            nn.AdaptiveAvgPool2d(1),  # [B, ch[-1], 1,1]
        )
        in_lin = ch[-1] + cfg.blstats_dim + 2  # + hero_xy (2)
        self.proj = MLP(in_lin, hidden=256, out_dim=256, num_layers=2)
        self.mu = nn.Linear(256, cfg.latent_dim)
        self.logvar = nn.Linear(256, cfg.latent_dim)

    def forward(self, obs: torch.Tensor, blstats: Optional[torch.Tensor], hero_xy: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.backbone(obs).flatten(1)
        feats = [x]
        if blstats is not None and blstats.numel() > 0:
            feats.append(blstats)
        if hero_xy is not None and hero_xy.numel() > 0:
            feats.append(hero_xy)
        h = torch.cat(feats, dim=1)
        h = self.proj(h)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z


class Heads(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        Z = cfg.latent_dim
        H = 256
        # Shared trunk for heads
        self.trunk = MLP(Z, H, H, num_layers=2)
        # Ego‑centric k×k semantics
        k = cfg.ego_window
        C_sem = cfg.ego_classes
        self.ego_sem = nn.Linear(H, k * k * C_sem)
        # 8‑direction passability/legality
        self.passability = nn.Linear(H, cfg.passability_dirs)
        # Reward, done, values@k
        self.reward = nn.Linear(H, 1)
        self.done = nn.Linear(H, 1)
        self.value = nn.Linear(H, len(cfg.value_horizons))
        # Bag head (optional)
        if cfg.bag_dim > 0:
            self.bag = MLP(H, H, cfg.bag_dim, num_layers=2)
        else:
            self.bag = None
        # Stats head (optional)
        if cfg.stats_dim > 0:
            self.stats = MLP(H, H, cfg.stats_dim, num_layers=2)
        else:
            self.stats = None

    def forward(self, z: torch.Tensor, cfg: VAEConfig) -> Dict[str, torch.Tensor]:
        h = F.relu(self.trunk(z))
        out: Dict[str, torch.Tensor] = {}
        k = cfg.ego_window
        C_sem = cfg.ego_classes
        ego_logits = self.ego_sem(h).view(-1, C_sem, k, k)  # [B, C_sem, k, k]
        out['ego_sem_logits'] = ego_logits
        out['passability_logits'] = self.passability(h)     # [B, 8]
        out['reward'] = self.reward(h).squeeze(-1)          # [B]
        out['done_logit'] = self.done(h).squeeze(-1)        # [B]
        out['value'] = self.value(h)                        # [B, M]
        if self.bag is not None:
            out['bag_logits'] = self.bag(h)                # [B, P]
        if self.stats is not None:
            out['stats_pred'] = self.stats(h)              # [B, S]
        return out


class LatentForwardModel(nn.Module):
    """Action‑conditional latent dynamics: z_{t+1} ≈ f(z_t, a_t)."""
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.enabled = cfg.action_dim > 0
        if not self.enabled:
            return
        self.net = MLP(cfg.latent_dim + cfg.action_dim, cfg.forward_hidden, cfg.latent_dim, num_layers=3)

    def forward(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("Forward model disabled (action_dim=0).")
        x = torch.cat([z, action_onehot], dim=1)
        return self.net(x)


# ----------------------------
# Main model
# ----------------------------

class DecisionVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ScreenEncoder(cfg)
        self.heads = Heads(cfg)
        self.fwd = LatentForwardModel(cfg)
        self.std_prior = StandardNormalPrior()
        # Placeholder HMM prior (mixture of Gaussians); set num_skills>0 to enable
        self.hmm_prior = StickyHMMPriorPlaceholder(cfg.latent_dim, num_skills=0)

        # Flags
        self.detach_bag_prior = True  # if you later inject bag priors into pixel heads

    # ------------------------
    # Forward / encode only
    # ------------------------
    def encode(self, obs: torch.Tensor, blstats: Optional[torch.Tensor] = None, hero_xy: Optional[torch.Tensor] = None):
        return self.encoder(obs, blstats, hero_xy)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = batch['obs']
        blstats = batch.get('blstats', None)
        hero_xy = batch.get('hero_xy', None)
        mu, logvar, z = self.encode(obs, blstats, hero_xy)
        out = {'z_mu': mu, 'z_logvar': logvar, 'z': z}
        out.update(self.heads(z, self.cfg))
        if self.fwd.enabled and ('action_onehot' in batch):
            out['z_next_pred'] = self.fwd(z, batch['action_onehot'])
        return out

    # ------------------------
    # Loss computation
    # ------------------------
    def compute_losses(self, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor],
                       modality_weights: Optional[Dict[str, float]] = None,
                       prior_mode: Optional[str] = None,
                       prior_blend_alpha: Optional[float] = None,
                       beta: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        w = modality_weights or cfg.raw_modality_weights
        prior_mode = prior_mode or cfg.prior_mode
        alpha = cfg.prior_blend_alpha if prior_blend_alpha is None else prior_blend_alpha
        beta = cfg.beta if beta is None else beta

        losses: Dict[str, torch.Tensor] = {}
        B = out['z'].size(0)

        # 1) Reconstruction / auxiliary heads
        # Ego‑centric semantics (per‑cell CE)
        if 'ego_sem_target' in batch:
            ego_logits = out['ego_sem_logits']  # [B, C, k, k]
            target = batch['ego_sem_target'].long()  # [B, k, k]
            ce = F.cross_entropy(ego_logits, target, reduction='mean')
            losses['ego_sem'] = ce * w.get('ego_sem', 0.0)

        # Passability / legality (8 BCE)
        if 'passability_target' in batch:
            logits = out['passability_logits']      # [B, 8]
            target = batch['passability_target']    # [B, 8]
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
            losses['passability'] = bce * w.get('passability', 0.0)

        # Reward / done / value@k
        if 'reward_target' in batch:
            pred = out['reward']
            mse = F.mse_loss(pred, batch['reward_target'], reduction='mean')
            losses['reward'] = mse * w.get('reward', 0.0)
        if 'done_target' in batch:
            bce = F.binary_cross_entropy_with_logits(out['done_logit'], batch['done_target'], reduction='mean')
            losses['done'] = bce * w.get('done', 0.0)
        if 'value_k_target' in batch:
            pred = out['value']
            target = batch['value_k_target']
            mse = F.mse_loss(pred, target, reduction='mean')
            losses['value'] = mse * w.get('value', 0.0)

        # Bag head (normalized BCE with class weights; ensure gradients flow)
        if self.heads.bag is not None and 'bag_target' in batch:
            logits = out['bag_logits']            # [B, P]
            target = batch['bag_target']          # [B, P], 0/1
            # Optional per‑class weights can be provided as 'bag_weight' (same shape)
            weight = batch.get('bag_weight', torch.ones_like(target))
            per_class = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction='none')  # [B,P]
            denom = weight.sum(dim=1).clamp_min(1.0)
            bag_loss = (per_class.sum(dim=1) / denom).mean()
            losses['bag'] = bag_loss * w.get('bag', 0.0)

        # Stats head (MSE)
        if self.heads.stats is not None and 'stats_target' in batch:
            pred = out['stats_pred']
            target = batch['stats_target']
            mse = F.mse_loss(pred, target, reduction='mean')
            losses['stats'] = mse * w.get('stats', 0.0)

        # Forward model z_{t+1}
        if self.fwd.enabled and ('z_next_detached' in batch) and ('action_onehot' in batch) and ('z_next_pred' in out):
            z_next_pred = out['z_next_pred']
            z_next_tgt = batch['z_next_detached']  # should be detached by caller
            # Cosine + MSE combo for stability
            mse = F.mse_loss(z_next_pred, z_next_tgt, reduction='mean')
            cos = 1.0 - F.cosine_similarity(z_next_pred, z_next_tgt, dim=1).mean()
            forward_loss = 0.5 * (mse + cos)
            losses['forward'] = forward_loss * w.get('forward', 0.0)

        # 2) KL term (prior_mode: standard | hmm | blend)
        mu, logvar = out['z_mu'], out['z_logvar']
        kl_std = self.std_prior(mu, logvar)

        if prior_mode == 'standard':
            kl = kl_std
        elif prior_mode == 'hmm':
            kl_hmm = self.hmm_prior(mu, logvar, batch.get('skills'))
            # If HMM prior disabled/unavailable, fall back to std
            kl = torch.where(kl_hmm > 0, kl_hmm, kl_std)
        elif prior_mode == 'blend':
            kl_hmm = self.hmm_prior(mu, logvar, batch.get('skills'))
            # If hmm prior not active, reduce to standard
            if self.hmm_prior.mu is None or batch.get('skills') is None:
                kl = kl_std
            else:
                kl = alpha * kl_std + (1.0 - alpha) * kl_hmm
        else:
            raise ValueError(f"Unknown prior_mode: {prior_mode}")

        kl_loss = (kl.mean()) * beta
        losses['kl'] = kl_loss

        # Total
        total = sum(losses.values()) if len(losses) > 0 else kl_loss
        return total, losses


# ----------------------------
# Convenience factory
# ----------------------------

def build_model(**kwargs) -> DecisionVAE:
    cfg = VAEConfig(**kwargs)
    return DecisionVAE(cfg)

# model.py — Decision‑sufficient, predictive VAE for NetHack (skill‑prior ready)
# ----------------------------------------------------------------------------
# This implementation focuses on control‑useful supervision instead of full map
# reconstruction. It supports toggleable heads and a pluggable KL prior:
#   • Ego‑centric semantic classes (k×k around the hero)
#   • 8‑direction passability / legality
#   • Reward / Done / Value@k
#   • Bag of (glyph_char, glyph_color) pairs (global presence)
#   • BL stats head
#   • Latent forward model  z_{t+1} ≈ f(z_t, a_t)
#   • (Optional) Inverse dynamics  a_t ≈ g(z_t, z_{t+1})
#   • (Optional) Ego semantic bits (masked multi‑label refinements)
#   • (Optional) 8‑direction safety risk
#   • (Optional) Goal direction / distance (tiny global summary)
#   • (Optional) Low‑res global occupancy
#   • (Optional) Ego char/color heads (tiny weight safety‑net)
#   • (Optional) Skill belief logits (filtered) for future sticky HDP‑HMM
#   • KL prior: standard normal, placeholder HMM mixture, or a blend
#
# Observation input can be either:
#   (A) Float tensor [B,C,H,W] already embedded, or
#   (B) Dict with 'glyph': Long[B,H,W] and optional 'color': Long[B,H,W], in which
#       case a learnable ObsEmbedder converts to float channels before the CNN.
#
# Author: (rewritten by ChatGPT)

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# Small building blocks
# ============================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupNormConv(nn.Module):
    """Conv2d + GroupNorm + ReLU. Preferable to BatchNorm for small RL batches."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, groups: int = 8):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ObsEmbedder(nn.Module):
    """Embed discrete glyph/color grids to dense feature maps for the CNN.
    Expects glyph ids remapped so 0 == unknown/out‑of‑sight (padding_idx=0).
    """
    def __init__(self,
                 glyph_vocab: Optional[int], color_vocab: Optional[int],
                 glyph_dim: int = 24, color_dim: int = 8,
                 out_ch: int = 32, padding_idx: int = 0):
        super().__init__()
        self.enabled = glyph_vocab is not None
        self.use_color = self.enabled and (color_vocab is not None)
        if self.enabled:
            self.glyph_emb = nn.Embedding(glyph_vocab, glyph_dim, padding_idx=padding_idx)
            if self.use_color:
                self.color_emb = nn.Embedding(color_vocab, color_dim, padding_idx=padding_idx)
                in_ch = glyph_dim + color_dim
            else:
                self.color_emb = None
                in_ch = glyph_dim
            self.to_channels = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.glyph_emb = None
            self.color_emb = None
            self.to_channels = None

    def forward(self, obs):
        if not self.enabled:
            raise RuntimeError("ObsEmbedder called but glyph_vocab=None (disabled).")
        glyph = obs['glyph'].long()  # [B,H,W]
        g = self.glyph_emb(glyph)    # [B,H,W,Dg]
        if self.use_color and ('color' in obs) and (obs['color'] is not None):
            color = obs['color'].long()
            c = self.color_emb(color)  # [B,H,W,Dc]
            x = torch.cat([g, c], dim=-1)
        else:
            x = g
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
        return self.to_channels(x)               # [B, out_ch, H, W]


# ============================
# Config
# ============================

@dataclass
class VAEConfig:
    # Input pathway
    input_channels: int = 32                   # CNN stem channels if obs already float
    glyph_vocab: Optional[int] = None          # set to enable embedding path
    color_vocab: Optional[int] = None
    glyph_dim: int = 24
    color_dim: int = 8

    # Latent
    latent_dim: int = 128

    # Encoder/decoder stem
    enc_channels: Tuple[int, int, int] = (64, 128, 256)

    # Hero/context
    blstats_dim: int = 0
    use_hero_xy: bool = True

    # Ego‑centric semantics
    ego_window: int = 11                       # must be odd
    ego_classes: int = 12
    ego_bits: Tuple[str, ...] = ()             # e.g., ('has_item','door_open','monster_danger_high')

    # Passability / safety (8 directions)
    passability_dirs: int = 8

    # Value heads
    value_horizons: Tuple[int, ...] = (5, 10)

    # Optional global/context heads
    bag_dim: int = 0                           # 0 disables bag head
    stats_dim: int = 0                         # 0 disables stats head
    goal_dir_dim: int = 0                      # 0 disables (set 2 for (dx,dy) regression)
    lowres_occ_shape: Tuple[int, int] = (0, 0) # (h', w'), (0,0) disables

    # Optional ego safety‑net heads
    ego_char_vocab: int = 0                    # 0 disables
    ego_color_vocab: int = 0                   # 0 disables

    # Dynamics heads
    action_dim: int = 0                        # one‑hot; 0 disables forward/inverse
    forward_hidden: int = 256
    use_inverse_dynamics: bool = True

    # Skill belief (future sticky HMM integration)
    skill_num: int = 0                         # number of skills; 0 disables skill head

    # KL prior mode
    prior_mode: str = "standard"               # "standard" | "hmm" | "blend"
    prior_blend_alpha: float = 0.5

    # Loss weights
    raw_modality_weights: Dict[str, float] = field(default_factory=lambda: {
        'ego_sem': 3.5,
        'ego_bits': 1.0,
        'passability': 1.8,
        'safety': 0.8,
        'reward': 1.0,
        'done': 1.0,
        'value': 1.0,
        'bag': 8.0,
        'stats': 0.5,
        'goal': 1.0,
        'lowres_occ': 0.3,
        'forward': 1.0,
        'inverse': 0.8,
        'ego_char': 0.3,
        'ego_color': 0.3,
    })

    # KL settings
    beta: float = 1.0                          # trainer can do capacity scheduling externally


# ============================
# Priors
# ============================

class StandardNormalPrior(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar ) per sample
        kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
        return kl.sum(dim=1)


class StickyHMMPriorPlaceholder(nn.Module):
    """Mixture‑of‑Gaussians placeholder for a future sticky HDP‑HMM prior.
    Expects soft skill posteriors s = q(h_t) [B,K]. If not provided, returns zeros
    (caller falls back to standard normal). Emission params are learnable here.
    """
    def __init__(self, latent_dim: int, num_skills: int = 0):
        super().__init__()
        self.num_skills = num_skills
        if num_skills > 0:
            self.mu = nn.Parameter(torch.zeros(num_skills, latent_dim))
            self.logvar = nn.Parameter(torch.zeros(num_skills, latent_dim))
        else:
            self.register_parameter('mu', None)
            self.register_parameter('logvar', None)

    def forward(self, mu_q: torch.Tensor, logvar_q: torch.Tensor, skill_probs: Optional[torch.Tensor]) -> torch.Tensor:
        B, D = mu_q.shape
        if (self.mu is None) or (skill_probs is None):
            return torch.zeros(B, device=mu_q.device, dtype=mu_q.dtype)
        q_var = torch.exp(logvar_q)                 # [B,D]
        P_var = torch.exp(self.logvar)              # [K,D]
        inv_P_var = torch.exp(-self.logvar)         # [K,D]
        term1 = torch.einsum('kd,bd->bk', inv_P_var, q_var)
        diff2 = (self.mu.unsqueeze(0) - mu_q.unsqueeze(1)).pow(2)
        term2 = torch.einsum('bkd,kd->bk', diff2, inv_P_var)
        term3 = torch.sum(self.logvar, dim=1).unsqueeze(0) - torch.sum(logvar_q, dim=1, keepdim=True)
        kl_bk = 0.5 * (term1 + term2 - D + term3)
        kl = (skill_probs * kl_bk).sum(dim=1)
        return kl


# ============================
# Encoder / Heads / Dynamics
# ============================

class ScreenEncoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        # Optional embedding path
        self.embedder = None
        if cfg.glyph_vocab is not None:
            self.embedder = ObsEmbedder(cfg.glyph_vocab, cfg.color_vocab, cfg.glyph_dim, cfg.color_dim, cfg.input_channels)

        C = cfg.input_channels
        ch = cfg.enc_channels
        self.backbone = nn.Sequential(
            GroupNormConv(C, ch[0], 5, 2),
            GroupNormConv(ch[0], ch[1], 3, 2),
            GroupNormConv(ch[1], ch[2], 3, 2),
            nn.AdaptiveAvgPool2d(1),  # [B, ch[-1], 1, 1]
        )
        extra = (cfg.blstats_dim if cfg.blstats_dim > 0 else 0) + (2 if cfg.use_hero_xy else 0)
        self.proj = MLP(ch[-1] + extra, 256, 256, num_layers=2)
        self.mu = nn.Linear(256, cfg.latent_dim)
        self.logvar = nn.Linear(256, cfg.latent_dim)

    def forward(self, obs, blstats: Optional[torch.Tensor], hero_xy: Optional[torch.Tensor]):
        if self.embedder is not None and isinstance(obs, dict):
            x_img = self.embedder(obs)
        else:
            x_img = obs  # expect float [B,C,H,W]
        x = self.backbone(x_img).flatten(1)
        feats = [x]
        if blstats is not None and blstats.numel() > 0:
            feats.append(blstats)
        if hero_xy is not None and hero_xy.numel() > 0:
            feats.append(hero_xy)
        h = self.proj(torch.cat(feats, dim=1))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return mu, logvar, z


class Heads(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        Z = cfg.latent_dim
        H = 256
        self.trunk = MLP(Z, H, H, num_layers=2)
        k = cfg.ego_window
        C_sem = cfg.ego_classes
        nb = len(cfg.ego_bits)

        # Core heads
        # self.ego_sem = nn.Linear(H, k * k * C_sem)
        # self.passability = nn.Linear(H, cfg.passability_dirs)
        # self.reward = nn.Linear(H, 1)
        # self.done = nn.Linear(H, 1)
        # self.value = nn.Linear(H, len(cfg.value_horizons))

        # Optional/context heads
        # self.bag = MLP(H, H, cfg.bag_dim, num_layers=2) if cfg.bag_dim > 0 else None
        # self.stats = MLP(H, H, cfg.stats_dim, num_layers=2) if cfg.stats_dim > 0 else None

        # Optional ego bits
        # self.ego_bits = nn.Linear(H, nb * k * k) if nb > 0 else None

        # Optional safety risk (8 dirs)
        # self.safety = nn.Linear(H, cfg.passability_dirs)

        # Goal direction / distance
        # self.goal = nn.Linear(H, cfg.goal_dir_dim) if cfg.goal_dir_dim > 0 else None

        # Low‑res global occupancy
        # hpr, wpr = cfg.lowres_occ_shape
        # self.lowres_occ = nn.Linear(H, hpr * wpr) if (hpr > 0 and wpr > 0) else None
        # self.lowres_shape = (hpr, wpr)

        # Ego exact char/color (safety‑net)
        # self.ego_char = nn.Linear(H, k * k * cfg.ego_char_vocab) if cfg.ego_char_vocab > 0 else None
        # self.ego_color = nn.Linear(H, k * k * cfg.ego_color_vocab) if cfg.ego_color_vocab > 0 else None

        # Skill belief (filtered)
        # self.skill_belief = nn.Linear(H, cfg.skill_num) if cfg.skill_num > 0 else None

    def forward(self, z: torch.Tensor, cfg: VAEConfig) -> Dict[str, torch.Tensor]:
        h = F.relu(self.trunk(z))
        out: Dict[str, torch.Tensor] = {}
        k = cfg.ego_window
        C_sem = cfg.ego_classes
        nb = len(cfg.ego_bits)

        out['ego_sem_logits'] = self.ego_sem(h).view(-1, C_sem, k, k)
        out['passability_logits'] = self.passability(h)
        out['reward'] = self.reward(h).squeeze(-1)
        out['done_logit'] = self.done(h).squeeze(-1)
        out['value'] = self.value(h)

        if self.bag is not None:
            out['bag_logits'] = self.bag(h)
        if self.stats is not None:
            out['stats_pred'] = self.stats(h)
        if self.ego_bits is not None and nb > 0:
            out['ego_bits_logits'] = self.ego_bits(h).view(-1, nb, k, k)
        # safety always present (tiny head); compute loss only if provided
        out['safety_logits'] = self.safety(h)
        if self.goal is not None:
            out['goal_pred'] = self.goal(h)
        if self.lowres_occ is not None:
            hpr, wpr = self.lowres_shape
            out['lowres_occ_logits'] = self.lowres_occ(h).view(-1, 1, hpr, wpr)
        if self.ego_char is not None:
            out['ego_char_logits'] = self.ego_char(h).view(-1, cfg.ego_char_vocab, k, k)
        if self.ego_color is not None:
            out['ego_color_logits'] = self.ego_color(h).view(-1, cfg.ego_color_vocab, k, k)
        if self.skill_belief is not None:
            out['skill_logits'] = self.skill_belief(h)
        return out


class LatentForwardModel(nn.Module):
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


class InverseDynamics(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.enabled = (cfg.action_dim > 0) and cfg.use_inverse_dynamics
        if not self.enabled:
            return
        self.net = MLP(2 * cfg.latent_dim, 256, cfg.action_dim, num_layers=3)

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("Inverse dynamics disabled.")
        x = torch.cat([z_t, z_tp1], dim=1)
        return self.net(x)  # logits [B, A]


# ============================
# Main model
# ============================

class DecisionVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ScreenEncoder(cfg)
        self.heads = Heads(cfg)
        self.fwd = LatentForwardModel(cfg)
        self.inv = InverseDynamics(cfg)
        self.std_prior = StandardNormalPrior()
        self.hmm_prior = StickyHMMPriorPlaceholder(cfg.latent_dim, num_skills=cfg.skill_num)
        self.detach_bag_prior = True  # reserved if you later inject bag priors elsewhere

    # Encode only
    def encode(self, obs, blstats: Optional[torch.Tensor] = None, hero_xy: Optional[torch.Tensor] = None):
        return self.encoder(obs, blstats, hero_xy)

    # Full forward
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = batch['obs']
        blstats = batch.get('blstats', None)
        hero_xy = batch.get('hero_xy', None) if self.cfg.use_hero_xy else None
        mu, logvar, z = self.encode(obs, blstats, hero_xy)
        out = {'z_mu': mu, 'z_logvar': logvar, 'z': z}
        out.update(self.heads(z, self.cfg))
        if self.fwd.enabled and ('action_onehot' in batch):
            out['z_next_pred'] = self.fwd(z, batch['action_onehot'])
        if self.inv.enabled and ('z_next_detached' in batch):
            out['inv_logits'] = self.inv(z, batch['z_next_detached'])
        return out

    # Losses
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

        # --- Core heads ---
        if 'ego_sem_target' in batch:
            ce = F.cross_entropy(out['ego_sem_logits'], batch['ego_sem_target'].long(), reduction='mean')
            losses['ego_sem'] = ce * w.get('ego_sem', 0.0)

        if 'passability_target' in batch:
            bce = F.binary_cross_entropy_with_logits(out['passability_logits'], batch['passability_target'], reduction='mean')
            losses['passability'] = bce * w.get('passability', 0.0)

        if 'reward_target' in batch:
            mse = F.mse_loss(out['reward'], batch['reward_target'], reduction='mean')
            losses['reward'] = mse * w.get('reward', 0.0)
        if 'done_target' in batch:
            bce = F.binary_cross_entropy_with_logits(out['done_logit'], batch['done_target'], reduction='mean')
            losses['done'] = bce * w.get('done', 0.0)
        if 'value_k_target' in batch:
            mse = F.mse_loss(out['value'], batch['value_k_target'], reduction='mean')
            losses['value'] = mse * w.get('value', 0.0)

        if (self.heads.bag is not None) and ('bag_target' in batch):
            logits = out['bag_logits']
            target = batch['bag_target']
            weight = batch.get('bag_weight', torch.ones_like(target))
            per_class = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction='none')
            denom = weight.sum(dim=1).clamp_min(1.0)
            bag_loss = (per_class.sum(dim=1) / denom).mean()
            losses['bag'] = bag_loss * w.get('bag', 0.0)

        if (self.heads.stats is not None) and ('stats_target' in batch):
            mse = F.mse_loss(out['stats_pred'], batch['stats_target'], reduction='mean')
            losses['stats'] = mse * w.get('stats', 0.0)

        # --- Optional ego bits ---
        if ('ego_bits_target' in batch) and ('ego_bits_mask' in batch) and ('ego_bits_logits' in out):
            bce = F.binary_cross_entropy_with_logits(out['ego_bits_logits'], batch['ego_bits_target'], reduction='none')
            masked = bce * batch['ego_bits_mask']
            loss_bits = masked.sum() / (batch['ego_bits_mask'].sum().clamp_min(1.0))
            losses['ego_bits'] = loss_bits * w.get('ego_bits', 0.0)

        # --- Safety risk (8 dirs) ---
        if 'safety_target' in batch:
            bce = F.binary_cross_entropy_with_logits(out['safety_logits'], batch['safety_target'], reduction='mean')
            losses['safety'] = bce * w.get('safety', 0.0)

        # --- Goal direction ---
        if ('goal_target' in batch) and ('goal_pred' in out):
            mse = F.mse_loss(out['goal_pred'], batch['goal_target'], reduction='mean')
            losses['goal'] = mse * w.get('goal', 0.0)

        # --- Low‑res occupancy ---
        if ('lowres_occ_target' in batch) and ('lowres_occ_logits' in out):
            bce = F.binary_cross_entropy_with_logits(out['lowres_occ_logits'], batch['lowres_occ_target'], reduction='mean')
            losses['lowres_occ'] = bce * w.get('lowres_occ', 0.0)

        # --- Ego exact char/color (tiny weight) ---
        if ('ego_char_target' in batch) and ('ego_char_logits' in out):
            ce = F.cross_entropy(out['ego_char_logits'], batch['ego_char_target'].long(), reduction='mean')
            losses['ego_char'] = ce * w.get('ego_char', 0.0)
        if ('ego_color_target' in batch) and ('ego_color_logits' in out):
            ce = F.cross_entropy(out['ego_color_logits'], batch['ego_color_target'].long(), reduction='mean')
            losses['ego_color'] = ce * w.get('ego_color', 0.0)

        # --- Dynamics ---
        if self.fwd.enabled and ('z_next_detached' in batch) and ('action_onehot' in batch) and ('z_next_pred' in out):
            z_next_pred = out['z_next_pred']
            z_next_tgt = batch['z_next_detached']
            mse = F.mse_loss(z_next_pred, z_next_tgt, reduction='mean')
            cos = 1.0 - F.cosine_similarity(z_next_pred, z_next_tgt, dim=1).mean()
            losses['forward'] = (0.5 * (mse + cos)) * w.get('forward', 0.0)

        if self.inv.enabled and ('inv_logits' in out):
            if 'action_index' in batch:
                ce = F.cross_entropy(out['inv_logits'], batch['action_index'].long(), reduction='mean')
            elif 'action_onehot' in batch:
                ce = F.cross_entropy(out['inv_logits'], batch['action_onehot'].argmax(dim=1).long(), reduction='mean')
            else:
                ce = out['inv_logits'].sum() * 0.0  # no target; zero loss
            losses['inverse'] = ce * w.get('inverse', 0.0)

        # --- KL prior ---
        mu, logvar = out['z_mu'], out['z_logvar']
        kl_std = self.std_prior(mu, logvar)
        if prior_mode == 'standard':
            kl = kl_std
        elif prior_mode == 'hmm':
            kl_hmm = self.hmm_prior(mu, logvar, batch.get('skills'))
            kl = torch.where(kl_hmm > 0, kl_hmm, kl_std)
        elif prior_mode == 'blend':
            kl_hmm = self.hmm_prior(mu, logvar, batch.get('skills'))
            if (self.hmm_prior.mu is None) or (batch.get('skills') is None):
                kl = kl_std
            else:
                kl = alpha * kl_std + (1.0 - alpha) * kl_hmm
        else:
            raise ValueError(f"Unknown prior_mode: {prior_mode}")
        losses['kl'] = kl.mean() * beta

        total = sum(losses.values()) if len(losses) > 0 else losses['kl']
        return total, losses


# ============================
# Factory
# ============================

def build_model(**kwargs) -> DecisionVAE:
    cfg = VAEConfig(**kwargs)
    return DecisionVAE(cfg)

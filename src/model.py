# MiniHack Variational Auto‑Encoder (glyph‐chars + colours + messages + blstats + hero info)
# ---------------------------------------------------------------------
# Author: Xu Chen
# Date  : 30/06/2025
"""
A comprehensive VAE for MiniHack observations with mixed discrete/continuous modeling.

This VAE handles multiple input modalities and provides both embedding-level and raw-level
reconstruction for flexible representation learning and generation.

Inputs (per time-step)
---------------------
* glyph_chars : LongTensor[B,H,W] - ASCII character codes (32-127) per map cell
* glyph_colors: LongTensor[B,H,W] - color indices (0-15) per map cell  
* blstats     : FloatTensor[B,27] - game statistics (hp, gold, position, etc.) - processed to 43 dims
* msg_tokens  : LongTensor[B,256] - tokenized message text (0-127 + SOS/EOS)
* hero_info   : LongTensor[B,4] - hero attributes [role, race, gender, alignment]

Outputs  
-------
* Raw Reconstruction Logits (for discrete generation):
* char_logits, color_logits - pixel-wise categorical distributions
* stats_continuous - continuous stats on original scale  
* stats_discrete_logits - hunger/dungeon/level classification logits
* msg_logits - token-wise categorical distributions
* hero_logits - hero attribute classification logits
* inv_oclasses_logits - inventory object class classification logits
* inv_strs_logits - inventory string character-wise classification logits

* Embedding Reconstructions (for continuous generation):
* glyph_emb_decoded, stats_emb_decoded, msg_emb_decoded - continuous embeddings
* inv_emb_decoded - inventory embedding reconstruction
* mu, logvar - latent Gaussian parameters (for KL divergence)

Architecture
-----------
**Encoder**: 6 parallel feature extractors → fusion → latent space (64-dim)
    1. GlyphCNN: [B,H,W] → [B,64] - spatial glyph features
    2. GlyphBag: [B,H,W] → [B,32] - permutation-invariant glyph features  
    3. StatsMLP: [B,27] → [B,16] - preprocessed game statistics (43 processed dims)
    4. MessageGRU: [B,256] → [B,12] - bidirectional message encoding
    5. HeroEmb: [B,4] → [B,16] - hero attribute embeddings

**Decoder**: Shared latent → specialized heads for each modality
    - Supports both discrete (logits) and continuous (embeddings) reconstruction
    - Uses mixed loss: MSE for continuous stats, CrossEntropy for discrete components

**Key Features**:
- Low-rank + diagonal covariance for flexible posterior modeling
- BlstatsPreprocessor: BatchNorm + embeddings for mixed continuous/discrete stats
- Teacher forcing for autoregressive message decoding
- Optional glyph bag and hero info for enhanced representation learning
- InventoryEncoder for handling inventory object classes and descriptions
"""

from __future__ import annotations
from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math
from typing import Optional, List, Dict
from enum import IntEnum
from dataclasses import dataclass, field
from nle import nethack

# Import NetHackCategory from data_collection
try:
    from .data_collection import NetHackCategory, crop_ego, categorize_glyph_tensor, nearest_stairs_vector, discounted_k_step_multi_with_mask
except ImportError:
    # Fallback for when running as script
    from src.data_collection import NetHackCategory, crop_ego, categorize_glyph_tensor, nearest_stairs_vector, discounted_k_step_multi_with_mask

# ------------------------- hyper‑params ------------------------------ #
CHAR_DIM = 96      # ASCII code space for characters shown on the map (32-127)
COLOR_DIM = 16      # colour id space for colours
BLSTATS_DIM = 27   # raw scalar stats (hp, gold, …)
MSG_PAD = 0  
MSG_VOCAB = 128     # Nethack takes 32-127 byte for char of messages
MSG_SOS = MSG_VOCAB  # start-of-sequence token id
MSG_EOS = MSG_VOCAB + 1  # end-of-sequence token id
MSG_VSIZE = MSG_VOCAB + 2  # vocab size including SOS/EOS
MSG_MAXLEN = 256  # max length of message text (padded/truncated)
GLYPH_DIM = CHAR_DIM * COLOR_DIM  # glyph dim for char + color
BLANK_CHAR = 32        # blank space
BLACK_COLOR = 0        # black color (background)
# NetHack map dimensions (from NetHack source: include/config.h)
MAP_HEIGHT = 21     # ROWNO - number of rows in the map
MAP_WIDTH = 79      # COLNO-1 - playable columns (COLNO=80, but rightmost is UI)
MAX_X_COORD = 78    # Maximum x coordinate (width - 1)
MAX_Y_COORD = 20    # Maximum y coordinate (height - 1)
ROLE_CAD = 13     # role cardinality for MiniHack (e.g. 4 for 'knight')
RACE_CAD = 5      # race cardinality for MiniHack (e.g. 0 for 'human')
GEND_CAD = 3      # gender cardinality for MiniHack (e.g. 'male', 'female', 'neuter')
ALIGN_CAD = 3     # alignment cardinality for MiniHack (e.g. 1 for 'lawful')

# Condition mask constants (blstats[25] bit field)
CONDITION_BITS = [
    0x00000001,  # Stoned
    0x00000002,  # Slimed  
    0x00000004,  # Strangled
    0x00000008,  # Food Poisoning
    0x00000010,  # Terminal Illness
    0x00000020,  # Blind
    0x00000040,  # Deaf
    0x00000080,  # Stunned
    0x00000100,  # Confused
    0x00000200,  # Hallucinating
    0x00000400,  # Levitating
    0x00000800,  # Flying
    0x00001000,  # Riding
]
NUM_CONDITIONS = len(CONDITION_BITS)  # 13 condition bits

# blstats meanings
BLSTATS_CAT = [
    "x_coordinate",      # 0: Current x position (0-78)
    "y_coordinate",      # 1: Current y position (0-20)
    "strength_25",       # 2: Strength (base 25)
    "strength_125",      # 3: Strength (base 125) 
    "dexterity",         # 4: Dexterity (3-25)
    "constitution",      # 5: Constitution (3-25)
    "intelligence",      # 6: Intelligence (3-25)
    "wisdom",            # 7: Wisdom (3-25)
    "charisma",          # 8: Charisma (3-25)
    "score",             # 9: Current score (0-999999+)
    "hitpoints",         # 10: Current hit points (0-max_hp)
    "max_hitpoints",     # 11: Maximum hit points (1-999)
    "depth",             # 12: Current dungeon depth (1-50+)
    "gold",              # 13: Amount of gold (0-999999+)
    "energy",            # 14: Current magical energy (0-max_energy)
    "max_energy",        # 15: Maximum magical energy (0-999)
    "armor_class",       # 16: Armor class (-20 to 20)
    "monster_level",     # 17: Monster level (0-30)
    "experience_level",  # 18: Experience level (1-30)
    "experience_points", # 19: Experience points (0-999999+)
    "time",              # 20: Game time (0-999999+)
    "hunger_state",      # 21: Hunger state (0-6: Satiated, Not Hungry, Hungry, Weak, Fainting, Faint, Starved)
    "carrying_capacity", # 22: How much you can carry (0-999)
    "dungeon_number",    # 23: Which dungeon you're in (0-10)
    "level_number",      # 24: Level within the dungeon (0-50)
    "condition_mask",    # 25: Condition mask (bitfield)
    "alignment",         # 26: Alignment (-1 to 1, -1=chaotic, 0=neutral, 1=lawful)
]

COMMON_CHARS = [ord('#'), ord('.'), ord('-'), ord('|'), ord('>'), ord('<'), ord('@')]
COMMON_GLYPHS = [(a, c) for a in COMMON_CHARS for c in range(COLOR_DIM)] # (char, color) pairs for common glyphs
HERO_CHAR = ord('@')  # hero character code (ASCII 64)

def _gn_groups(c: int) -> int:
    """Pick a sensible GroupNorm group count that divides c."""
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1

def _pair_index(char_ascii: torch.Tensor, color_id: torch.Tensor) -> torch.Tensor:
    """
    Map ASCII char (32..127) + color (0..15) -> flat pair index [0..GLYPH_DIM-1].
    Both tensors are same shape; returns long tensor of that shape.
    """
    ch = (char_ascii - 32).clamp(0, CHAR_DIM - 1)
    co = color_id.clamp(0, COLOR_DIM - 1)
    return ch * COLOR_DIM + co

def _to_pair_idx(pair_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of _pair_index: flat pair index [0..GLYPH_DIM-1] -> (char_ascii [32..127], color_id [0..15])
    """
    ch = (pair_idx // COLOR_DIM) + 32
    co = pair_idx % COLOR_DIM
    return ch, co

def make_pair_bag(glyph_chars: torch.Tensor, glyph_colors: torch.Tensor) -> torch.Tensor:
    """
    Returns a binary presence vector for (char,color) pairs, excluding (32,0).
    glyph_chars/colors: [B,H,W]  ->  bag: [B, GLYPH_DIM] in {0,1}.
    """
    B, H, W = glyph_chars.shape
    flat_idx = _pair_index(glyph_chars, glyph_colors)   # [B,H,W]
    bag = torch.zeros(B, GLYPH_DIM, device=glyph_chars.device, dtype=torch.float32)
    bag.scatter_add_(1, flat_idx.view(B, -1), torch.ones(B, H*W, device=glyph_chars.device))
    # remove the padding pair (space, black)
    pad_idx = _pair_index(torch.full_like(glyph_chars, BLANK_CHAR), torch.zeros_like(glyph_colors))
    pad_counts = torch.zeros(B, GLYPH_DIM, device=glyph_chars.device).scatter_add_(
        1, pad_idx.view(B, -1), torch.ones(B, H*W, device=glyph_chars.device)
    )
    bag = (bag - pad_counts).clamp_min_(0.0)
    bag = (bag > 0).float()
    return bag

def bag_marginals(bag_pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    bag_pairs: [B, GLYPH_DIM] -> (char_presence [B,CHAR_DIM], color_presence [B,COLOR_DIM])
    """
    B = bag_pairs.size(0)
    pairs = bag_pairs.view(B, CHAR_DIM, COLOR_DIM)
    char_pres  = 1.0 - torch.prod(1.0 - pairs, dim=2)
    color_pres = 1.0 - torch.prod(1.0 - pairs, dim=1)
    return char_pres, color_pres

def bag_presence_to_glyph_sets(bag_presence: torch.Tensor) -> list[set[tuple[int, int]]]:
    """
    Convert bag_presence tensor to list of sets containing (char, color) pairs.
    
    Args:
        bag_presence: [B, GLYPH_DIM] tensor with 1s indicating presence of glyph pairs
        
    Returns:
        List of length B, where each element is a set containing (char_ascii, color_id) tuples
        for glyphs that are present in the bag (bag_presence == 1)
    """
    B = bag_presence.shape[0]
    result = []
    
    for b in range(B):
        # Get indices where bag_presence is 1 for this batch element
        # exclude padding (32,0) which is at index 0
        present_indices = torch.where(bag_presence[b] == 1)[0]  # [num_present]
        present_indices = present_indices[present_indices != 0]  # Exclude padding index

        # Convert flat indices to (char, color) pairs using _to_pair_idx
        bag_set = set()
        for idx in present_indices:
            char_ascii, color_id = _to_pair_idx(idx)
            bag_set.add((char_ascii.item(), color_id.item()))
        
        result.append(bag_set)
    
    return result

def hero_presence_and_centroid(glyph_chars: torch.Tensor, blstats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Presence p in {0,1} and centroid (y,x) in [-1,1]^2 for '@'.
    If absent, centroid is (0,0) and p=0 (loss masked by p).
    """
    B= glyph_chars.shape[0]

    x_coord, y_coord = blstats[:, 0], blstats[:, 1]
    # Convert coordinates to long for indexing and clamp to valid range
    y_coord_idx = torch.clamp(y_coord.long(), 0, MAX_Y_COORD)
    x_coord_idx = torch.clamp(x_coord.long(), 0, MAX_X_COORD)
    present = (glyph_chars[torch.arange(B, device=glyph_chars.device), y_coord_idx, x_coord_idx] == HERO_CHAR).float()  # [B]
    cy = y_coord / MAX_Y_COORD * 2 - 1  # y coordinate in [-1,1]
    cx = x_coord / MAX_X_COORD * 2 - 1  # x coordinate in [-1,1]
    centroid = torch.stack([cy, cx], dim=1)                      # [B,2]
    return present, centroid

def _broadcast_nk(x: torch.Tensor | None, N: int, K: int, device) -> torch.Tensor | None:
    if x is None:
        return None
    x = x.to(device)
    if x.dim() == 1 and x.shape[0] == N:
        x = x.unsqueeze(1).expand(N, K)
    elif x.dim() == 2 and x.shape == (N, K):
        pass
    else:
        raise ValueError(f"mask/weight must be [N] or [N,{K}], got {tuple(x.shape)}")
    return x

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

class ConvBlock(nn.Module):
    """
    Conv -> GroupNorm -> ReLU -> Dropout.
    Supports anisotropic kernel/dilation and correct padding.
    """
    def __init__(self, c_in, c_out, k=3, dilation=1, drop=0.0):
        super().__init__()
        # normalize args to tuples
        if isinstance(k, int): kh, kw = k, k
        else: kh, kw = k
        if isinstance(dilation, int): dh, dw = dilation, dilation
        else: dh, dw = dilation

        pad_h = dh * (kh // 2)
        pad_w = dw * (kw // 2)

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(kh, kw), padding=(pad_h, pad_w), dilation=(dh, dw)),
            nn.GroupNorm(num_groups=_gn_groups(c_out), num_channels=c_out),
            nn.ReLU(),
            nn.Dropout2d(drop) if drop and drop > 0 else nn.Identity()
        )

    def forward(self, x):
        res = self.net(x)
        return res
    
class AttnPool(nn.Module):
    """
    Masked attention pooling with K learnable queries.
    Returns either concatenated [B, K*C]
    """
    def __init__(self, C: int, K: int = 1):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(K, C))
        self.proj = nn.Linear(C, C, bias=False)
        # Fallback for fully blank frames (per head)
        self.empty_vec = nn.Parameter(torch.zeros(C))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, C, H, W]
        mask : [B, 1, H, W] in {0,1}; 1 = keep, 0 = blank
        """
        B, C, H, W = feats.shape
        X = feats.view(B, C, H * W).transpose(1, 2)     # [B, HW, C]
        X = self.proj(X)                                # [B, HW, C]
        Q = self.Q                                      # [K, C]

        # scores: [B, K, HW]
        scores = torch.einsum('bhc,kc->bkh', X, Q)
        mflat = mask.view(B, 1, H * W)                  # [B, 1, HW]
        scores = scores.masked_fill(mflat == 0, float('-inf'))

        # identify heads with all -inf (no valid locations)
        bad = torch.isinf(scores).all(dim=-1)           # [B, K]
        # safe softmax
        attn = torch.softmax(scores, dim=-1)            # [B, K, HW]
        attn = torch.nan_to_num(attn, nan=0.0)

        # pooled per head: [B, K, C]
        Z = torch.einsum('bkh,bhc->bkc', attn, X)

        if bad.any():
            # replace bad heads with learnable empty vector
            # Ensure dtype compatibility for mixed precision training
            Z[bad] = self.empty_vec.to(Z.dtype)

        return Z.reshape(B, -1)                     # [B, K*C]
    
class FiLM(nn.Module):
    """Per-sample channel-wise affine: y = x * (1 + gamma) + beta"""
    def __init__(self, z_dim: int, c: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, 2 * c)
    def forward(self, x: torch.Tensor, z: torch.Tensor):
        B, C, H, W = x.shape
        gamma, beta = self.fc(z).chunk(2, dim=-1)  # [B,C], [B,C]
        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)
        return x * (1.0 + gamma) + beta

class ResBlockFiLM(nn.Module):
    """Conv-GN-FiLM-ReLU x2 with residual; supports anisotropic dilation."""
    def __init__(self, c_in: int, c_out: int, dilation=(1,1), drop=0.0, z_dim: int = 0):
        super().__init__()
        dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        pad = (dh, dw)
        self.conv1 = nn.Conv2d(c_in,  c_out, 3, padding=pad, dilation=(dh, dw))
        self.gn1   = nn.GroupNorm(_gn_groups(c_out), c_out)
        self.film1 = FiLM(z_dim, c_out)

        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=pad, dilation=(dh, dw))
        self.gn2   = nn.GroupNorm(_gn_groups(c_out), c_out)
        self.film2 = FiLM(z_dim, c_out)

        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.skip  = (nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity())

    def forward(self, x, z):
        h = self.conv1(x); h = self.gn1(h); h = self.film1(h, z); h = self.act(h); h = self.drop(h)
        h = self.conv2(h); h = self.gn2(h); h = self.film2(h, z); h = self.act(h); h = self.drop(h)
        return h + self.skip(x)


@dataclass
class VAEConfig:
    
    # Embedding dimensions
    # --- Character and color embeddings
    char_emb: int = 16      # char‑id embedding dim (0-15)
    color_emb: int = 4       # colour‑id embedding dim
    glyph_emb: int = field(init=False)  # glyph embedding dim
    
    # --- Entity embeddings
    role_emb: int = 8        # role embedding dim
    race_emb: int = 4      # race embedding dim
    gend_emb: int = 2      # gender embedding dim
    align_emb: int = 2     # alignment embedding dim

    encoder_dropout: float = 0.0    # dropout rate for all encoder blocks
    decoder_dropout: float = 0.0    # dropout rate for all decoder blocks

    # --- Latent space
    latent_dim: int = 96     # z‑dim for VAE
    bag_dim: int = 16        # bag embedding dim (for glyph-bag)
    core_dim: int = field(init=False)  # core embedding dim (for map encoder)
    low_rank: int = 0        # low-rank factorisation rank for covariance
    
    # Encoder settings
    # --- Map encoder ---
    attn_queries: int = 2          # number of attention queries for pooling
    map_feat_channels: int = 96         # feature channels in map encoder
    # --- Stats encoder ---
    stat_feat_channels: int = 16         # feature channels in stats encoder
    # --- Message encoder ---
    msg_emb: int = 32                     # message embedding dim
    msg_hidden: int = 12                # message hidden dim

    # Decoder settings
    # --- Map decoder ---
    ego_window: int = 11                       # must be odd
    ego_classes: int = NetHackCategory.get_category_count()  # number of ego classes
    map_dec_base_ch: int = 64          # base channels in map decoder
    
    # --- Stats decoder ---
    stats_cond_dim: int = 19  # number of continuous stats
    
    # --- Message decoder ---
    msg_ch: int = 128
    
    # Dynamics heads
    action_dim: int = len(nethack.ACTIONS)     # one‑hot; 0 disables forward/inverse
    forward_hidden: int = 256
    use_world_model: bool = True
    use_inverse_dynamics: bool = True
    goal_dir_dim: int = 2                      # 0 disables (set 2 for (dx,dy) regression)
    passability_dirs: int = 8  # number of passability directions (N,NE,E,SE,S,SW,W,NW)
    value_horizons: List[int] = field(default_factory=lambda: [1, 5, 10])
    # Skill belief (future sticky HMM integration)
    skill_num: int = 0                         # number of skills; 0 disables skill head

    # Goal and value hyperparameters (moved from data collection)
    gamma: float = 0.95                        # discount factor for k-step returns
    goal_prefer: str = '>'                     # preferred stairs type ('>' for down, '<' for up)

    # KL prior mode
    prior_mode: str = "standard"               # "standard" | "hmm" | "blend"
    prior_blend_alpha: float = 0.5
    
    focal_loss_alpha = 0.5
    focal_loss_gamma = 2.0

    # Loss weights
    raw_modality_weights: Dict[str, float] = field(default_factory=lambda: {
        'ego_class': 5.0,
        'passability': 20.0,
        'safety': 20.0,
        'reward': 0.5,
        'done': 1.0,
        'value_k': 1.0,
        'bag': 3.0,
        'stats': 0.5,
        'msg': 1.0,
        'goal': 3.0,
        'occupy': 0.5,
        'forward': 5.0,
        'inverse': 10,
        'ego_char': 0.5,
        'ego_color': 0.5,
        'hero_loc': 20.0,
    })
    
    # KL Beta parameters for adaptive loss weighting
    initial_mi_beta: float = 0.0001
    final_mi_beta: float = 1.0
    mi_beta_shape: str = 'cosine'
    initial_tc_beta: float = 0.0001
    final_tc_beta: float = 1.0
    tc_beta_shape: str = 'cosine'
    initial_dw_beta: float = 0.0001
    final_dw_beta: float = 1.0
    dw_beta_shape: str = 'cosine'
    warmup_epoch_ratio: float = 0.3
    free_bits: float = 0.0
    # Contrastive addition for world model (mild by default)
    nce_weight: float = 0.1
    nce_temperature: float = 0.2

    def __post_init__(self):
        self.glyph_emb = self.char_emb + self.color_emb
        self.core_dim = self.latent_dim - self.bag_dim
        if self.core_dim <= 0:
            raise ValueError("Core dimension must be positive; check latent_dim and bag_dim settings.")
        if self.ego_window % 2 == 0:
            raise ValueError("Ego window must be odd; got {}".format(self.ego_window))
        if self.action_dim < 0:
            raise ValueError("Action dimension must be non-negative; got {}".format(self.action_dim))

class MapEncoder(nn.Module):
    """
    NetHack map encoder:
      - char & color embeddings (space can be true padding)
      - +1 foreground mask channel
      - +2 CoordConv channels (x,y in [-1,1])
      - Conv backbone mixing standard & anisotropic dilated convs
      - Masked attention pooling (K queries) or masked mean

    Forward returns a vector suitable for a VAE encoder MLP or heads.
    """
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.char_emb = nn.Embedding(CHAR_DIM , config.char_emb)
        self.col_emb  = nn.Embedding(COLOR_DIM, config.color_emb)
        in_ch = config.char_emb + config.color_emb + 1  # + foreground mask
        in_ch += 2 # +2 for CoordConv (x,y in [-1,1])

        # Stem
        self.stem = ConvBlock(in_ch, 64, k=3, dilation=1, drop=config.encoder_dropout)

        # Body: mix normal and width-focused dilations (map is 21x79)
        self.body = nn.Sequential(
            ConvBlock(64, 64, k=3, dilation=1,        drop=config.encoder_dropout),      # local
            ConvBlock(64, 64, k=3, dilation=(1, 3),   drop=config.encoder_dropout),      # widen RF (width)
            ConvBlock(64, 64, k=3, dilation=1,        drop=config.decoder_dropout),
            ConvBlock(64, 96, k=3, dilation=(1, 9),   drop=config.decoder_dropout),
            ConvBlock(96, config.map_feat_channels, k=3, dilation=1, drop=config.decoder_dropout),
        )

        # Pooling
        self.pool = AttnPool(C=config.map_feat_channels, K=config.attn_queries)
        pooled_dim = config.map_feat_channels * config.attn_queries

        # Tiny head you can keep or replace downstream
        self.out_dim = pooled_dim
        self.norm_out = nn.LayerNorm(self.out_dim)

        
    @staticmethod
    def _coords(B, H, W, device, dtype):
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return yy, xx

    def forward(self, glyph_chars: torch.IntTensor, glyph_colors: torch.IntTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # glyph_chars in ASCII [32..127]; shift to [0..95];
        gc = torch.clamp(glyph_chars - 32, 0, CHAR_DIM - 1)
        B, H, W = gc.shape
        device = gc.device
        fg = ((gc != 0) & (glyph_colors != 0)).unsqueeze(1).float()  # [B,1,H,W]
        ce = self.char_emb(gc).permute(0,3,1,2)         # [B,16,H,W]
        co = self.col_emb(glyph_colors).permute(0,3,1,2)# [B, 4,H,W]
        yy, xx = self._coords(B, H, W, device, ce.dtype)
        x = torch.cat([ce * fg, co * fg, fg, yy, xx], dim=1)
        x = self.stem(x)
        x = self.body(x)
        pooled = self.pool(x, fg)
        return self.norm_out(pooled)
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return self.out_dim
    
class MapDecoder(nn.Module):
    """Latent vector -> occupancy/glyph/color logits at 21x79, with FiLM and CoordConv."""
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.z_bag_dim = config.bag_dim
        self.z_core_dim = config.core_dim
        self.latent_dim = config.latent_dim
        self.ego_window = config.ego_window
        self.ego_classes = config.ego_classes
        self.base_ch = config.map_dec_base_ch
        self.register_buffer("full_yy", torch.linspace(-1, 1, MAP_HEIGHT).view(1,1,MAP_HEIGHT,1))
        self.register_buffer("full_xx", torch.linspace(-1, 1, MAP_WIDTH ).view(1,1,1,MAP_WIDTH))
        self.register_buffer("ego_yy",  torch.linspace(-1, 1, self.ego_window).view(1,1,self.ego_window,1))
        self.register_buffer("ego_xx",  torch.linspace(-1, 1, self.ego_window).view(1,1,1,self.ego_window))
        
        self.bag_head = MLP(self.z_bag_dim, 128, GLYPH_DIM, num_layers=2, dropout=config.decoder_dropout)
        
        self.hero_presence_head = nn.Linear(self.z_bag_dim, 1)   # logits
        self.hero_centroid_head = nn.Linear(self.z_bag_dim, 2)   # tanh -> [-1,1]
        self.ego_head = MLP(config.latent_dim, 128, self.ego_window * self.ego_window * self.ego_classes, num_layers=2, dropout=config.decoder_dropout)

        # project z to a broadcast feature map
        self.z_core_to_map = nn.Sequential(nn.Linear(self.z_core_dim, config.map_dec_base_ch), nn.ReLU(inplace=True), nn.Linear(config.map_dec_base_ch, config.map_dec_base_ch))
        self.head_occ = nn.Sequential(nn.Conv2d(config.map_dec_base_ch, config.map_dec_base_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(config.map_dec_base_ch//2, 1, 1))

        self.z_to_ego_map = nn.Sequential(nn.Linear(self.latent_dim, config.map_dec_base_ch), nn.ReLU(inplace=True), nn.Linear(config.map_dec_base_ch, config.map_dec_base_ch))
        # FiLM-conditioned residual body (anisotropic to cover width)
        self.core_block = ResBlockFiLM(config.map_dec_base_ch, config.map_dec_base_ch, (1,9),  config.decoder_dropout, z_dim=self.latent_dim)

        # heads
        self.head_chr = nn.Sequential(nn.Conv2d(config.map_dec_base_ch, config.map_dec_base_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(config.map_dec_base_ch//2, CHAR_DIM, 1))
        self.head_col = nn.Sequential(nn.Conv2d(config.map_dec_base_ch, config.map_dec_base_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(config.map_dec_base_ch//2, COLOR_DIM, 1))

    def _prior_to_bias(self, prior: torch.Tensor, strength: float) -> torch.Tensor:
        """
        prior: probabilities in [0,1], shape [B,C].
        Returns additive bias (logits) [B,C] scaled by bag_bias_strength.
        """
        p = prior.clamp(1e-6, 1 - 1e-6)
        return strength * torch.log(p / (1 - p))


    def forward(self, z: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        B = z.size(0)
        z_bag  = z[:, :self.z_bag_dim]
        z_core = z[:, self.z_bag_dim:]

        bag_head_logits = self.bag_head(z_bag)  # [B, GLYPH_DIM]
        hero_logits = self.hero_presence_head(z_bag).squeeze(1)  # [B,]
        hero_centroid = torch.tanh(self.hero_centroid_head(z_bag))  # [B,2] in [-1,1]
        ego_logits = self.ego_head(z).view(-1, self.ego_classes, self.ego_window, self.ego_window)  # [B, C, k, k]
        
        z_core_map = self.z_core_to_map(z_core).view(B, self.base_ch, 1, 1).expand(B, self.base_ch, MAP_HEIGHT, MAP_WIDTH)  # [B, C, H, W]
        coords_full = torch.cat([self.full_yy.expand(B,-1,-1,-1), self.full_xx.expand(B,-1,-1,-1)], dim=1)
        x_full = torch.cat([z_core_map, coords_full], dim=1)
        occ = self.head_occ(x_full) # [B, 1, H, W]
        z_ego_map = self.z_to_ego_map(z).view(B, self.base_ch, 1, 1).expand(B, self.base_ch, self.ego_window, self.ego_window)  # [B, C, k, k]
        coords_ego = torch.cat([self.ego_yy.expand(B,-1,-1,-1), self.ego_xx.expand(B,-1,-1,-1)], dim=1)
        x_ego_in = torch.cat([z_ego_map, coords_ego], dim=1)
        x_ego = self.core_block(x_ego_in, z) # [B, C, k, k]
        g_logits = self.head_chr(x_ego)  # [B, CHAR_DIM, k, k]
        c_logits = self.head_col(x_ego)  # [B, COLOR_DIM, k, k]
        
        out['bag_logits'] = bag_head_logits
        out['hero_presence_logits'] = hero_logits
        out['hero_centroid'] = hero_centroid
        out['ego_class_logits'] = ego_logits
        out['occupy_logits'] = occ
        out['ego_char_logits'] = g_logits
        out['ego_color_logits'] = c_logits
        return out

    @staticmethod
    def _sample_per_pixel(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        logits: [B, C, H, W] -> returns indices [B, H, W]
        """
        if deterministic:
            return logits.argmax(dim=1)

        B, C, H, W = logits.shape
        x = (logits / max(temperature, 1e-6)).permute(0, 2, 3, 1).reshape(B*H*W, C)  # [BHW,C]

        if top_k > 0:
            vals, inds = torch.topk(x, k=min(top_k, C), dim=1)
            mask = torch.full_like(x, float("-inf"))
            mask.scatter_(1, inds, vals)
            x = mask

        if 0.0 < top_p < 1.0:
            probs = x.softmax(dim=1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=1)
            cum = sorted_probs.cumsum(dim=1)
            keep = cum <= top_p
            # always keep at least 1
            keep[:, 0] = True
            filtered = torch.full_like(x, float("-inf"))
            filtered.scatter_(1, sorted_idx, torch.where(keep, x.gather(1, sorted_idx), torch.full_like(sorted_probs, float("-inf"))))
            x = filtered

        idx = torch.distributions.Categorical(logits=x).sample()  # [BHW]
        return idx.view(B, H, W)

    @staticmethod
    def sample_ego_from_logits(
        ego_char_logits: torch.Tensor,
        ego_color_logits: torch.Tensor,
        ego_class_logits: torch.Tensor,
        temperature: float = 1.0,
        glyph_top_k: int = 0,
        glyph_top_p: float = 1.0,
        color_top_k: int = 0,
        color_top_p: float = 1.0,
        class_top_k: int = 0,
        class_top_p: float = 1.0,
        deterministic: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Inputs are logits:
          - ego_char_logits:  [B, CHAR_DIM, k, k]
          - ego_color_logits: [B, COLOR_DIM, k, k]
          - ego_class_logits: [B, NUM_CLASSES, k, k]
        Returns integer maps:
          - ego_chars:  [B, k, k]    ASCII in [32..127] (space==32)
          - ego_colors: [B, k, k]    in [0..15]
          - ego_class:  [B, k, k]    class indices
        """
        # sample per-pixel indices
        char_idx = MapDecoder._sample_per_pixel(
            ego_char_logits, temperature, glyph_top_k, glyph_top_p, deterministic
        )                                # [B,k,k] in [0..CHAR_DIM-1]
        color_idx = MapDecoder._sample_per_pixel(
            ego_color_logits, temperature, color_top_k, color_top_p, deterministic
        )                                # [B,k,k] in [0..COLOR_DIM-1]
        class_idx = MapDecoder._sample_per_pixel(
            ego_class_logits, temperature, class_top_k, class_top_p, deterministic
        )                                # [B,k,k]

        # shift chars back to ASCII range
        ego_chars = (char_idx + 32).clamp(32, 127)     # [B,k,k] ASCII
        ego_colors = color_idx                         # [B,k,k] 0..15
        ego_class  = class_idx                         # [B,k,k]

        return {"ego_chars": ego_chars, "ego_colors": ego_colors, "ego_class": ego_class}

    @staticmethod
    def format_passability_safety(
        passability_presence: torch.Tensor,           # [B, 8]
        safety_presence: torch.Tensor,                # [B, 8]
        pass_mask: torch.Tensor | None = None,        # [B] or [B,8], 0 = ignore (OOB, etc.)
        safe_mask: torch.Tensor | None = None,        # [B] or [B,8]
        pass_weight: torch.Tensor | None = None,      # [B] or [B,8], e.g. 0.2 for unknown tiles
        safe_weight: torch.Tensor | None = None,      # [B] or [B,8]
        masked_value: float | None = None             # e.g., float('nan') to visualize masked spots
    ) -> Dict[str, torch.Tensor]:
        """
        Format passability/safety + (optional) masks & weights into 3x3 hero-centric grids.

        Direction order expected: (N, E, S, W, NE, SE, SW, NW)
        Returns a dict with keys:
        - 'pass_grid', 'safe_grid'              -> [B,3,3] values
        - 'pass_mask_grid', 'safe_mask_grid'    -> [B,3,3] {0,1}
        - 'pass_weight_grid', 'safe_weight_grid'-> [B,3,3] >=0
        The hero center cell [1,1] is set to -1 in value grids, 1 in mask grids, 0 in weight grids.
        """
        device = passability_presence.device
        B, K = passability_presence.shape
        assert safety_presence.shape == (B, K)
        assert K == 8, "Expected 8 directions (N,E,S,W,NE,SE,SW,NW)."

        # Broadcast mask/weight to [B,8] if provided
        pass_mask   = _broadcast_nk(pass_mask,   B, K, device)
        safe_mask   = _broadcast_nk(safe_mask,   B, K, device)
        pass_weight = _broadcast_nk(pass_weight, B, K, device)
        safe_weight = _broadcast_nk(safe_weight, B, K, device)

        # Allocate grids
        pass_grid       = torch.zeros(B, 3, 3, device=device, dtype=passability_presence.dtype)
        safe_grid       = torch.zeros(B, 3, 3, device=device, dtype=safety_presence.dtype)
        pass_mask_grid  = torch.ones(B, 3, 3, device=device)   # center marked valid by default
        safe_mask_grid  = torch.ones(B, 3, 3, device=device)
        pass_w_grid     = torch.zeros(B, 3, 3, device=device)
        safe_w_grid     = torch.zeros(B, 3, 3, device=device)

        # Direction -> grid mapping
        # Grid:
        #  (0,0)=NW  (0,1)=N  (0,2)=NE
        #  (1,0)=W   (1,1)=@  (1,2)=E
        #  (2,0)=SW  (2,1)=S  (2,2)=SE
        dir_pos = [(0,1), (1,2), (2,1), (1,0), (0,2), (2,2), (2,0), (0,0)]  # N,E,S,W,NE,SE,SW,NW

        for i, (r, c) in enumerate(dir_pos):
            pass_grid[:, r, c] = passability_presence[:, i]
            safe_grid[:, r, c] = safety_presence[:, i]

            if pass_mask is not None:
                pass_mask_grid[:, r, c] = pass_mask[:, i].to(pass_mask_grid.dtype)
            if safe_mask is not None:
                safe_mask_grid[:, r, c] = safe_mask[:, i].to(safe_mask_grid.dtype)

            if pass_weight is not None:
                pass_w_grid[:, r, c] = pass_weight[:, i].to(pass_w_grid.dtype)
            if safe_weight is not None:
                safe_w_grid[:, r, c] = safe_weight[:, i].to(safe_w_grid.dtype)

        # Center (hero)
        pass_grid[:, 1, 1] = -1.0
        safe_grid[:, 1, 1] = -1.0
        pass_mask_grid[:, 1, 1] = 1.0
        safe_mask_grid[:, 1, 1] = 1.0
        pass_w_grid[:, 1, 1] = 0.0
        safe_w_grid[:, 1, 1] = 0.0

        # Optionally visualize masked cells as NaN (or any sentinel)
        if masked_value is not None:
            mv = torch.tensor(masked_value, device=device, dtype=pass_grid.dtype)
            if pass_mask is not None:
                pm = (pass_mask_grid[:, :, :] < 0.5)
                pass_grid[pm] = mv
            if safe_mask is not None:
                sm = (safe_mask_grid[:, :, :] < 0.5)
                safe_grid[sm] = mv
                
        return {
            'passability_grid': pass_grid,
            'safety_grid': safe_grid,
            'pass_mask_grid': pass_mask_grid,
            'safe_mask_grid': safe_mask_grid,
            'pass_weight_grid': pass_w_grid,
            'safe_weight_grid': safe_w_grid,
        }

class BlstatsPreprocessor(nn.Module):
    def __init__(self, stats_dim=BLSTATS_DIM):
        super().__init__()
        self.stats_dim = stats_dim
        self.useful = [i for i in range(stats_dim) if i not in [20, 26]]
        self.disc = [21, 23, 24]
        self.cond_idx = 25
        self.cont = [i for i in self.useful if i not in self.disc + [self.cond_idx]]
        self.bn = nn.BatchNorm1d(len(self.cont) - 2)
        # embeddings for encoder side (not used in decoder loss)
        self.hunger_emb  = nn.Embedding(7, 3)
        self.dungeon_emb = nn.Embedding(11, 4)
        self.level_emb   = nn.Embedding(51, 4)
    def forward(self, bl):
        # encoded features for encoder MLP
        cont = bl[:, self.cont].float()
        proc = cont.clone()
        # coords
        proc[:, 0] = cont[:, 0] / MAX_X_COORD
        proc[:, 1] = cont[:, 1] / MAX_Y_COORD
        # log1p for score,gold,exp
        for idx in [9, 13, 19]:
            if idx in self.cont:
                j = self.cont.index(idx)
                proc[:, j] = torch.log1p(proc[:, j].clamp(min=0))
        # ratios
        hp_idx = self.cont.index(10); maxhp_idx = self.cont.index(11)
        en_idx = self.cont.index(14); maxen_idx = self.cont.index(15)
        proc[:, hp_idx] = cont[:, 10] / torch.clamp(cont[:, 11], min=1)
        proc[:, en_idx] = cont[:, 14] / torch.clamp(cont[:, 15], min=1)
        keep = [i for i in range(proc.size(1)) if i not in [maxhp_idx, maxen_idx]]
        proc = proc[:, keep]
        proc = self.bn(proc)
        hunger = torch.clamp(bl[:, 21].long(), 0, 6)
        dung   = torch.clamp(bl[:, 23].long(), 0, 10)
        level  = torch.clamp(bl[:, 24].long(), 0, 50)
        h_emb  = self.hunger_emb(hunger)
        d_emb  = self.dungeon_emb(dung)
        l_emb  = self.level_emb(level)
        cond_mask = bl[:, self.cond_idx].long()
        cond_vec = torch.stack([((cond_mask & v) > 0).float() for v in CONDITION_BITS], dim=1)
        enc = torch.cat([proc, h_emb, d_emb, l_emb, cond_vec], dim=1)  # [B,43]
        return enc
    def get_output_dim(self):
        return 43
    # targets for decoder loss
    def targets(self, bl) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            cont = bl[:, self.cont].float()
            proc = cont.clone()
            proc[:, 0] = cont[:, 0] / MAX_X_COORD
            proc[:, 1] = cont[:, 1] / MAX_Y_COORD
            for idx in [9, 13, 19]:
                if idx in self.cont:
                    j = self.cont.index(idx)
                    proc[:, j] = torch.log1p(proc[:, j].clamp(min=0))
            hp_idx = self.cont.index(10); maxhp_idx = self.cont.index(11)
            en_idx = self.cont.index(14); maxen_idx = self.cont.index(15)
            proc[:, hp_idx] = cont[:, hp_idx] / torch.clamp(cont[:, maxhp_idx], min=1)
            proc[:, en_idx] = cont[:, en_idx] / torch.clamp(cont[:, maxen_idx], min=1)
            keep = [i for i in range(proc.size(1)) if i not in [maxhp_idx, maxen_idx]]
            # Apply keep filtering BEFORE batch norm, same as in forward()
            proc = proc[:, keep]
            cont_norm = self.bn(proc)  # uses running stats; fine for targets
            hunger = torch.clamp(bl[:, 21].long(), 0, 6)
            dung   = torch.clamp(bl[:, 23].long(), 0, 10)
            level  = torch.clamp(bl[:, 24].long(), 0, 50)
            cond_mask = bl[:, 25].long()
            cond_vec = torch.stack([((cond_mask & v) > 0).float() for v in CONDITION_BITS], dim=1)
        return {"cont": cont_norm, "hunger": hunger, "dungeon": dung, "level": level, "cond": cond_vec}

class StatsEncoder(nn.Module):
    def __init__(self, config: VAEConfig, stats: int = BLSTATS_DIM):
        super().__init__()
        self.preprocessor = BlstatsPreprocessor(stats_dim=stats)
        input_dim = self.preprocessor.get_output_dim()  # 43 dimensions after preprocessing
        self.out_dim = config.stat_feat_channels  # 16 output channels
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, config.stat_feat_channels), nn.ReLU(),
        )
        
    def forward(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self.preprocessor(stats)  # [B, 27] -> [B, 43]
        features = self.net(processed)  # [B, 43] -> [B, 16]
        return features
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return self.out_dim

class StatsDecoder(nn.Module):
    """Predict stats with proper likelihoods: continuous (Gaussian), discrete (CE), conditions (BCE)."""
    def __init__(self, config: VAEConfig):
        super().__init__()
        hid = 256
        self.net = nn.Sequential(nn.Linear(config.core_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.mu      = nn.Linear(hid, config.stats_cond_dim)
        self.logvar  = nn.Linear(hid, config.stats_cond_dim)
        self.hunger  = nn.Linear(hid, 7)
        self.dungeon = nn.Linear(hid, 11)
        self.level   = nn.Linear(hid, 51)
        self.cond    = nn.Linear(hid, len(CONDITION_BITS))
    def forward(self, z):
        h = self.net(z)
        return {"mu": self.mu(h), "logvar": self.logvar(h), "hunger": self.hunger(h), "dungeon": self.dungeon(h), "level": self.level(h), "cond": self.cond(h)}

class MessageEncoder(nn.Module):
    def __init__(self, config: VAEConfig,vocab: int=MSG_VSIZE):
        super().__init__()
        self.emb_dim = config.msg_emb
        self.hid_dim = config.msg_hidden
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, self.emb_dim, padding_idx=MSG_PAD)
        self.gru = nn.GRU(self.emb_dim, self.hid_dim, batch_first=True, bidirectional=True)  # bidirectional GRU
        self.out = nn.Linear(self.hid_dim * 2, self.hid_dim)  # output layer to reduce to hid_dim
        self.sos = MSG_SOS  # start-of-sequence token
        self.eos = MSG_EOS  # end-of-sequence token

    def forward(self, msg_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        msg_tokens: [B, 256] padded with 0s
        returns: [B, hid_dim] encoding, [B, 256, emb_dim] embeddings
        """
        # calculate lengths for packing
        lengths = (msg_tokens != MSG_PAD).sum(dim=1)  # [B,] - count non-padding tokens
        # for entries who length is 0, pad a ' ' token
        bempty = (lengths == 0)
        padded_msg_tokens = msg_tokens.clone()  # clone to avoid modifying original
        padded_msg_tokens[bempty, 0] = ord(' ')  # set space token for empty messages
        lengths[bempty] = 1  # set length to 1 for empty messages
        msg_tokens_with_eos = padded_msg_tokens.clone()
        bHasSpace = (lengths < padded_msg_tokens.size(1))  # check if there is space for EOS
        msg_tokens_with_eos[bHasSpace, lengths[bHasSpace]] = self.eos  # set end-of-sequence token at the end of each message
        x = self.emb(msg_tokens_with_eos)  # [B, 256, emb_dim]
        # pack so GRU skips padding
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)
        packed_out, h_n = self.gru(packed) # h_n: [2, B, hid_dim]
        h_fw, h_bw = h_n[0], h_n[1] # forward and backward hidden states
        h = torch.cat([h_fw, h_bw], dim=1) # [B, hid_dim * 2]
        out = self.out(h)  # [B, hid_dim * 2] -> [B, hid_dim]
        return out

    def get_output_channels(self) -> int:
        """Returns the number of output channels."""
        return self.hid_dim
    
class MessageDecoder(nn.Module):
    def __init__(self, config: VAEConfig, L: int = MSG_MAXLEN, vocab: int = MSG_VSIZE):
        super().__init__()
        self.L = L; self.vocab = vocab
        x = torch.linspace(-1, 1, L).view(1,1,L)
        self.register_buffer("pos", x)
        self.z_to_line = nn.Sequential(nn.Linear(config.core_dim, config.msg_ch), nn.ReLU(), nn.Linear(config.msg_ch, config.msg_ch))
        self.stem = nn.Sequential(nn.Conv1d(config.msg_ch+1, config.msg_ch, 3, padding=1), nn.GroupNorm(_gn_groups(config.msg_ch), config.msg_ch), nn.ReLU())
        self.block = nn.Sequential(
            nn.Conv1d(config.msg_ch, config.msg_ch, 3, padding=1, dilation=1),  nn.GroupNorm(_gn_groups(config.msg_ch), config.msg_ch), nn.ReLU(),
            nn.Conv1d(config.msg_ch, config.msg_ch, 3, padding=2, dilation=2),  nn.GroupNorm(_gn_groups(config.msg_ch), config.msg_ch), nn.ReLU(),
            nn.Conv1d(config.msg_ch, config.msg_ch, 3, padding=4, dilation=4),  nn.GroupNorm(_gn_groups(config.msg_ch), config.msg_ch), nn.ReLU(),
            nn.Conv1d(config.msg_ch, config.msg_ch, 3, padding=8, dilation=8),  nn.GroupNorm(_gn_groups(config.msg_ch), config.msg_ch), nn.ReLU(),
        )
        self.head = nn.Conv1d(config.msg_ch, vocab, 1)
    def forward(self, z):
        B = z.size(0)
        line = self.z_to_line(z).unsqueeze(-1).expand(B, -1, self.L)
        x = torch.cat([line, self.pos.expand(B, -1, -1)], dim=1)
        x = self.stem(x)
        x = self.block(x)
        logits = self.head(x).transpose(1, 2)  # [B,L,V]
        return logits
    
    @staticmethod
    def _filter_topk_topp(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """Apply top-k/top-p over the last dim (vocab). logits: [B,L,V]."""
        B, L, V = logits.shape
        out = logits
        if top_k is not None and top_k > 0 and top_k < V:
            kth = torch.topk(out, top_k, dim=-1).values[..., -1, None]
            out = out.masked_fill(out < kth, float('-inf'))
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(out, dim=-1, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove = remove | torch.isneginf(sorted_logits)
            to_remove = torch.zeros_like(out, dtype=torch.bool)
            to_remove.scatter_(-1, sorted_idx, remove)
            out = out.masked_fill(to_remove, float('-inf'))
        return out

    @staticmethod
    def sample_from_logits(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        deterministic: bool = False,
        allow_eos: bool = True,
        forbid_eos_at_start: bool = True,
        allow_pad: bool = False,
        sos_id: int | None = MSG_SOS,
        eos_id: int | None = MSG_EOS,
        pad_id: int | None = MSG_PAD,
    ) -> torch.Tensor:
        """
        Sample token sequences from non-AR message logits.
        Args:
            logits: [B, L, V]
            temperature: softmax temperature (>0). 1.0 = no change
            top_k/top_p: nucleus filtering along vocab dim
            deterministic: if True, returns argmax per position (greedy)
            allow_eos: if True, EOS can be sampled; tokens after first EOS are set to PAD
            forbid_eos_at_start: if True, EOS is disallowed at position 0
            allow_pad: if False, PAD is disallowed before EOS (it will still be inserted after EOS)
            sos_id/eos_id/pad_id: special token ids; set to None to disable masking
        Returns:
            tokens: [B, L] long
        """
        assert temperature > 0, "temperature must be > 0"
        B, L, V = logits.shape
        logit = logits / temperature
        # mask unwanted specials
        if sos_id is not None:
            logit[..., sos_id] = float('-inf')
        if not allow_pad and pad_id is not None:
            logit[..., pad_id] = float('-inf')
        if forbid_eos_at_start and eos_id is not None:
            logit[:, 0, eos_id] = float('-inf')
        # filter
        logit = MessageDecoder._filter_topk_topp(logit, top_k=top_k, top_p=top_p)
        if deterministic:
            tokens = logit.argmax(dim=-1)
        else:
            probs = F.softmax(logit, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            flat = probs.reshape(B * L, V)
            # fallback for degenerate rows (all zeros)
            zero_row = (flat.sum(dim=-1, keepdim=True) == 0)
            if zero_row.any():
                arg = logit.reshape(B*L, V).argmax(dim=-1, keepdim=True)
                one_hot = torch.zeros_like(flat).scatter_(1, arg, 1.0)
                flat = torch.where(zero_row, one_hot, flat)
            idx = torch.multinomial(flat, 1).squeeze(-1)  # [B*L]
            tokens = idx.view(B, L)
        # post-process: after first EOS -> PAD
        if eos_id is not None and pad_id is not None and allow_eos:
            with torch.no_grad():
                is_eos = (tokens == eos_id)
                # first eos position per batch; if none, set to L
                first = torch.where(is_eos.any(dim=1), is_eos.float().argmax(dim=1), torch.full((B,), L, device=tokens.device))
                arange = torch.arange(L, device=tokens.device).view(1, L).expand(B, L)
                after = arange > first.view(B, 1)
                tokens = torch.where(after, torch.full_like(tokens, pad_id), tokens)
        # never emit SOS
        if sos_id is not None:
            tokens = torch.where(tokens == sos_id, torch.full_like(tokens, (eos_id if allow_eos and eos_id is not None else pad_id if pad_id is not None else 0)), tokens)
        return tokens

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convenience: z -> logits -> tokens."""
        logits = self.forward(z)
        return self.sample_from_logits(logits, **kwargs)
    
class HeroEmbedding(nn.Module):
    """Hero embedding for MiniHack."""
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.role_emb = nn.Embedding(ROLE_CAD, config.role_emb)
        self.race_emb = nn.Embedding(RACE_CAD, config.race_emb)
        self.gend_emb = nn.Embedding(GEND_CAD, config.gend_emb)
        self.align_emb = nn.Embedding(ALIGN_CAD, config.align_emb)
        self.out_dim = config.role_emb + config.race_emb + config.gend_emb + config.align_emb

    def forward(self, role: torch.Tensor, race: torch.Tensor,
                gend: torch.Tensor, align: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for hero embedding."""
        role_emb = self.role_emb(role)
        race_emb = self.race_emb(race)
        gend_emb = self.gend_emb(gend)
        # normalise alignment to the range [0, ALIGN_CAD-1]
        align = torch.clamp(align + 1, 0, ALIGN_CAD - 1)  # Ensure alignment is within bounds
        align_emb = self.align_emb(align)
        
        hero_vec = torch.cat([role_emb, race_emb, gend_emb, align_emb], dim=-1)
        return hero_vec  # [B, 16]
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels for hero embedding."""
        return self.out_dim  # 16

class ActionAuxHead(nn.Module):
    """Action auxiliary head for VAE."""
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.passability = MLP(cfg.latent_dim, cfg.forward_hidden, cfg.passability_dirs, 3, cfg.decoder_dropout)
        self.safety = MLP(cfg.latent_dim, cfg.forward_hidden, cfg.passability_dirs, 3, cfg.decoder_dropout)
        self.goal = MLP(cfg.latent_dim, cfg.forward_hidden, cfg.goal_dir_dim, 2, cfg.decoder_dropout) if cfg.goal_dir_dim > 0 else None

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass for action value head.
        
        Args:
            z: [B, latent_dim] latent vector
        
        Returns:
            Dictionary with outputs:
            - passability: [B, passability_dirs]
            - safety: [B, passability_dirs]
            - goal: [B, goal_dir_dim] (if applicable)
        """
        out = {
            "passability_logits": self.passability(z),
            "safety_logits": self.safety(z),
        }
        if self.goal is not None:
            out["goal_pred"] = torch.tanh(self.goal(z))  # in [-1,1]
        return out

class WorldModel(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.enabled = (cfg.action_dim > 0) and cfg.use_world_model
        self.skill_num = cfg.skill_num
        if not self.enabled:
            return
        self.residual_alpha = nn.Parameter(torch.tensor(0.5))  # learnable residual scale
        self.skill_belief = MLP(cfg.latent_dim, cfg.forward_hidden, cfg.skill_num, 2, cfg.decoder_dropout) if cfg.skill_num > 0 else None
        self.reward = MLP(cfg.latent_dim + cfg.action_dim + cfg.skill_num, cfg.forward_hidden, 1, 2, cfg.decoder_dropout)
        self.done = MLP(cfg.latent_dim + cfg.action_dim + cfg.skill_num, cfg.forward_hidden, 1, 2, cfg.decoder_dropout)
        self.value_k = MLP(cfg.latent_dim + cfg.action_dim + cfg.skill_num, cfg.forward_hidden, len(cfg.value_horizons), 2, cfg.decoder_dropout)
        # predict delta; final pred = z + alpha * tanh(delta)
        self.z_delta = MLP(cfg.latent_dim + cfg.action_dim + cfg.skill_num, cfg.forward_hidden, cfg.latent_dim, num_layers=3)

    def forward(self, z: torch.Tensor, action_onehot: torch.Tensor, h_onehot: torch.Tensor = None) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("Forward model disabled (action_dim=0).")
        if self.skill_num > 0:
            assert h_onehot is not None and h_onehot.size(1) == self.skill_num, \
                f"Expected h_onehot of shape [B, {self.skill_num}], got {h_onehot.shape}"
            action_onehot = torch.cat([action_onehot, h_onehot], dim=1)  # [B, A + skill_num]
        x = torch.cat([z, action_onehot], dim=1)
        delta = torch.tanh(self.z_delta(x))
        latent_pred = z + self.residual_alpha * delta
        out = {
            "reward": self.reward(x),
            "done_logits": self.done(x),
            "value_k": self.value_k(x),
            "latent_pred" : latent_pred,
        }
        if self.skill_belief is not None:
            out["skill_logits"] = self.skill_belief(z)
        return out


class InverseDynamics(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.enabled = (cfg.action_dim > 0) and cfg.use_inverse_dynamics
        self.skill_num = cfg.skill_num
        if not self.enabled:
            return
        self.net = MLP(2 * cfg.latent_dim, 256, cfg.action_dim + cfg.skill_num, num_layers=3)

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("Inverse dynamics disabled.")
        x = torch.cat([z_t, z_tp1], dim=1)
        return self.net(x)  # logits [B, A + skill_num]

class MultiModalHackVAE(nn.Module):
    def __init__(self, config: VAEConfig,
                 logger: logging.Logger | None=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        
        self.map_encoder = MapEncoder(config)
        self.stats_encoder = StatsEncoder(config)
        self.msg_encoder = MessageEncoder(config)
        self.hero_emb = HeroEmbedding(config)

        fusion_in = 2 * self.map_encoder.get_output_channels() + \
                    self.stats_encoder.get_output_channels() + \
                    self.msg_encoder.get_output_channels() + \
                    self.hero_emb.get_output_channels() # cnn + stats + msg + hero embedding
        
        # Add dropout to the encoder fusion layer if enabled
        if config.encoder_dropout > 0.0:
            self.to_latent = nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Dropout(config.encoder_dropout),
                nn.Linear(fusion_in, 256), nn.ReLU(),
                nn.Dropout(config.encoder_dropout),
            )
        else:
            self.to_latent = nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Linear(fusion_in, 256), nn.ReLU(),
            )
        self.latent_dim = config.latent_dim
        self.lowrank_dim = config.low_rank
        self.z_bag_dim = config.bag_dim  # Add missing attribute
        self.z_core_dim = config.core_dim  # Add missing attribute
        self.mu_head     = nn.Linear(256, self.latent_dim)
        self.logvar_diag_head = nn.Linear(256, self.latent_dim)  # diagonal part
        self.lowrank_factor_head = nn.Linear(256, self.latent_dim * self.lowrank_dim) if self.lowrank_dim else None # low-rank factors

        self.z_norm = torch.nn.LayerNorm(self.latent_dim, elementwise_affine=False)
        self.map_decoder  = MapDecoder(config)
        self.stats_decoder = StatsDecoder(config)
        self.msg_decoder   = MessageDecoder(config)
        self.action_aux_head = ActionAuxHead(config)
        self.world_model = WorldModel(config)
        self.inverse_dynamics = InverseDynamics(config)
        

    def encode(self, glyph_chars, glyph_colors, blstats, msg_tokens, hero_info):
        """
        Encodes the input features into a latent space.
        
        Args:
            glyph_chars: [B, 21, 79] - character glyphs
            glyph_colors: [B, 21, 79] - color glyphs
            blstats: [B, BLSTATS_DIM] - baseline stats
            msg_tokens: [B, 256] - message tokens (padded)
            hero_info: [B, 4] - hero information
                    
        Returns:
            mu: [B, LATENT_DIM] - mean of the latent distribution
            logvar_diag: [B, LATENT_DIM] - diagonal log variance of the latent distribution  
            lowrank_factors: [B, LATENT_DIM, LOW_RANK] - low-rank factors if applicable
        """
        
        # glyphs
        if glyph_chars.size(0) != glyph_colors.size(0):
            raise ValueError("glyph_chars and glyph_colors must have the same batch size.")
        B = glyph_chars.size(0)
        if glyph_chars.size(1) != glyph_colors.size(1):
            raise ValueError("glyph_chars and glyph_colors must have the same height and width.")
        
        # Encode all features and get embeddings
        glyph_feat = self.map_encoder(glyph_chars, glyph_colors)  # [B,64]
        
        # rare glyph features
        rare_glyph_chars = torch.full_like(glyph_chars, ord(' '))  # fill with space character
        rare_glyph_colors = torch.zeros_like(glyph_colors)  # fill with black color
        is_fg = (glyph_chars != ord(' ')) & (glyph_colors != 0)  # foreground mask
        is_common = torch.zeros_like(is_fg, dtype=torch.bool)  # common glyphs mask
        for c in COMMON_GLYPHS:
            is_common |= (glyph_chars == c[0]) & (glyph_colors == c[1])
        is_rare = is_fg & ~is_common  # rare glyphs mask
        rare_glyph_chars[is_rare] = glyph_chars[is_rare]
        rare_glyph_colors[is_rare] = glyph_colors[is_rare]
        rare_glyph_feat = self.map_encoder(rare_glyph_chars, rare_glyph_colors)  # [B,64]

        stats_feat = self.stats_encoder(blstats)
        
        # Message encoding
        msg_feat = self.msg_encoder(msg_tokens)  # [B,hid_dim], [B,256,emb_dim]
        
        # Hero embedding
        # check shape of hero_info should be [B, 4] with role, race, gend, align
        if hero_info.size(0) != B:
            raise ValueError("hero_info must have the same batch size as glyph_chars and glyph_colors.")
        if hero_info.size(1) != 4:
            raise ValueError("hero_info must have shape [B, 4]")
        hero_info = hero_info.long()
        role, race, gend, align = hero_info[:, 0], hero_info[:, 1], hero_info[:, 2], hero_info[:, 3]
        hero_feat = self.hero_emb(role, race, gend, align)  # [B,16], dict
        features = [glyph_feat, rare_glyph_feat, stats_feat, msg_feat, hero_feat]
        
        fused = torch.cat(features, dim=-1)
        
        # Encode to latent space
        h = self.to_latent(fused)
        mu = self.mu_head(h)
        logvar_diag = self.logvar_diag_head(h)
        lowrank_factors = self.lowrank_factor_head(h).view(B, self.latent_dim, self.lowrank_dim) if self.lowrank_factor_head is not None else None

        return {
            'mu': mu,  # [B, LATENT_DIM]
            'logvar': logvar_diag,  # [B, LATENT_DIM]
            'lowrank_factors': lowrank_factors,  # [B, LATENT_DIM, LOW_RANK] or None
            "stats_encoder": self.stats_encoder
        }
          
    def _reparameterise(self, mu, logvar, lowrank_factors=None):
        diag_std = torch.exp(0.5*logvar)
        eps1 = torch.randn_like(diag_std)
        z = mu + diag_std*eps1
        if lowrank_factors is not None:
            # If low-rank factors are provided, combine them with the diagonal std
            eps2 = torch.randn((lowrank_factors.size(0), lowrank_factors.size(2)), device=lowrank_factors.device)  # [B, RANK]
            # Assuming lowrank_factors is [B, LATENT_DIM, RANK]
            lowrank_std = torch.bmm(lowrank_factors, eps2.unsqueeze(-1)).squeeze(-1)  # [B, LATENT_DIM]
            z += lowrank_std
        return z    # [B, LATENT_DIM]

    def decode(self, z, action_onehot=None, z_next_detach=None, z_current_for_dynamics=None):
        # split z -> [z_bag | z_core]
        z_n    = self.z_norm(z)
        z_core = z_n[:, self.z_bag_dim:]

        out_map = self.map_decoder(z_n)  # decode map features
        stats_pred = self.stats_decoder(z_core)
        msg_logits = self.msg_decoder(z_core)
        out_aux = self.action_aux_head(z_n)
        # Keep dynamics in the same normalized coordinate system
        z_cur_n  = self.z_norm(z_current_for_dynamics) if z_current_for_dynamics is not None else None
        z_next_n = self.z_norm(z_next_detach)          if z_next_detach is not None else None
        out_world = self.world_model(z_cur_n, action_onehot) if action_onehot is not None and z_cur_n is not None and self.world_model.enabled else None
        inverse_dynamics_logits = self.inverse_dynamics(z_cur_n, z_next_n) if z_next_n is not None and z_cur_n is not None and self.inverse_dynamics.enabled else None
        
        out = {
            **out_map,
            "stats_pred": stats_pred, 
            "msg_logits": msg_logits, 
            **out_aux,
            "inverse_dynamics_logits": inverse_dynamics_logits,  # [B * (T-1), A + skill_num] or None
        }
        
        if out_world is not None:
            out.update(out_world)

        return out

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        glyph_chars = batch['game_chars'] # [B, 21, 79]
        glyph_colors = batch['game_colors'] # [B, 21, 79]
        blstats = batch['blstats'] # [B, BLSTATS_DIM]
        msg_tokens = batch['message_chars'] # [B, 256]
        hero_info = batch['hero_info'] # [B, 4]
        enc = self.encode(glyph_chars, glyph_colors, blstats, msg_tokens, hero_info)
        z = self._reparameterise(enc["mu"], enc["logvar"])  # [B,D]
        
        # Calculate z_next_detach if we have sequential data and save to batch
        if 'has_next' in batch and 'original_batch_shape' in batch:
            B, T = batch['original_batch_shape']
            if T > 1:
                # Reshape z to [B, T, latent_dim] to work with sequences
                z_reshaped = z.view(B, T, -1)  # [B, T, latent_dim]
                
                # Create z_next by shifting: z_next[t] = z[t+1]
                # Only valid for t < T-1, so we exclude the last timestep
                z_next = z_reshaped[:, 1:, :].contiguous()  # [B, T-1, latent_dim]
                z_current_for_dynamics = z_reshaped[:, :-1, :].contiguous()  # [B, T-1, latent_dim]
                
                # Flatten back - save whole tensors and let vae_loss use has_next mask
                z_next_flat = z_next.view(-1, z_next.shape[-1])  # [B*(T-1), latent_dim]
                z_current_for_dynamics_flat = z_current_for_dynamics.view(-1, z_current_for_dynamics.shape[-1])  # [B*(T-1), latent_dim]
                
                # Save whole tensors to batch - let vae_loss handle masking with has_next
                z_next_flat_detach = z_next_flat.detach()  # [B*(T-1), latent_dim]
                
                action_onehot = batch.get('action_onehot')
                action_onehot_reshaped = action_onehot.view(B, T, -1) if action_onehot is not None else None
                action_onehot_for_dynamics = action_onehot_reshaped[:, :-1, :].contiguous() if action_onehot_reshaped is not None else None  # [B, T-1, A]
                action_onehot_for_dynamics_flat = action_onehot_for_dynamics.view(-1, action_onehot_for_dynamics.shape[-1]) if action_onehot_for_dynamics is not None else None  # [B*(T-1), A]
                has_next = batch.get('has_next')
                has_next_reshaped = has_next.view(B, T)
                has_next_for_dynamics = has_next_reshaped[:, :-1].contiguous()
                has_next_for_dynamics_flat = has_next_for_dynamics.view(-1)  # [B*(T-1)]
                rewards = batch.get('reward_target')
                rewards_reshaped = rewards.view(B, T)
                rewards_for_dynamics = rewards_reshaped[:, 1:].contiguous()
                rewards_for_dynamics_flat = rewards_for_dynamics.view(-1)  # [B*(T-1)]
                dones = batch.get('done')
                dones_reshaped = dones.view(B, T)
                dones_for_dynamics = dones_reshaped[:, 1:].contiguous()
                dones_for_dynamics_flat = dones_for_dynamics.view(-1)  # [B*(T-1)]
            else:
                z_next_flat_detach = None  # No next state if T <= 1
                z_current_for_dynamics_flat = None  # No current state if T <= 1
                action_onehot_for_dynamics_flat = None  # No action if T <= 1
                has_next_for_dynamics_flat = None  # No has_next if T <= 1
                rewards_for_dynamics_flat = None  # No rewards if T <= 1
                dones_for_dynamics_flat = None  # No dones if T <= 1
        else:
            z_next_flat_detach = None  # No next state if T <= 1
            z_current_for_dynamics_flat = None  # No current state if T <= 1
            action_onehot_for_dynamics_flat = None  # No action if T <= 1
            has_next_for_dynamics_flat = None  # No has_next if T <= 1
            rewards_for_dynamics_flat = None  # No rewards if T <= 1
            dones_for_dynamics_flat = None  # No dones if T <= 1
                
        batch['z_next_detached'] = z_next_flat_detach  # [B*(T-1), latent_dim]
        batch['action_onehot_for_dynamics'] = action_onehot_for_dynamics_flat  # [B*(T-1), A]
        batch['has_next_for_dynamics'] = has_next_for_dynamics_flat  # [B*(T-1)]
        batch['rewards_for_dynamics'] = rewards_for_dynamics_flat  # [B*(T-1)]
        batch['dones_for_dynamics'] = dones_for_dynamics_flat  # [B*(T-1)]
        
        dec = self.decode(z, action_onehot=action_onehot_for_dynamics_flat, z_next_detach=z_next_flat_detach, z_current_for_dynamics=z_current_for_dynamics_flat)
        return {**enc, **dec, 'z': z}  # include z in the output
    
    @torch.no_grad()
    def sample(
        self,
        glyph_chars: torch.Tensor | None = None,
        glyph_colors: torch.Tensor | None = None,
        blstats: torch.Tensor | None = None,
        msg_tokens: torch.Tensor | None = None,
        hero_info: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        use_mean: bool = True,
        include_logits: bool = False,
        # map sampling
        map_temperature: float = 1.0,
        map_occ_thresh: float = 0.5,
        bag_presence_thresh: float = 0.5,
        hero_presence_thresh: float = 0.5,
        passability_thresh: float = 0.5,
        safety_thresh: float = 0.5,
        map_deterministic: bool = True,
        glyph_top_k: int = 0,
        glyph_top_p: float = 1.0,
        color_top_k: int = 0,
        color_top_p: float = 1.0,
        class_top_k: int = 0,
        class_top_p: float = 1.0,
        # message sampling
        msg_temperature: float = 1.0,
        msg_top_k: int = 0,
        msg_top_p: float = 1.0,
        msg_deterministic: bool = True,
        allow_eos: bool = True,
        forbid_eos_at_start: bool = True,
        allow_pad: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Convenience sampler.
        If `z` is None, encodes provided inputs and samples z (or takes mean if `use_mean`).
        Returns a dict with hard map (`fg_mask, chars, colors`), tokenized message, and optional logits.
        """
        assert (z is not None) or (glyph_chars is not None and glyph_colors is not None and blstats is not None and msg_tokens is not None), \
            "Provide either z or the full set of inputs to encode."

        if z is None:
            enc = self.encode(glyph_chars, glyph_colors, blstats, msg_tokens, hero_info)
            mu, logvar = enc["mu"], enc["logvar"]
            z = mu if use_mean else self._reparameterise(mu, logvar)
        else:
            enc = None

        B = z.size(0)
        
        dec = self.decode(z)

        # --- Map sampling ---
        ego_samples = MapDecoder.sample_ego_from_logits(
            dec["ego_char_logits"], dec["ego_color_logits"], dec["ego_class_logits"],
            temperature=map_temperature,
            deterministic=map_deterministic,
            glyph_top_k=glyph_top_k, glyph_top_p=glyph_top_p,
            color_top_k=color_top_k, color_top_p=color_top_p,
            class_top_k=class_top_k, class_top_p=class_top_p
        )
        
        occupy_logit = dec["occupy_logits"]
        hero_presence_logit = dec["hero_presence_logits"]
        bag_presence_logit = dec["bag_logits"]
        passability_logit = dec["passability_logits"]
        safety_logit = dec["safety_logits"]
        hero_centroid = dec["hero_centroid"]
        if map_deterministic:
            occupy = (torch.sigmoid(occupy_logit) > map_occ_thresh).float()  # [B,1,21,79]
            has_hero = (torch.sigmoid(hero_presence_logit) > hero_presence_thresh)  # [B]
            bag_presence = (torch.sigmoid(bag_presence_logit) > bag_presence_thresh)  # [B, GLYPH_DIM]
            passability_presence = (torch.sigmoid(passability_logit) > passability_thresh).float()  # [B, 8]
            safety_presence = (torch.sigmoid(safety_logit) > safety_thresh).float()  # [B, 8]
        else:
            occupy = torch.bernoulli(torch.sigmoid(occupy_logit))  # [B,1,21,79]
            has_hero = torch.bernoulli(torch.sigmoid(hero_presence_logit))  # [B]
            bag_presence = torch.bernoulli(torch.sigmoid(bag_presence_logit))  # [B, GLYPH_DIM]
            passability_presence = torch.bernoulli(torch.sigmoid(passability_logit))  # [B, 8]
            safety_presence = torch.bernoulli(torch.sigmoid(safety_logit))  # [B, 8]
        
        bag_sets = bag_presence_to_glyph_sets(bag_presence)  # [B, bag_dim]
        pass_safety_dict = MapDecoder.format_passability_safety(passability_presence, safety_presence)  # [B, 21, 79], [B, 21, 79]

        # --- Message sampling ---
        msg_tokens_out = MessageDecoder.sample_from_logits(
            dec["msg_logits"],
            temperature=msg_temperature,
            top_k=msg_top_k, top_p=msg_top_p,
            deterministic=msg_deterministic,
            allow_eos=allow_eos,
            forbid_eos_at_start=forbid_eos_at_start,
            allow_pad=allow_pad,
            sos_id=MSG_SOS, eos_id=MSG_EOS, pad_id=MSG_PAD,
        )

        # --- Stats point predictions (no sampling) ---
        sp = dec["stats_pred"]
        stats_point = {
            "cont_mu": sp["mu"],
            "hunger": sp["hunger"].argmax(dim=-1),
            "dungeon": sp["dungeon"].argmax(dim=-1),
            "level": sp["level"].argmax(dim=-1),
            "cond": (torch.sigmoid(sp["cond"]) > 0.5),
        }

        out = {
            "occupancy_mask": occupy,
            "hero_presence": has_hero,
            "hero_centroid": hero_centroid,
            **ego_samples,
            "msg_tokens": msg_tokens_out,
            "stats_point": stats_point,
            "bag_sets": bag_sets,
            "z": z,
            **pass_safety_dict
        }
        if include_logits:
            out.update({
                "occupy_logits": occupy_logit,
                "ego_char_logits": dec["ego_char_logits"],
                "ego_color_logits": dec["ego_color_logits"],
                "ego_class_logits": dec["ego_class_logits"],
                "hero_presence_logits": hero_presence_logit,
                "hero_centroid": hero_centroid,
                "msg_logits": dec["msg_logits"],
                "bag_logits": bag_presence_logit,
                "passability_logits": passability_logit,
                "safety_logits": safety_logit,
            })
        return out
        

# ------------------------- loss helpers ------------------------------ #

def _bce_focal_with_logits(logits, target, alpha_pos=0.25, gamma=2.0, reduction='mean'):
    # logits, target: [B,H,W]
    p = torch.sigmoid(logits)
    p_t = torch.where(target > 0.5, p, 1.0 - p)
    alpha_t = torch.where(target > 0.5, torch.full_like(target, alpha_pos), torch.full_like(target, 1.0 - alpha_pos))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def _message_ce_loss(logits: torch.Tensor, tgt: torch.Tensor, pad_id: int = MSG_PAD, eos_id: int = MSG_EOS) -> torch.Tensor:
    # logits: [B,L,V], tgt: [B,L]
    B, L, V = logits.shape
    with torch.no_grad():
        pad_mask = (tgt == pad_id)
        if eos_id is not None:
            eos_pos = (tgt == eos_id).float()
            after = torch.cumsum(eos_pos, dim=1) > 0
            after[:, 0] = False
            mask = pad_mask | after
        else:
            mask = pad_mask
    ce = F.cross_entropy(logits.reshape(B*L, V), tgt.reshape(B*L), reduction='none').view(B, L)
    keep = (~mask).float()
    return (ce * keep).sum() / B

@torch.no_grad()
def jaccard_index_from_logits(
    logits: torch.Tensor,          # [N, 1, H, W]
    targets: torch.Tensor,         # [N, 1, H, W] float/bool {0,1}
    mask: torch.Tensor | None = None,  # [N, 1, H, W] bool (optional)
    threshold: float = 0.5
) -> float:
    """
    Returns: Jaccard index (float)
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    if mask is not None:
        m = mask.float()
        intersection = ((preds * targets).float() * m).sum()
        union = (((preds + targets) >= 1.0).float() * m).sum()
    else:
        intersection = (preds * targets).float().sum()
        union = ((preds + targets) >= 1.0).float().sum()
    return intersection / union.clamp_min(1.0e-8)

@torch.no_grad()
def ego_metrics_from_logits(
    logits: torch.Tensor,          # [N, C, k, k]
    targets: torch.Tensor,         # [N, k, k] (long)
    mask: torch.Tensor | None = None,  # [N] or [N,k,k] or [N,1,k,k] (bool)
    ignore_index: int | None = None,
    topk: tuple[int, ...] = (1, 3),
    ece_bins: int = 15,
) -> dict:
    """
    Computes robust multi-class metrics for imbalanced pixel labels.

    Returns scalars:
      - acc_top1, acc_topK (for K in topk)
      - ece
    """
    assert logits.dim() == 4 and targets.dim() == 3, "Shapes: logits [N,C,k,k], targets [N,k,k]"
    N, C, k, k2 = logits.shape
    assert k == k2 and targets.shape == (N, k, k)

    # --- build a pixel mask ---
    if mask is None:
        pix_mask = torch.ones((N, 1, k, k), dtype=torch.bool, device=logits.device)
    else:
        if mask.dim() == 1:      # [N]
            pix_mask = mask.view(N, 1, 1, 1).expand(N, 1, k, k)
        elif mask.dim() == 3:    # [N,k,k]
            pix_mask = mask.unsqueeze(1).to(dtype=torch.bool)
        elif mask.dim() == 4:    # [N,1,k,k] or [N,?,k,k]
            pix_mask = mask[:, :1].to(dtype=torch.bool)
        else:
            raise ValueError(f"Bad mask shape: {tuple(mask.shape)}")

    # Ignore index mask
    if ignore_index is not None:
        ig = (targets == ignore_index).unsqueeze(1)  # [N,1,k,k]
        pix_mask = pix_mask & (~ig)

    valid = pix_mask.squeeze(1)                      # [N,k,k] bool
    num_valid = valid.sum()
    if num_valid == 0:
        returned_metrics = {}
        for K in topk:
            if K <= 1:
                returned_metrics['acc_top1'] = 0.0
            else:
                returned_metrics[f'acc_top{K}'] = 0.0
        returned_metrics['ece'] = 0.0
        return returned_metrics

    # --- flatten valid pixels ---
    t = targets[valid]                               # [M]
    l = logits.permute(0, 2, 3, 1)[valid]            # [M,C]
    probs = F.softmax(l, dim=1)                      # [M,C]
    pred_top1 = probs.argmax(dim=1)                  # [M]               

    # --- top-k accuracy ---
    metrics = {}
    for K in topk:
        if K <= 1:
            acc = (pred_top1 == t).float().mean().item()
            metrics[f'acc_top1'] = acc
        else:
            topk_idx = probs.topk(K, dim=1).indices   # [M,K]
            correct = (topk_idx == t.view(-1,1)).any(dim=1).float().mean().item()
            metrics[f'acc_top{K}'] = correct
    
    # --- ECE (Expected Calibration Error) on top-1 probs ---
    conf, pred = probs.max(dim=1)            # [M]
    correct = (pred == t).float()
    # bin by confidence
    bin_ids = torch.clamp((conf * ece_bins).long(), min=0, max=ece_bins-1)
    ece = torch.tensor(0.0, device=logits.device)
    for b in range(ece_bins):
        m = (bin_ids == b)
        n_b = m.sum()
        if n_b > 0:
            acc_b = correct[m].mean()
            conf_b = conf[m].mean()
            ece += (n_b.float() / conf.numel()) * (acc_b - conf_b).abs()
    metrics['ego/ece'] = ece.item()

    return metrics

@torch.no_grad()
def binary_accuracy_from_logits(
    logits: torch.Tensor,              # [N, K]
    targets: torch.Tensor,             # [N, K] in {0,1}
    mask: torch.Tensor | None = None,  # [N] or [N,K] (0=ignore)
    weight: torch.Tensor | None = None,# [N] or [N,K] (importance)
    threshold: List[float] = [0.4, 0.6],
) -> dict:
    """
    Weighted + masked binary accuracy.
    - mask: elements with mask==0 are ignored (hard mask).
    - weight: per-element importance (e.g., down-weight 'unknown' tiles).
    Returns:
        {
          'acc': float,               # micro accuracy over all elements
          'acc_per_dim': Tensor[K],   # weighted accuracy per direction
          'support_per_dim': Tensor[K]# effective weight per direction
        }
    """
    # predictions
    probs = torch.sigmoid(logits)
    preds = torch.where(probs <= threshold[0], torch.zeros_like(probs),
            torch.where(probs >= threshold[1], torch.ones_like(probs),
                torch.full_like(probs, 0.5))).to(torch.float32)
    tgt   = targets.to(torch.float32)

    # build combined weights W: same shape as logits [N,K]
    W = torch.ones_like(preds, dtype=torch.float32)
    if mask is not None:
        m = mask.to(torch.float32)
        if m.dim() == 1: m = m.unsqueeze(1)           # [N,1] -> [N,K] by broadcast
        W = W * m
    if weight is not None:
        w = weight.to(torch.float32)
        if w.dim() == 1: w = w.unsqueeze(1)
        W = W * w

    # correctness matrix
    corr = (preds == tgt).to(torch.float32)

    # micro accuracy
    num = (corr * W).sum()
    den = W.sum().clamp_min(1e-6)
    acc_micro = (num / den).item()

    # per-dimension accuracy
    num_d = (corr * W).sum(dim=0)                 # [K]
    den_d = W.sum(dim=0).clamp_min(1e-6)          # [K]
    acc_per_dim = (num_d / den_d).detach().cpu()

    return {
        'acc': acc_micro,
        'acc_per_dim': acc_per_dim,
        'support_per_dim': den_d.detach().cpu(),
    }
    
@torch.no_grad()
def goal_metrics(
    pred_vec: torch.Tensor,     # [N, 2] (x,y) in [-1,1], typically tanh output
    tgt_vec: torch.Tensor,      # [N, 2] (x,y) in [-1,1]
    mask: torch.Tensor | None = None,  # [N] bool
    map_W: int = 79, map_H: int = 21
) -> dict:
    """
    Computes:
      - mae_tiles_x/y: per-axis MAE in *tiles*
      - mae_tiles_L2: Euclidean MAE in tiles
      - angle_mae_deg / angle_med_deg: angular error in degrees
    """
    p = pred_vec
    t = tgt_vec
    if mask is not None:
        m = mask.view(-1,1).float()
        denom = m.sum().clamp_min(1.0)
        p = p * m; t = t * m
    else:
        denom = torch.tensor(float(p.shape[0]), device=p.device)

    # scale normalized [-1,1] vectors to tile units (half-extent = W/2, H/2)
    dx_tiles = (p[:,0] - t[:,0]) * (map_W - 1)
    dy_tiles = (p[:,1] - t[:,1]) * (map_H - 1)

    mae_x = dx_tiles.abs().sum() / denom
    mae_y = dy_tiles.abs().sum() / denom
    mae_L2 = torch.sqrt(dx_tiles**2 + dy_tiles**2).sum() / denom

    # angle error (wrap to [-180, 180])
    ang_p = torch.atan2(p[:,1].clamp(-1,1), p[:,0].clamp(-1,1))  # radians
    ang_t = torch.atan2(t[:,1].clamp(-1,1), t[:,0].clamp(-1,1))
    d = (ang_p - ang_t) * (180.0 / math.pi)
    d = ( (d + 180.0) % 360.0 ) - 180.0
    if mask is not None:
        m1 = mask.float()
        angle_mae = d.abs().sum() / m1.sum().clamp_min(1.0)
        angle_med = d.abs()[m1.bool()].median() if m1.sum() > 0 else torch.tensor(0.0)
    else:
        angle_mae = d.abs().mean()
        angle_med = d.abs().median()

    return {
        'goal_mae_tiles_x': mae_x.item(),
        'goal_mae_tiles_y': mae_y.item(),
        'goal_mae_tiles_L2': mae_L2.item(),
        'goal_angle_mae_deg': angle_mae.item(),
        'goal_angle_med_deg': angle_med.item(),
    }
    
@torch.no_grad()
def dynamics_metrics(
    z_next_pred: torch.Tensor,   # [P, Z]
    z_next_true: torch.Tensor,   # [P, Z] (detached target)
) -> dict:
    """
    Returns mean/median cosine similarity and R^2 (overall and per-dim mean).
    """
    # cosine per pair
    cos = F.cosine_similarity(z_next_pred, z_next_true, dim=1)  # [P]
    cos_mean = cos.mean().item()
    cos_median = cos.median().item()

    # R^2 overall (flatten across dims)
    y = z_next_true
    y_hat = z_next_pred
    y_bar = y.mean()
    ss_res = ((y - y_hat)**2).sum()
    ss_tot = ((y - y_bar)**2).sum().clamp_min(1e-8)
    r2_overall = (1.0 - ss_res / ss_tot).item()

    # R^2 per-dimension then averaged (more robust to dim scaling)
    y_mean_dim = y.mean(dim=0, keepdim=True)
    ss_res_d = ((y - y_hat)**2).sum(dim=0)
    ss_tot_d = ((y - y_mean_dim)**2).sum(dim=0).clamp_min(1e-8)
    r2_per_dim = (1.0 - ss_res_d / ss_tot_d)
    r2_mean_dim = r2_per_dim.mean().item()
    r2_median_dim = r2_per_dim.median().item()

    return {
        'dyn_cosine_mean': cos_mean,
        'dyn_cosine_median': cos_median,
        'dyn_r2_overall': r2_overall,
        'dyn_r2_mean_dim': r2_mean_dim,
        'dyn_r2_median_dim': r2_median_dim
    }

def _info_nce(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Batch InfoNCE with in-batch negatives; q and k are [N,D]."""
    q = F.normalize(q, dim=1)
    k = F.normalize(k.detach(), dim=1)   # stop-grad on targets
    logits = (q @ k.t()) / max(temperature, 1e-6)  # [N,N]
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)

def vae_loss(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: VAEConfig,
    mi_beta: float = 1.0,
    tc_beta: float = 1.0,
    dw_beta: float = 1.0
    ):
    """
    VAE loss with separate embedding and raw reconstruction losses with free-bits KL and Gaussian TC proxy.
    """
    # Extract inputs from batch
    glyph_chars = batch['game_chars']  # [B, 21, 79]
    glyph_colors = batch['game_colors']  # [B, 21, 79]
    blstats = batch['blstats'] # [B, BLSTATS_DIM]
    msg_tokens = batch['message_chars']  # [B, 256]
    valid_screen = batch['valid_screen']  # [B] boolean mask for valid samples
    # Determine batch dimensions
    total_samples = len(valid_screen)
    
    if 'original_batch_shape' in batch:
        B, T = batch['original_batch_shape']
    else:
        # Fallback: try to infer from data
        B = batch.get('batch_size', 1)
        T = total_samples // B if B > 0 else total_samples
    
    # ============= On-the-fly Goal and Value Target Computation =============
    # Compute goal_target, goal_mask, and value_k_target on-the-fly using config parameters
    if 'game_chars' in batch and 'game_colors' in batch and 'reward_target' in batch and 'done_target' in batch:
        if T > 1:
            # Reshape data back to [B, T, ...] for goal and value computation
            game_chars_reshaped = batch['game_chars'].view(B, T, 21, 79)  # [B, T, H, W]
            game_colors_reshaped = batch['game_colors'].view(B, T, 21, 79)  # [B, T, H, W]
            hero_info_reshaped = batch['hero_info'].view(B, T, 4)  # [B, T, 4]
            reward_reshaped = batch['reward_target'].view(B, T)  # [B, T]
            done_reshaped = batch['done_target'].view(B, T)  # [B, T]
            
            # Initialize goal and value targets
            goal_target_flat = torch.zeros(total_samples, 2, dtype=torch.float32, device=batch['game_chars'].device)
            goal_mask_flat = torch.zeros(total_samples, dtype=torch.float32, device=batch['game_chars'].device)
            value_k_target_flat = torch.zeros(B*(T-1), len(config.value_horizons), dtype=torch.float32, device=batch['game_chars'].device)
            value_k_mask_flat = torch.zeros(B*(T-1), len(config.value_horizons), dtype=torch.float32, device=batch['game_chars'].device)
            
            # Process each game sequence
            for g in range(B):
                # Compute k-step value returns for this game sequence
                rewards_np = reward_reshaped[g].cpu().numpy()
                done_np = done_reshaped[g].cpu().numpy().astype(bool)
                
                value_targets, value_masks = discounted_k_step_multi_with_mask(rewards_np, done_np, config.value_horizons, config.gamma)

                # Stack and convert to tensor [T, num_horizons]
                value_targets = torch.tensor(value_targets, dtype=torch.float32, device=batch['game_chars'].device)
                value_masks = torch.tensor(value_masks, dtype=torch.float32, device=batch['game_chars'].device)

                # Fill value targets for this game
                start_idx = g * (T-1)
                end_idx = start_idx + (T-1)
                value_k_target_flat[start_idx:end_idx] = value_targets[:-1,:]  # Exclude last timestep which has no next
                value_k_mask_flat[start_idx:end_idx] = value_masks[:-1,:]
                
                # Compute goal targets for each timestep in this game
                for t in range(T):
                    flat_idx = g * T + t
                    
                    # Skip if not a valid screen
                    if not valid_screen[flat_idx]:
                        continue
                        
                    chars_map = game_chars_reshaped[g, t]  # [H, W]
                    colors_map = game_colors_reshaped[g, t]  # [H, W]
                    
                    # Find hero position
                    hero_pos = (chars_map == ord('@')).nonzero(as_tuple=True)
                    if len(hero_pos[0]) > 0:
                        hero_y, hero_x = int(hero_pos[0][0]), int(hero_pos[1][0])
                        
                        # Compute goal vector using config parameters
                        goal_vec = nearest_stairs_vector(chars_map, hero_y, hero_x, prefer=config.goal_prefer)
                        if goal_vec is not None:
                            goal_target_flat[flat_idx, 0] = goal_vec[0]  # dy
                            goal_target_flat[flat_idx, 1] = goal_vec[1]  # dx
                            goal_mask_flat[flat_idx] = 1.0
                        # else: goal_target remains zero, goal_mask remains 0.0
            
            # Add computed targets to batch
            batch['goal_target'] = goal_target_flat
            batch['goal_mask'] = goal_mask_flat  
            batch['value_k_target'] = value_k_target_flat
            batch['value_k_mask'] = value_k_mask_flat

    # stats
    pre = model_output['stats_encoder'].preprocessor
    pre = pre.to(blstats.device)
    pre.eval()
    with torch.no_grad():
        t = pre.targets(blstats[valid_screen])
    # Extract outputs from model
    mu = model_output['mu'][valid_screen]  # [valid_B, LATENT_DIM]
    logvar = model_output['logvar'][valid_screen]
    lowrank_factors = model_output['lowrank_factors']
    if lowrank_factors is not None:
        lowrank_factors = lowrank_factors[valid_screen]  # [valid_B, LATENT_DIM, LOW_RANK]
    
    # Raw reconstruction logits
    occupy_logits = model_output['occupy_logits'][valid_screen]  # [B, 1, 21, 79]
    ego_char_logits = model_output['ego_char_logits'][valid_screen]  # [B, CHAR_DIM, k, k]
    ego_color_logits = model_output['ego_color_logits'][valid_screen]  # [B, COLOR_DIM, k, k]
    ego_class_logits = model_output['ego_class_logits'][valid_screen]  # [B, CLASS_DIM, k, k]
    hero_presence_logits = model_output['hero_presence_logits'][valid_screen]  # [B]
    hero_centroid = model_output['hero_centroid'][valid_screen]  # [B, 2]
    bag_logits = model_output['bag_logits'][valid_screen]  # [B, GLYPH_DIM]
    
    # message logits
    msg_logits = model_output['msg_logits'][valid_screen]  # [B, 256, MSG_VSIZE]
    
    # action related outputs
    passability_logits = model_output['passability_logits'][valid_screen]  # [B, PASSABILITY_DIRS]
    safety_logits = model_output['safety_logits'][valid_screen]  # [B, PASSABILITY_DIRS]
    skill_logits = model_output.get('skill_logits', None)  # [B, SKILL_NUM] if applicable
    if skill_logits is not None:
        skill_logits = skill_logits[valid_screen]  # [B, SKILL_NUM]
    
    # latent forward model
    latent_pred = model_output.get('latent_pred', None)
    valid_screen_reshape = valid_screen.view(B, T)  # [B, T]
    valid_screen_for_dynamics = valid_screen_reshape[:, :-1].contiguous()  # [B, T-1]
    valid_screen_for_dynamics = valid_screen_for_dynamics.view(-1)
    reward = model_output['reward'][valid_screen_for_dynamics].squeeze(1)  # [B]
    done_logits = model_output['done_logits'][valid_screen_for_dynamics].squeeze(1)  # [B]
    value_k = model_output['value_k'][valid_screen_for_dynamics]  # [B, NUM_HORIZONS]
    if latent_pred is not None:
        latent_pred = latent_pred[valid_screen_for_dynamics]
    
    # inverse dynamics logits
    inverse_dynamics_logits = model_output.get('inverse_dynamics_logits', None)
    if inverse_dynamics_logits is not None:
        inverse_dynamics_logits = inverse_dynamics_logits[valid_screen_for_dynamics]  # [B*(T-1), A + skill_num]
    
    valid_B = valid_screen.sum().item()  # Number of valid samples
    assert valid_B > 0, "No valid samples for loss calculation. Check valid_screen mask."
    
    # ============= Raw Reconstruction Losses =============
    raw_losses = {}
    metrics = {}
    
    # Glyph reconstruction (chars + colors)
    H, W = glyph_chars.shape[1], glyph_chars.shape[2]
    gc = glyph_chars[valid_screen]
    col_t = glyph_colors[valid_screen]
    fg = (gc != ord(' '))                 # [valid_B,H,W]
    occ_t = fg.float().unsqueeze(1)           # [valid_B,1,H,W]
    char_t = torch.clamp(gc - 32, 0, CHAR_DIM - 1)  # [valid_B,H,W]
    bag_t = make_pair_bag(glyph_chars[valid_screen], glyph_colors[valid_screen])  # [valid_B,GLYPH_DIM]
    hero_p_t, hero_c_t = hero_presence_and_centroid(glyph_chars[valid_screen], blstats[valid_screen]) # [valid_B], [valid_B,2]
    
    # ---- hero presence + centroid ----
    hero_bce = F.binary_cross_entropy_with_logits(hero_presence_logits, hero_p_t, pos_weight=torch.full_like(hero_presence_logits, 10), reduction='none')  # [valid_B]
    hero_mse = ((hero_centroid - hero_c_t) ** 2).sum(dim=1) * hero_p_t # [valid_B,2] -> [valid_B]
    hero_loss = hero_bce + 5.0 * hero_mse
    raw_losses['hero_loc'] = hero_loss.mean()  # Average over valid samples

    # occupancy focal BCE (mean over pixels)
    #occ_loss_per_sample = _bce_focal_with_logits(occupy_logits.squeeze(1), occ_t.squeeze(1), alpha_pos=focal_loss_alpha, gamma=focal_loss_gamma, reduction='none').sum(dim=[1, 2])  # [valid_B]
    with torch.no_grad():
        pos = occ_t.sum()
        neg = occ_t.numel() - pos
        pos_weight = (neg / pos).clamp(max=10.0)
    occ_loss_per_sample = F.binary_cross_entropy_with_logits(occupy_logits, occ_t, reduction='none', pos_weight=pos_weight).sum(dim=[1,2,3])  # [valid_B]
    assert occ_loss_per_sample.shape == (valid_B,), f"occupy loss shape mismatch: {occ_loss_per_sample.shape} != ({valid_B},)"
    raw_losses['occupy'] = occ_loss_per_sample.mean()  # Average over valid samples
    metrics['metrics/occupy_jaccard'] = jaccard_index_from_logits(occupy_logits, occ_t, threshold=0.5)
    
    # ---- bag presence BCE (weighted) ----
    # weights: de-emphasize very common chars
    with torch.no_grad():
        w = torch.ones(valid_B, GLYPH_DIM, device=bag_t.device)
        # downweight common chars (all colors)
        common_idx = []
        for ch in COMMON_GLYPHS:
            ci = (ch[0] - 32)
            if 0 <= ci < CHAR_DIM:
                base = ci * COLOR_DIM + ch[1]
                common_idx.append(base)
        if common_idx:
            w[:, common_idx] = 0.1

        w[:, 0] = 0.0  # ignore space glyph

    bag_bce = F.binary_cross_entropy_with_logits(bag_logits, bag_t, weight=w, reduction='none').sum(dim=1).mean()
    
    raw_losses['bag'] = bag_bce
    metrics['metrics/bag_jaccard'] = jaccard_index_from_logits(bag_logits.unsqueeze(1), bag_t.unsqueeze(1), threshold=0.5)
    
    sp = model_output['stats_pred']
    # Filter stats_pred by valid_screen to match target dimensions
    sp_filtered = {}
    for key, value in sp.items():
        if isinstance(value, torch.Tensor):
            sp_filtered[key] = value[valid_screen]
        else:
            sp_filtered[key] = value
    
    var = sp_filtered['logvar'].exp().clamp_min(1e-6)
    nll_cont = 0.5 * (((t['cont'] - sp_filtered['mu'])**2 / var) + sp_filtered['logvar'] + math.log(2*math.pi))
    nll_cont = nll_cont.sum(dim=1).mean()
    hung_loss = F.cross_entropy(sp_filtered['hunger'], t['hunger'], reduction='mean')
    dung_loss = F.cross_entropy(sp_filtered['dungeon'], t['dungeon'], reduction='mean')
    level_loss = F.cross_entropy(sp_filtered['level'], t['level'], reduction='mean')
    cond_loss = F.binary_cross_entropy_with_logits(sp_filtered['cond'], t['cond'], reduction='none').sum(dim=1).mean()
    stats_loss = nll_cont + hung_loss + dung_loss + level_loss + cond_loss
    raw_losses['stats'] = stats_loss

    # message CE (mask PAD + after EOS)
    msg_t = msg_tokens[valid_screen]
    msg_loss = _message_ce_loss(msg_logits, msg_t, pad_id=MSG_PAD, eos_id=MSG_EOS)
    raw_losses['msg'] = msg_loss
    
    # --- Core heads ---
    # Compute ego targets on-the-fly from game map data
    valid_glyph_chars = glyph_chars[valid_screen]   # [valid_B, H, W]
    valid_glyph_colors = glyph_colors[valid_screen] # [valid_B, H, W]
    valid_blstats = blstats[valid_screen]           # [valid_B, BLSTATS_DIM]
    
    # Compute ego view data on-the-fly for each valid sample
    valid_B = valid_screen.sum().item()
    ego_window = config.ego_window  # Get ego window size from config
    ego_char_targets = torch.zeros((valid_B, ego_window, ego_window), dtype=torch.long, device=glyph_chars.device)  # Start with 0s
    ego_color_targets = torch.zeros((valid_B, ego_window, ego_window), dtype=torch.long, device=glyph_chars.device)
    ego_class_targets = torch.zeros((valid_B, ego_window, ego_window), dtype=torch.long, device=glyph_chars.device)
    
    for i in range(valid_B):
        # Get hero position (centroids are in [x, y] format)
        hero_x, hero_y = int(valid_blstats[i, 0].item()), int(valid_blstats[i, 1].item())
        
        # Crop ego view
        ego_chars, ego_colors = crop_ego(valid_glyph_chars[i], valid_glyph_colors[i], hero_y, hero_x, ego_window)
        ego_class = categorize_glyph_tensor(ego_chars, ego_colors)
        
        # Convert ASCII characters to model space (32-127 -> 0-95)
        ego_char_targets[i] = torch.clamp(ego_chars - 32, 0, CHAR_DIM - 1)
        ego_color_targets[i] = ego_colors  
        ego_class_targets[i] = ego_class
    
    # Compute ego losses using on-the-fly targets
    ego_char_weight = torch.ones(CHAR_DIM, device=ego_char_targets.device)
    for ch in COMMON_CHARS:
        ci = ch - 32
        if 0 <= ci < CHAR_DIM:
            ego_char_weight[ci] = 0.1
    ego_char_weight[0] = 0.01  # downweight space char even more
    raw_losses['ego_char'] = F.cross_entropy(ego_char_logits, ego_char_targets, weight=ego_char_weight, reduction='none').sum(dim=[1,2]).mean()
    ego_color_weight = torch.ones(COLOR_DIM, device=ego_color_targets.device)
    ego_color_weight[7] = 0.5  # downweight common color (gray)
    raw_losses['ego_color'] = F.cross_entropy(ego_color_logits, ego_color_targets, weight=ego_color_weight, reduction='none').sum(dim=[1,2]).mean()
    ego_class_weight = torch.ones(NetHackCategory.get_category_count(), device=ego_class_targets.device)
    ego_class_weight[:4] = 0.1  # downweight very common classes (space, wall, door, floor)
    raw_losses['ego_class'] = F.cross_entropy(ego_class_logits, ego_class_targets, weight=ego_class_weight, reduction='none').sum(dim=[1,2]).mean()

    ego_char_metrics = ego_metrics_from_logits(ego_char_logits, ego_char_targets, topk=(1,3))
    ego_color_metrics = ego_metrics_from_logits(ego_color_logits, ego_color_targets, topk=(1,3))
    ego_class_metrics = ego_metrics_from_logits(ego_class_logits, ego_class_targets, topk=(1,3))
    
    metrics.update({f'metrics/ego_char/{k}': v for k, v in ego_char_metrics.items()})
    metrics.update({f'metrics/ego_color/{k}': v for k, v in ego_color_metrics.items()})
    metrics.update({f'metrics/ego_class/{k}': v for k, v in ego_class_metrics.items()})

    if 'passability_target' in batch:
        loss = F.binary_cross_entropy_with_logits(passability_logits, batch['passability_target'][valid_screen], reduction='none')  # [B, PASSABILITY_DIRS]
        if 'hard_mask' in batch:
            m = batch['hard_mask'][valid_screen].float()  # [B, PASSABILITY_DIRS]
            loss = loss * m
        else:
            m = None
        if 'weight' in batch:
            w = batch['weight'][valid_screen].float()  # [B, PASSABILITY_DIRS]
            loss = loss * w
        else:
            w = None
        denom = (m if m is not None else torch.ones_like(loss))
        denom = denom * (w if w is not None else 1.0)
        denom = denom.sum(dim=1).clamp_min(1.0)  # [B]
        raw_losses['passability'] = (loss.sum(dim=1) / denom).mean()  # Mean over valid samples
        pass_metrics = binary_accuracy_from_logits(passability_logits, batch['passability_target'][valid_screen], m, w)
        metrics.update({f'metrics/passability/{k}': v for k, v in pass_metrics.items()})

    if 'rewards_for_dynamics' in batch:
        rewards_for_dynamics = batch['rewards_for_dynamics'][valid_screen_for_dynamics]  # [B*(T-1)]
        # Higher weight for non-zero rewards (20x weight)
        weights = torch.where(rewards_for_dynamics != 0, 20.0, 1.0)
        weighted_loss = ((reward - rewards_for_dynamics)**2 * weights).sum()
        total_weight = weights.sum()
        raw_losses['reward'] = weighted_loss / total_weight.clamp_min(1.0)

    if 'dones_for_dynamics' in batch:
        done_target = batch['dones_for_dynamics'][valid_screen_for_dynamics].float()  # Convert bool to float
        raw_losses['done'] = F.binary_cross_entropy_with_logits(done_logits, done_target, reduction='mean')

    if 'value_k_target' in batch:
        tgt = batch['value_k_target'][valid_screen_for_dynamics]
        mask = batch['value_k_mask'][valid_screen_for_dynamics]
        raw_losses['value_k'] = ((value_k - tgt)**2 * mask).sum() / mask.sum().clamp_min(1.0)  # Mean over valid samples

    # --- Safety risk (8 dirs) ---
    if 'safety_target' in batch:
        loss = F.binary_cross_entropy_with_logits(safety_logits, batch['safety_target'][valid_screen], reduction='none')  # [B, PASSABILITY_DIRS]
        if 'hard_mask' in batch:
            m = batch['hard_mask'][valid_screen].float()  # [B, PASSABILITY_DIRS]
            loss = loss * m
        else:
            m = None
        if 'weight' in batch:
            w = batch['weight'][valid_screen].float()  # [B, PASSABILITY_DIRS]
            loss = loss * w
        else:
            w = None
        denom = (m if m is not None else torch.ones_like(loss))
        denom = denom * (w if w is not None else 1.0)
        denom = denom.sum(dim=1).clamp_min(1.0)  # [B]
        raw_losses['safety'] = (loss.sum(dim=1) / denom).mean()  # Mean over valid samples
        safety_metrics = binary_accuracy_from_logits(safety_logits, batch['safety_target'][valid_screen], m, w)
        metrics.update({f'metrics/safety/{k}': v for k, v in safety_metrics.items()})

    # --- Goal direction ---
    if ('goal_target' in batch) and ('goal_pred' in model_output):
        goal_pred_filtered = model_output['goal_pred'][valid_screen]
        g_tgt  = batch['goal_target'][valid_screen]
        g_m    = batch['goal_mask'][valid_screen].unsqueeze(1)  # [B,1]
        num    = g_m.sum().clamp_min(1.0)
        if goal_pred_filtered is not None:
            raw_losses['goal'] = ((goal_pred_filtered - g_tgt)**2 * g_m).sum() / num
            goal_metrics_dict = goal_metrics(goal_pred_filtered, batch['goal_target'][valid_screen], mask=batch.get('goal_mask')[valid_screen], map_W=79, map_H=21)
            metrics.update({f'metrics/goal/{k}': v for k, v in goal_metrics_dict.items()})

    # --- Dynamics ---
    has_next_mask = batch['has_next_for_dynamics'][valid_screen_for_dynamics]  # [B*(T-1)]
    # Forward dynamics: predict z_next from z_current and action
    if (latent_pred is not None) and ('z_next_detached' in batch):
        z_next_tgt = batch['z_next_detached'][valid_screen_for_dynamics][has_next_mask]
        latent_pred_filtered = latent_pred[has_next_mask]
        
        mse = F.mse_loss(latent_pred_filtered, z_next_tgt, reduction='none').sum(dim=1).mean()  # Mean over valid samples
        cos = 1.0 - F.cosine_similarity(latent_pred_filtered, z_next_tgt, dim=1).mean()
        nce = _info_nce(latent_pred_filtered, z_next_tgt, temperature=config.nce_temperature)
        raw_losses['forward'] = (0.5 * (mse + cos)) + config.nce_weight * nce
        forward_metrics = dynamics_metrics(latent_pred_filtered, z_next_tgt)
        metrics.update({f'metrics/forward/{k}': v for k, v in forward_metrics.items()})
        metrics['metrics/forward/nce'] = float(nce.item())

    # Inverse dynamics: predict action from z_current and z_next
    if inverse_dynamics_logits is not None:
        # Apply has_next mask if available for proper filtering
        inverse_logits_filtered = inverse_dynamics_logits[has_next_mask]
        action_tgt = batch['action_onehot_for_dynamics'][valid_screen_for_dynamics][has_next_mask]
        ce = F.cross_entropy(inverse_logits_filtered, action_tgt.argmax(dim=1).long(), reduction='mean')
        raw_losses['inverse'] = ce
        inv_metrics = ego_metrics_from_logits(inverse_logits_filtered.view(*inverse_logits_filtered.shape, 1, 1), action_tgt.argmax(dim=1).long().view(*action_tgt.argmax(dim=1).shape, 1, 1), topk=(1,3))
        metrics.update({f'metrics/inverse/{k}': v for k, v in inv_metrics.items()})

    # Sum raw reconstruction losses
    total_raw_loss = sum(raw_losses[k] * config.raw_modality_weights.get(k, 1.0) for k in raw_losses)

    # KL divergence
    # Sigma_q = lowrank_factors @ lowrank_factors.T + torch.diag(torch.exp(logvar))
    # KL divergence for low-rank approximation
    eps = 1e-6
    logvar = logvar.clamp(min=-10.0, max=10.0)  # Prevent extreme values
    var = torch.exp(logvar).clamp_min(eps)  # [valid_B, LATENT_DIM]
    mu2 = mu.square().sum(dim=1)  # mu^T * mu # [valid_B,]
    d = mu.size(1)
    if lowrank_factors is not None:
        r = lowrank_factors.size(2)
        U = lowrank_factors.to(torch.float64)  # [valid_B, LATENT_DIM, LOW_RANK]
        UUT_bar = torch.zeros(d, d, device=mu.device, dtype=torch.float64)  # [LATENT_DIM, LATENT_DIM]
        UUT_bar += torch.einsum('bik,bjk->ij', U, U) / valid_B # [LATENT_DIM, LATENT_DIM]
        diag_E_Sigma = var.mean(dim=0, dtype=torch.float64) + (U * U).mean(dim=(0,2))  # E[diag(Sigma_q)], [LATENT_DIM]
        Sigma_q_bar = UUT_bar + torch.diag(diag_E_Sigma) - torch.diag((U * U).mean(dim=(0,2)))  # [LATENT_DIM, LATENT_DIM]
        Sigma_q_bar.fill_diagonal_(0.0)
        Sigma_q_bar = Sigma_q_bar + torch.diag(diag_E_Sigma)  # [LATENT_DIM, LATENT_DIM]
        tr_Sigma_q = var.sum(dim=1) + (U * U).sum(dim=(1,2))  # Trace of Sigma_q
        Dinvu = U / var.unsqueeze(-1)  # [valid_B, LATENT_DIM, LOW_RANK]
        I_plus = torch.bmm(U.transpose(1, 2), Dinvu)  # [valid_B, LOW_RANK, LOW_RANK]
        eye_r = torch.eye(r, device=mu.device, dtype=torch.float64).unsqueeze(0)  # [1, LOW_RANK, LOW_RANK]
        I_plus = I_plus + eye_r  # [valid_B, LOW_RANK, LOW_RANK]
        sign2, logdet_small = torch.linalg.slogdet(I_plus)  # log(det(I + U^T * diag(1/var) * U))
        if not torch.all(sign2 > 0):
            raise ValueError("Matrix I + U^T * diag(1/var) * U is not positive definite.")
        log_det_Sigma_q = logdet_small + logvar.sum(dim=1).to(torch.float64)  # log(det(Sigma_q)) [valid_B,]
        log_det_Sigma_q = log_det_Sigma_q.to(mu.dtype)
    else:
        # If no low-rank factors, just use diagonal covariance
        Sigma_q_bar = torch.diag(var.mean(dim=0, dtype=torch.float64))  # [LATENT_DIM, LATENT_DIM]
        tr_Sigma_q = var.sum(dim=1)  # Trace of Sigma_q
        log_det_Sigma_q = logvar.sum(dim=1)  # log(det(Sigma_q)) [valid_B,]
    # KL divergence: D_KL(q(z|x) || p(z)) =
    # 0.5 * (tr(Sigma_q) + mu^T * mu - d - log(det(Sigma_q)))
    # where d is the dimensionality of the latent space (LATENT_DIM)
    
    kl_per_sample = 0.5 * (tr_Sigma_q + mu2 - d - log_det_Sigma_q)  # [valid_B,]
    kl_loss = kl_per_sample.mean()  # Average over batch

    # Cov(mu) = E[mu * mu^T] - E[mu] * E[mu]^T
    mu64 = mu.to(torch.float64)
    mu_bar = mu64.mean(dim=0)  # Average mean vector over batch
    mu_c = mu64 - mu_bar
    S_mu = (mu_c.t() @ mu_c) / valid_B  # Covariance matrix of the posterior distribution
    assert S_mu.shape == (d, d), f"Covariance matrix of mu shape mismatch: {S_mu.shape} != {(d, d)}"
    total_S = Sigma_q_bar + S_mu  # Total covariance matrix of the posterior distribution
    # Ensure the covariance matrix is symmetric
    total_S = (total_S + total_S.T) / 2 # [LATENT_DIM, LATENT_DIM]
    total_S = total_S + eps * torch.eye(d, device=mu.device, dtype=torch.float64)  # Numerical stability
    
    # Dimension-wise KL divergence with free bits regularization
    diag_S_bar = torch.diag(total_S).clamp_min(eps)  # Diagonal of the covariance matrix, [LATENT_DIM]
    dwkl_per_dim = 0.5 * (diag_S_bar + mu_bar.pow(2) - 1.0 - diag_S_bar.log())  # [LATENT_DIM]
    dwkl_fb_per_dim = torch.clamp(dwkl_per_dim - config.free_bits, min=0)  # Free bits regularization
    dwkl = dwkl_per_dim.sum().to(mu.dtype)  # Sum over dimensions
    dwkl_fb = dwkl_fb_per_dim.sum().to(mu.dtype)  # Free bits regularization sum
    
    # total correlation term - use FP32 for linear algebra operations
    with torch.amp.autocast('cuda', enabled=False):  # Disable autocast for numerical stability
        # Convert to FP32 for linear algebra operations
        D_sqrt_inv = torch.diag(1.0 / torch.sqrt(diag_S_bar)).to(torch.float64) 
        R = D_sqrt_inv @ total_S @ D_sqrt_inv  # R = inv_sqrt * total_S * inv_sqrt
        R = 0.5 * (R + R.t()) + eps * torch.eye(d, device=mu.device, dtype=torch.float64)  # Ensure symmetry and positive definiteness
        sign, logabsdet_R = torch.linalg.slogdet(R)  # log(det(R))
        if not torch.all(sign > 0):
            raise ValueError("Covariance matrix R is not positive definite.")
        total_correlation = -0.5 * logabsdet_R  # Total correlation term
        
        # Convert back to original dtype if needed
        total_correlation = total_correlation.to(mu.dtype)
    
    mutual_information = kl_loss - total_correlation - dwkl  # Mutual information term
    
    with torch.no_grad():
        # Compute the eigenvalues and eigenvectors - use FP32 for numerical stability
        with torch.amp.autocast('cuda', enabled=False):
            eigenvalues = torch.linalg.eigvalsh(total_S.float())
        kl_diagnosis = {
            'mutual_info': float(mutual_information.item()),
            'total_correlation': float(total_correlation.item()),
            'dimension_wise_kl': dwkl_per_dim,
            'dimension_wise_kl_sum': float(dwkl.item()),
            'eigenvalues': eigenvalues
        }
    
    # Total weighted loss
    total_loss = (total_raw_loss + 
                  mi_beta * mutual_information + 
                  tc_beta * total_correlation +
                  dw_beta * dwkl_fb)  # Free bits regularization
    
    return {
        'total_loss': total_loss,
        'total_raw_loss': total_raw_loss,
        'kl_loss': kl_loss,
        'raw_losses': raw_losses,
        'kl_diagnosis': kl_diagnosis,
        'metrics': metrics
    }

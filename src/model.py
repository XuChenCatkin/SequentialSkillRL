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
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Optional

# ------------------------- hyper‑params ------------------------------ #
CHAR_DIM = 96      # ASCII code space for characters shown on the map (32-127)
CHAR_EMB = 16      # char‑id embedding dim (0-15)
COLOR_DIM = 16      # colour id space for colours
COLOR_EMB = 4       # colour‑id embedding dim
GLYPH_EMB = CHAR_EMB + COLOR_EMB  # glyph embedding dim
LATENT_DIM = 96     # z‑dim for VAE
BLSTATS_DIM = 27   # raw scalar stats (hp, gold, …)
MSG_PAD = 0  
MSG_VOCAB = 128     # Nethack takes 32-127 byte for char of messages
MSG_SOS = MSG_VOCAB  # start-of-sequence token id
MSG_EOS = MSG_VOCAB + 1  # end-of-sequence token id
MSG_VSIZE = MSG_VOCAB + 2  # vocab size including SOS/EOS
MSG_MAXLEN = 256  # max length of message text (padded/truncated)
GLYPH_DIM = CHAR_DIM * COLOR_DIM  # glyph dim for char + color
PADDING_IDX = [32, 0]  # padding index for glyphs (blank space char, black color)
PADDING_CHAR = 32      # padding character (blank space)
PADDING_COLOR = 0       # padding color (black)
LOW_RANK = 0      # low-rank factorisation rank for covariance
# NetHack map dimensions (from NetHack source: include/config.h)
MAP_HEIGHT = 21     # ROWNO - number of rows in the map
MAP_WIDTH = 79      # COLNO-1 - playable columns (COLNO=80, but rightmost is UI)
MAX_X_COORD = 78    # Maximum x coordinate (width - 1)
MAX_Y_COORD = 20    # Maximum y coordinate (height - 1)
ROLE_CAD = 13     # role cardinality for MiniHack (e.g. 4 for 'knight')
RACE_CAD = 5      # race cardinality for MiniHack (e.g. 0 for 'human')
GEND_CAD = 3      # gender cardinality for MiniHack (e.g. 'male', 'female', 'neuter')
ALIGN_CAD = 3     # alignment cardinality for MiniHack (e.g. 1 for 'lawful')
ROLE_EMB = 8      # role embedding dim
RACE_EMB = 4      # race embedding dim
GEND_EMB = 2      # gender embedding dim
ALIGN_EMB = 2     # alignment embedding dim

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

COMMON_CHARS = [ord('#'), ord('.'), ord('-'), ord('|')]
COMMON_GLYPHS = [(a, 7) for a in COMMON_CHARS] # (char, color) pairs for common glyphs
HERO_CHAR = ord('@')  # hero character code (ASCII 64)
BAG_Z = 24         # slice of z reserved for glyph-bag / anchors
BAG_MLP = 32       # feature size from glyph-bag encoder


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
    pad_idx = _pair_index(torch.full_like(glyph_chars, PADDING_CHAR), torch.zeros_like(glyph_colors))
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

def hero_presence_and_centroid(glyph_chars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Presence p in {0,1} and centroid (y,x) in [-1,1]^2 for '@'.
    If absent, centroid is (0,0) and p=0 (loss masked by p).
    """
    B, H, W = glyph_chars.shape
    is_hero = (glyph_chars == HERO_CHAR).float()                  # [B,H,W]
    present = (is_hero.sum(dim=(1,2)) > 0).float()               # [B]
    # coords in [-1,1]
    yy = torch.linspace(-1, 1, H, device=glyph_chars.device).view(1, H, 1).expand(B, H, W)
    xx = torch.linspace(-1, 1, W, device=glyph_chars.device).view(1, 1, W).expand(B, H, W)
    area = is_hero.sum(dim=(1,2)).clamp_min(1.0)                 # avoid /0
    cy = (is_hero * yy).sum(dim=(1,2)) / area
    cx = (is_hero * xx).sum(dim=(1,2)) / area
    centroid = torch.stack([cy, cx], dim=1)                      # [B,2]
    # for absent cases, centroids won’t be used (masked by present)
    return present, centroid


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
    def __init__(self, dropout=0.0, attn_queries=2, feat_channels=96):
        super().__init__()
        self.char_emb = nn.Embedding(CHAR_DIM + 1, CHAR_EMB, padding_idx=0)
        self.col_emb  = nn.Embedding(COLOR_DIM + 1, COLOR_EMB, padding_idx=0)
        in_ch = CHAR_EMB + COLOR_EMB + 1  # + foreground mask
        in_ch += 2 # +2 for CoordConv (x,y in [-1,1])

        # Stem
        self.stem = ConvBlock(in_ch, 64, k=3, dilation=1, drop=dropout)

        # Body: mix normal and width-focused dilations (map is 21x79)
        self.body = nn.Sequential(
            ConvBlock(64, 64, k=3, dilation=1,        drop=dropout),      # local
            ConvBlock(64, 64, k=3, dilation=(1, 3),   drop=dropout),      # widen RF (width)
            ConvBlock(64, 64, k=3, dilation=1,        drop=dropout),
            ConvBlock(64, 96, k=3, dilation=(1, 9),   drop=dropout),
            ConvBlock(96, feat_channels, k=3, dilation=1, drop=dropout),
        )

        # Pooling
        self.pool = AttnPool(C=feat_channels, K=attn_queries)
        pooled_dim = feat_channels * attn_queries

        # Tiny head you can keep or replace downstream
        self.out_dim = pooled_dim
        self.norm_out = nn.LayerNorm(self.out_dim)

        
    @staticmethod
    def _coords(B, H, W, device, dtype):
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return yy, xx

    def forward(self, glyph_chars: torch.IntTensor, glyph_colors: torch.IntTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # glyph_chars in ASCII [32..127]; shift to [0..95]; space -> 0 (padding_idx)
        gc = torch.clamp(glyph_chars - 32, 0, CHAR_DIM - 1)
        B, H, W = gc.shape
        device = gc.device
        fg = (gc != 0).unsqueeze(1).float()  # [B,1,H,W]
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
    def __init__(self, z_bag_dim: int, z_core_dim:int, char_dim: int, color_dim: int,
                 H: int = MAP_HEIGHT, W: int = MAP_WIDTH,
                 base_ch: int = 256, mid_ch: int = 128, drop: float = 0.1,
                 occ_prior: float = 0.1, rare_cond_prior: float = 0.05):
        super().__init__()
        self.H, self.W = H, W
        self.base_ch = base_ch
        self.z_bag_dim = z_bag_dim
        self.z_core_dim = z_core_dim
        # coords
        yy = torch.linspace(-1, 1, H).view(1,1,H,1).expand(1,1,H,W).clone()
        xx = torch.linspace(-1, 1, W).view(1,1,1,W).expand(1,1,H,W).clone()
        self.register_buffer("coord_y", yy)
        self.register_buffer("coord_x", xx)
        # project z to a broadcast feature map
        self.z_core_to_map = nn.Sequential(nn.Linear(z_core_dim, base_ch), nn.ReLU(inplace=True), nn.Linear(base_ch, base_ch))
        # stem
        self.core_stem = nn.Sequential(
            nn.Conv2d(base_ch + 2, mid_ch, 3, padding=1),
            nn.GroupNorm(_gn_groups(mid_ch), mid_ch), nn.ReLU(inplace=True),
        )
        # FiLM-conditioned residual body (anisotropic to cover width)
        self.core_block1 = ResBlockFiLM(mid_ch, mid_ch, (1,9),  drop, z_dim=z_core_dim)
        self.core_block2 = ResBlockFiLM(mid_ch, mid_ch, (1,27), drop, z_dim=z_core_dim)
        # local refinement
        self.core_refine = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.GroupNorm(_gn_groups(mid_ch), mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.GroupNorm(_gn_groups(mid_ch), mid_ch), nn.ReLU(inplace=True),
        )
        # heads
        self.head_occ = nn.Sequential(nn.Conv2d(mid_ch, mid_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(mid_ch//2, 1, 1))
        self.head_chr = nn.Sequential(nn.Conv2d(mid_ch, mid_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(mid_ch//2, char_dim, 1))
        self.head_col = nn.Sequential(nn.Conv2d(mid_ch, mid_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(mid_ch//2, color_dim, 1))
        
        self.z_bag_to_map = nn.Sequential(nn.Linear(z_bag_dim, base_ch), nn.ReLU(inplace=True), nn.Linear(base_ch, base_ch))
        # stem
        self.bag_stem = nn.Sequential(
            nn.Conv2d(base_ch + 2, mid_ch, 3, padding=1),
            nn.GroupNorm(_gn_groups(mid_ch), mid_ch), nn.ReLU(inplace=True),
        )
        # FiLM-conditioned residual body (anisotropic to cover width)
        self.bag_block1 = ResBlockFiLM(mid_ch, mid_ch, (1,9),  drop, z_dim=z_bag_dim)
        self.bag_block2 = ResBlockFiLM(mid_ch, mid_ch, (1,27), drop, z_dim=z_bag_dim)
        # local refinement
        self.bag_refine = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.GroupNorm(_gn_groups(mid_ch), mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.GroupNorm(_gn_groups(mid_ch), mid_ch), nn.ReLU(inplace=True),
        )
        self.head_rare_occ = nn.Sequential(nn.Conv2d(mid_ch, mid_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(mid_ch//2, 1, 1))
        self.head_rare_chr = nn.Sequential(nn.Conv2d(mid_ch, mid_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(mid_ch//2, char_dim, 1))
        self.head_rare_col = nn.Sequential(nn.Conv2d(mid_ch, mid_ch//2, 1), nn.ReLU(inplace=True), nn.Conv2d(mid_ch//2, color_dim, 1))
        
        with torch.no_grad():
            b = math.log(max(occ_prior, 1e-6) / max(1 - occ_prior, 1e-6))
            self.head_occ[-1].bias.fill_(b)
            rb = math.log(max(rare_cond_prior, 1e-6) / max(1 - rare_cond_prior, 1e-6))
            self.head_rare_occ[-1].bias.fill_(rb)

    def _prior_to_bias(self, prior: torch.Tensor, strength: float) -> torch.Tensor:
        """
        prior: probabilities in [0,1], shape [B,C].
        Returns additive bias (logits) [B,C] scaled by bag_bias_strength.
        """
        p = prior.clamp(1e-6, 1 - 1e-6)
        return strength * torch.log(p / (1 - p))


    def forward(self, z: torch.Tensor,
                char_bag_prior: Optional[torch.Tensor] = None,   # [B,CHAR_DIM] optional
                color_bag_prior: Optional[torch.Tensor] = None,   # [B,COLOR_DIM] optional
                bag_bias_strength: float = 1.0,  # strength of prior bias
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = z.size(0)
        z_bag  = z[:, :self.z_bag_dim]
        z_core = z[:, self.z_bag_dim:]

        z_core_map = self.z_core_to_map(z_core).view(B, self.base_ch, 1, 1).expand(B, self.base_ch, self.H, self.W)
        x_core = torch.cat([z_core_map, self.coord_y.expand(B,-1,-1,-1), self.coord_x.expand(B,-1,-1,-1)], dim=1)
        x_core = self.core_stem(x_core)
        x_core = self.core_block1(x_core, z_core)
        x_core = self.core_block2(x_core, z_core)
        x_core = self.core_refine(x_core)
        occ = self.head_occ(x_core)
        g_log  = self.head_chr(x_core)
        c_log  = self.head_col(x_core)
        
        z_bag_map = self.z_bag_to_map(z_bag).view(B, self.base_ch, 1, 1).expand(B, self.base_ch, self.H, self.W)
        x_bag = torch.cat([z_bag_map, self.coord_y.expand(B,-1,-1,-1), self.coord_x.expand(B,-1,-1,-1)], dim=1)
        x_bag = self.bag_stem(x_bag)
        x_bag = self.bag_block1(x_bag, z_bag)
        x_bag = self.bag_block2(x_bag, z_bag)
        x_bag = self.bag_refine(x_bag)
        rare_occ = self.head_rare_occ(x_bag)
        g_rare_log  = self.head_rare_chr(x_bag)
        c_rare_log  = self.head_rare_col(x_bag)
        
        if char_bag_prior is not None:
            cb = self._prior_to_bias(char_bag_prior, bag_bias_strength).view(B, CHAR_DIM, 1, 1).expand_as(g_log)
            g_rare_log = g_rare_log + cb
        if color_bag_prior is not None:
            kb = self._prior_to_bias(color_bag_prior, bag_bias_strength).view(B, COLOR_DIM, 1, 1).expand_as(c_log)
            c_rare_log = c_rare_log + kb
        return occ, g_log, c_log, rare_occ, g_rare_log, c_rare_log

    @staticmethod
    def _filter_topk_topp_channels(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """Filter along channel dim (classes) for [B,C,H,W] logits."""
        B, C, H, W = logits.shape
        out = logits
        if top_k is not None and top_k > 0 and top_k < C:
            kth = torch.topk(out, top_k, dim=1).values[:, -1:, :, :]
            out = out.masked_fill(out < kth, float('-inf'))
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(out, dim=1, descending=True)  # [B,C,H,W]
            probs = F.softmax(sorted_logits, dim=1)
            cum = probs.cumsum(dim=1)
            remove = cum > top_p
            remove[:, 1:, :, :] = remove[:, :-1, :, :].clone()
            remove[:, 0, :, :] = False
            remove = remove | torch.isneginf(sorted_logits)
            to_remove = torch.zeros_like(out, dtype=torch.bool)
            to_remove.scatter_(1, sorted_idx, remove)
            out = out.masked_fill(to_remove, float('-inf'))
        return out

    @staticmethod
    def _sample_per_pixel(logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return class indices [B,H,W] from [B,C,H,W] logits."""
        if deterministic:
            return logits.argmax(dim=1)
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0)
        flat = probs.permute(0,2,3,1).contiguous().view(-1, C)  # [B*H*W, C]
        zero_row = flat.sum(dim=1, keepdim=True) == 0
        if zero_row.any():
            arg = logits.permute(0,2,3,1).contiguous().view(-1, C).argmax(dim=1, keepdim=True)
            one_hot = torch.zeros_like(flat).scatter_(1, arg, 1.0)
            flat = torch.where(zero_row, one_hot, flat)
        idx = torch.multinomial(flat, 1).squeeze(-1)
        return idx.view(B, H, W)

    @staticmethod
    def sample_from_logits(
        occupy_logits: torch.Tensor,
        rare_occupy_logits: torch.Tensor,
        glyph_logits: torch.Tensor,
        color_logits: torch.Tensor,
        rare_glyph_logits: torch.Tensor,
        rare_color_logits: torch.Tensor,
        temperature: float = 1.0,
        occ_thresh: float = 0.5,
        rare_occ_thresh: float = 0.5,
        deterministic: bool = True,
        glyph_top_k: int = 0,
        glyph_top_p: float = 1.0,
        color_top_k: int = 0,
        color_top_p: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Turn map logits into hard predictions.
        Returns (fg_mask [B,H,W] bool, chars [B,H,W] ASCII, colors [B,H,W]).
        """
        assert temperature > 0, "temperature must be > 0"
        device = occupy_logits.device
        B, _, H, W = occupy_logits.shape
        # occupancy -> foreground mask
        occ_p = torch.sigmoid(occupy_logits).squeeze(1)
        rare_occ_p = torch.sigmoid(rare_occupy_logits).squeeze(1)
        if deterministic:
            fg_mask = (occ_p > occ_thresh)
            rare_fg_mask = (rare_occ_p > rare_occ_thresh) & fg_mask
        else:
            fg_mask = torch.bernoulli(occ_p.clamp(1e-6, 1-1e-6)).bool()
            rare_fg_mask = torch.bernoulli(rare_occ_p.clamp(1e-6, 1-1e-6)).bool() & fg_mask
        # scale temperatures for class heads
        g_log = glyph_logits / temperature
        c_log = color_logits / temperature
        g_rare_log = rare_glyph_logits / temperature
        c_rare_log = rare_color_logits / temperature
        # filter
        if not deterministic and (glyph_top_k > 0 or color_top_k > 0 or glyph_top_p < 1.0 or color_top_p < 1.0):
            g_log = MapDecoder._filter_topk_topp_channels(g_log, top_k=glyph_top_k, top_p=glyph_top_p)
            c_log = MapDecoder._filter_topk_topp_channels(c_log, top_k=color_top_k, top_p=color_top_p)
            g_rare_log = MapDecoder._filter_topk_topp_channels(g_rare_log, top_k=glyph_top_k, top_p=glyph_top_p)
            c_rare_log = MapDecoder._filter_topk_topp_channels(c_rare_log, top_k=color_top_k, top_p=color_top_p)
        # choose indices
        g_idx = MapDecoder._sample_per_pixel(g_log, deterministic=deterministic)  # [B,H,W] in 0..CHAR_DIM-1
        c_idx = MapDecoder._sample_per_pixel(c_log, deterministic=deterministic)  # [B,H,W] in 0..COLOR_DIM-1
        rare_g_idx = MapDecoder._sample_per_pixel(g_rare_log, deterministic=deterministic)  # [B,H,W]
        rare_c_idx = MapDecoder._sample_per_pixel(c_rare_log, deterministic=deterministic)
        # write into outputs with background defaults
        chars = torch.full((B, H, W), ord(' '), device=device, dtype=torch.long)  # 32 ' '
        colors = torch.zeros((B, H, W), device=device, dtype=torch.long)             # 0
        chars_fg = (g_idx + 32).clamp(32, 127)
        rare_chars_fg = (rare_g_idx + 32).clamp(32, 127)
        chars[fg_mask] = chars_fg[fg_mask]
        colors[fg_mask] = c_idx[fg_mask]
        chars[rare_fg_mask] = rare_chars_fg[rare_fg_mask]
        colors[rare_fg_mask] = rare_c_idx[rare_fg_mask]
        return fg_mask, rare_fg_mask, chars, colors

    def generate(self, z: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: z -> logits -> hard maps."""
        occ, gl, cl, rare_occ, rare_gl, rare_cl = self.forward(z)
        return self.sample_from_logits(occ, gl, cl, rare_occ, rare_gl, rare_cl, **kwargs)

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
            proc[:, hp_idx] = cont[:, 10] / torch.clamp(cont[:, 11], min=1)
            proc[:, en_idx] = cont[:, 14] / torch.clamp(cont[:, 15], min=1)
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
    def __init__(self, stats: int=BLSTATS_DIM, emb_dim: int=64):
        super().__init__()
        self.preprocessor = BlstatsPreprocessor(stats_dim=stats)
        input_dim = self.preprocessor.get_output_dim()  # 43 dimensions after preprocessing
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        
    def forward(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self.preprocessor(stats)  # [B, 27] -> [B, 43]
        features = self.net(processed)  # [B, 43] -> [B, 16]
        return features
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return 16
    
class StatsDecoder(nn.Module):
    """Predict stats with proper likelihoods: continuous (Gaussian), discrete (CE), conditions (BCE)."""
    def __init__(self, z_dim: int, cont_dim: int = 19):
        super().__init__()
        hid = 256
        self.net = nn.Sequential(nn.Linear(z_dim, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU())
        self.mu      = nn.Linear(hid, cont_dim)
        self.logvar  = nn.Linear(hid, cont_dim)
        self.hunger  = nn.Linear(hid, 7)
        self.dungeon = nn.Linear(hid, 11)
        self.level   = nn.Linear(hid, 51)
        self.cond    = nn.Linear(hid, len(CONDITION_BITS))
    def forward(self, z):
        h = self.net(z)
        return {"mu": self.mu(h), "logvar": self.logvar(h), "hunger": self.hunger(h), "dungeon": self.dungeon(h), "level": self.level(h), "cond": self.cond(h)}

class MessageEncoder(nn.Module):
    def __init__(self, vocab: int=MSG_VSIZE, emb: int=32, hid: int=12):
        super().__init__()
        self.emb_dim = emb
        self.hid_dim = hid
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, emb, padding_idx=MSG_PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)  # bidirectional GRU
        self.out = nn.Linear(hid * 2, hid)  # output layer to reduce to hid_dim
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
    def __init__(self, z_dim: int, L: int = MSG_MAXLEN, ch: int = 128, drop: float = 0.1, vocab: int = MSG_VSIZE):
        super().__init__()
        self.L = L; self.vocab = vocab
        x = torch.linspace(-1, 1, L).view(1,1,L)
        self.register_buffer("pos", x)
        self.z_to_line = nn.Sequential(nn.Linear(z_dim, ch), nn.SiLU(), nn.Linear(ch, ch))
        self.stem = nn.Sequential(nn.Conv1d(ch+1, ch, 3, padding=1), nn.GroupNorm(_gn_groups(ch), ch), nn.SiLU())
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, dilation=1),  nn.GroupNorm(_gn_groups(ch), ch), nn.SiLU(),
            nn.Conv1d(ch, ch, 3, padding=2, dilation=2),  nn.GroupNorm(_gn_groups(ch), ch), nn.SiLU(),
            nn.Conv1d(ch, ch, 3, padding=4, dilation=4),  nn.GroupNorm(_gn_groups(ch), ch), nn.SiLU(),
            nn.Conv1d(ch, ch, 3, padding=8, dilation=8),  nn.GroupNorm(_gn_groups(ch), ch), nn.SiLU(),
        )
        self.head = nn.Conv1d(ch, vocab, 1)
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
    def __init__(self):
        super().__init__()
        self.role_emb = nn.Embedding(ROLE_CAD, ROLE_EMB)
        self.race_emb = nn.Embedding(RACE_CAD, RACE_EMB)
        self.gend_emb = nn.Embedding(GEND_CAD, GEND_EMB)
        self.align_emb = nn.Embedding(ALIGN_CAD, ALIGN_EMB)
    
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
        return ROLE_EMB + RACE_EMB + GEND_EMB + ALIGN_EMB  # 16

class MultiModalHackVAE(nn.Module):
    def __init__(self, 
                 bInclude_glyph_bag=True, 
                 bInclude_hero=True,
                 dropout_rate=0.0,
                 enable_dropout_on_latent=True,
                 enable_dropout_on_decoder=True,
                 logger: logging.Logger | None=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.dropout_rate = dropout_rate
        self.enable_dropout_on_latent = enable_dropout_on_latent
        self.enable_dropout_on_decoder = enable_dropout_on_decoder
        self.z_bag_dim = BAG_Z
        assert LATENT_DIM > self.z_bag_dim, "LATENT_DIM must be greater than BAG_Z"
        self.num_pairs = GLYPH_DIM
        
        self.map_encoder = MapEncoder(dropout=dropout_rate)
        self.stats_encoder = StatsEncoder()
        self.msg_encoder = MessageEncoder()
        self.hero_emb = HeroEmbedding() if bInclude_hero else None

        self.include_glyph_bag = bInclude_glyph_bag
        self.include_hero = bInclude_hero

        fusion_in = 2 * self.map_encoder.get_output_channels() + \
                    self.stats_encoder.get_output_channels() + \
                    self.msg_encoder.get_output_channels() + \
                    (self.hero_emb.get_output_channels() if bInclude_hero else 0) # cnn + stats + msg + hero embedding
        
        # Add dropout to the encoder fusion layer if enabled
        if self.dropout_rate > 0.0 and self.enable_dropout_on_latent:
            self.to_latent = nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Dropout(self.dropout_rate),
                nn.Linear(fusion_in, 256), nn.ReLU(),
                nn.Dropout(self.dropout_rate),
            )
        else:
            self.to_latent = nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Linear(fusion_in, 256), nn.ReLU(),
            )
        self.latent_dim = LATENT_DIM
        self.lowrank_dim = LOW_RANK
        self.mu_head     = nn.Linear(256, LATENT_DIM)
        self.logvar_diag_head = nn.Linear(256, LATENT_DIM)  # diagonal part
        self.lowrank_factor_head = nn.Linear(256, LATENT_DIM * LOW_RANK) if LOW_RANK else None # low-rank factors

        # We will have 3 categories of decoder heads:
        # 1. For reconstruction of observations:
        #    - glyphs (pixel-wise categorical for both char and color)
        #    - glyphs_bag (pixel-wise categorical) (optional)
        #    - stats (pixel-wise categorical)
        #    - messages (token-wise categorical)
        #    - heros (token-wise categorical) (optional)
        # 2. For reconstruction of frozen embeddings (optional):
        #    - glyph_char_embeddings (pixel-wise continuous)
        #    - glyph_color_embeddings (pixel-wise continuous)
        #    - glyph_bag_embeddings (pixel-wise continuous) (optional)
        #    - stats_embeddings (pixel-wise continuous)
        #    - messages_embeddings (pixel-wise continuous)
        #    - heros_embeddings (pixel-wise continuous) (optional)
        # 3. For dynamic predictions p(x_{t+1} | z_t, a_t, h_t, c)
        
        # This will be shared across all decoders
        # if self.dropout_rate > 0.0 and self.enable_dropout_on_decoder:
        #     self.decode_shared = nn.Sequential(
        #         nn.Linear(LATENT_DIM, 256), nn.ReLU(),
        #         nn.Dropout(self.dropout_rate),
        #     )
        # else:
        #     self.decode_shared = nn.Sequential(
        #         nn.Linear(LATENT_DIM, 256), nn.ReLU(),
        #     )
        core_dim = LATENT_DIM - self.z_bag_dim
        self.map_decoder  = MapDecoder(z_bag_dim=self.z_bag_dim, z_core_dim=core_dim, char_dim=CHAR_DIM, color_dim=COLOR_DIM, H=MAP_HEIGHT, W=MAP_WIDTH, base_ch=96, mid_ch=96, drop=dropout_rate)
        self.stats_decoder = StatsDecoder(z_dim=core_dim, cont_dim=19)
        self.msg_decoder   = MessageDecoder(z_dim=core_dim, L=MSG_MAXLEN, vocab=MSG_VSIZE)
        self.bag_head = nn.Sequential(
            nn.Linear(self.z_bag_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.num_pairs)
        )
        self.hero_presence_head = nn.Linear(self.z_bag_dim, 1)   # logits
        self.hero_centroid_head = nn.Linear(self.z_bag_dim, 2)   # tanh -> [-1,1]
        self.detach_bag_prior = True  # whether to detach the bag prior from the gradients

        # Dynamic prediction heads
        # It takes latent z, action a, HDP HMM state h and hero info c
        # TODO: Implement dynamic prediction heads
        

    def encode(self, glyph_chars, glyph_colors, blstats, msg_tokens, hero_info=None):
        """
        Encodes the input features into a latent space.
        
        Args:
            glyph_chars: [B, 20, 21, 79] - character glyphs
            glyph_colors: [B, 4, 21, 79] - color glyphs
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
        is_hero = (glyph_chars == HERO_CHAR)
        is_common = torch.zeros_like(is_fg, dtype=torch.bool)  # common glyphs mask
        for c in COMMON_GLYPHS:
            is_common |= (glyph_chars == c[0]) & (glyph_colors == c[1])
        is_rare = is_fg & ~is_common & ~is_hero  # rare glyphs mask
        rare_glyph_chars[is_rare] = glyph_chars[is_rare]
        rare_glyph_colors[is_rare] = glyph_colors[is_rare]
        rare_glyph_feat = self.map_encoder(rare_glyph_chars, rare_glyph_colors)  # [B,64]

        stats_feat = self.stats_encoder(blstats)
        
        # Message encoding
        msg_feat = self.msg_encoder(msg_tokens)  # [B,hid_dim], [B,256,emb_dim]

        features = [glyph_feat, rare_glyph_feat, stats_feat, msg_feat]

        if self.include_hero != (hero_info is not None):
            raise ValueError("hero_info must be provided if and only if include_hero is True.")
        
        # Hero embedding
        # check shape of hero_info should be [B, 4] with role, race, gend, align
        if hero_info is not None:
            if hero_info.size(0) != B:
                raise ValueError("hero_info must have the same batch size as glyph_chars and glyph_colors.")
            if hero_info.size(1) != 4:
                raise ValueError("hero_info must have shape [B, 4]")
            role, race, gend, align = hero_info[:, 0], hero_info[:, 1], hero_info[:, 2], hero_info[:, 3]
            hero_feat = self.hero_emb(role, race, gend, align)  # [B,16], dict
            features.append(hero_feat)
        
        fused = torch.cat(features, dim=-1)
        
        # Encode to latent space
        h = self.to_latent(fused)
        mu = self.mu_head(h)
        logvar_diag = self.logvar_diag_head(h)
        lowrank_factors = self.lowrank_factor_head(h).view(B, LATENT_DIM, LOW_RANK) if self.lowrank_factor_head is not None else None
        
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

    def decode(self, z):
        # split z -> [z_bag | z_core]
        z_bag  = z[:, :self.z_bag_dim]
        z_core = z[:, self.z_bag_dim:]

        # --- bag prediction ---
        bag_logits = self.bag_head(z_bag)                       # [B,GLYPH_DIM]
        bag_probs  = torch.sigmoid(bag_logits)
        char_prior, color_prior = bag_marginals(bag_probs)      # [B,CHAR_DIM], [B,COLOR_DIM]

        # optionally detach priors (training schedule)
        if self.detach_bag_prior:
            char_prior = char_prior.detach()
            color_prior = color_prior.detach()
        occupy_logits, char_logits, color_logits, rare_occ_logits, rare_char_logits, rare_color_logits = self.map_decoder(z, char_prior, color_prior, bag_bias_strength=3.0) 
        stats_pred = self.stats_decoder(z_core)
        msg_logits = self.msg_decoder(z_core)
        hero_logit   = self.hero_presence_head(z_bag).squeeze(-1)       # [B]
        hero_centre = torch.tanh(self.hero_centroid_head(z_bag))  # [B,2] -> [-1,1] range
        return {
            "occupy_logits": occupy_logits, 
            "char_logits": char_logits, 
            "color_logits": color_logits, 
            "stats_pred": stats_pred, 
            "msg_logits": msg_logits, 
            "bag_logits" : bag_logits,
            "bag_char_prior": char_prior,
            "bag_color_prior": color_prior,
            "rare_occ_logits": rare_occ_logits,
            "rare_char_logits": rare_char_logits,
            "rare_color_logits": rare_color_logits,
            "hero_presence_logits": hero_logit, 
            "hero_centroid": hero_centre
        }
        
    def set_bag_detach(self, flag: bool = True):
        self.detach_bag_prior = flag

    def forward(self, glyph_chars, glyph_colors, blstats, msg_tokens, hero_info=None, detach_bag=True):
        
        enc = self.encode(glyph_chars, glyph_colors, blstats, msg_tokens, hero_info)
        z = self._reparameterise(enc["mu"], enc["logvar"])  # [B,D]
        self.set_bag_detach(detach_bag)  # set detach flag for bag prior
        dec = self.decode(z)
        return {**enc, **dec}
    
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
        rare_occ_thresh: float = 0.5,
        hero_presence_thresh: float = 0.5,
        map_deterministic: bool = True,
        glyph_top_k: int = 0,
        glyph_top_p: float = 1.0,
        color_top_k: int = 0,
        color_top_p: float = 1.0,
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

        dec = self.decode(z)

        # --- Map sampling ---
        fg_mask, rare_fg_mask, chars, colors = MapDecoder.sample_from_logits(
            dec["occupy_logits"], 
            dec["rare_occ_logits"],
            dec["char_logits"], 
            dec["color_logits"],
            dec['rare_char_logits'],
            dec['rare_color_logits'],
            temperature=map_temperature,
            occ_thresh=map_occ_thresh,
            rare_occ_thresh=rare_occ_thresh,
            deterministic=map_deterministic,
            glyph_top_k=glyph_top_k, glyph_top_p=glyph_top_p,
            color_top_k=color_top_k, color_top_p=color_top_p,
        )
        
        hero_presence_logit = dec["hero_presence_logits"]
        if map_deterministic:
            has_hero = (torch.sigmoid(hero_presence_logit) > hero_presence_thresh)  # [B]
        else:
            has_hero = torch.bernoulli(torch.sigmoid(hero_presence_logit))  # [B]
        hero_centroid = dec["hero_centroid"]
        hero_yy = ((hero_centroid[:, 0] + 1) / 2 * MAP_HEIGHT).round().long()  # [B]
        hero_xx = ((hero_centroid[:, 1] + 1) / 2 * MAP_WIDTH).round().long()
        hero_yy = torch.clamp(hero_yy, 0, MAP_HEIGHT - 1)  # clamp to valid range
        hero_xx = torch.clamp(hero_xx, 0, MAP_WIDTH - 1)
        chars[has_hero, hero_yy[has_hero], hero_xx[has_hero]] = HERO_CHAR
        colors[has_hero, hero_yy[has_hero], hero_xx[has_hero]] = 7

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
            "fg_mask": fg_mask,
            "rare_fg_mask": rare_fg_mask,
            "chars": chars,
            "colors": colors,
            "msg_tokens": msg_tokens_out,
            "stats_point": stats_point,
            "z": z,
        }
        if include_logits:
            out.update({
                "occupy_logits": dec["occupy_logits"],
                "char_logits": dec["char_logits"],
                "color_logits": dec["color_logits"],
                "rare_occupy_logits": dec["rare_occ_logits"],
                "rare_char_logits": dec["rare_char_logits"],
                "rare_color_logits": dec["rare_color_logits"],
                "hero_presence_logits": hero_presence_logit,
                "hero_centroid": hero_centroid,
                "msg_logits": dec["msg_logits"],
            })
        return out
        
    def get_dropout_config(self):
        """
        Get dropout configuration for logging and debugging
        
        Returns:
            Dict containing dropout configuration
        """
        return {
            'dropout_rate': self.dropout_rate,
            'enable_dropout_on_latent': self.enable_dropout_on_latent,
            'enable_dropout_on_decoder': self.enable_dropout_on_decoder
        }

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
    return (ce * keep).sum() / keep.sum().clamp_min(1.0)

def vae_loss(
    model_output, 
    glyph_chars, 
    glyph_colors, 
    blstats, 
    msg_tokens,
    valid_screen,
    raw_modality_weights={
        'occupy': 2.0, 
        'rare_occupy': 2.0, 
        'common_char': 1.0, 
        'common_color': 1.0, 
        'rare_char': 2.0,
        'rare_color': 2.0,
        'stats': 0.5, 
        'msg': 0.5, 
        'bag': 50, 
        'hero_loc': 100
    },
    focal_loss_alpha=0.1,
    focal_loss_gamma=2.0,
    mi_beta=1.0,
    tc_beta=1.0,
    dw_beta=1.0,
    free_bits=0.0):
    """
    VAE loss with separate embedding and raw reconstruction losses with free-bits KL and Gaussian TC proxy.
    """
    # Extract outputs from model
    mu = model_output['mu'][valid_screen]  # [valid_B, LATENT_DIM]
    logvar = model_output['logvar'][valid_screen]
    lowrank_factors = model_output['lowrank_factors']
    if lowrank_factors is not None:
        lowrank_factors = lowrank_factors[valid_screen]  # [valid_B, LATENT_DIM, LOW_RANK]
    
    # Raw reconstruction logits
    occupy_logits = model_output['occupy_logits'][valid_screen]  # [B, 1, 21, 79]
    char_logits = model_output['char_logits'][valid_screen]  # [B, CHAR_DIM, 21, 79]
    color_logits = model_output['color_logits'][valid_screen]  # [B, COLOR_DIM, 21, 79]
    rare_occ_logits = model_output['rare_occ_logits'][valid_screen]  # [B, 1, 21, 79]
    rare_char_logits = model_output['rare_char_logits'][valid_screen]  # [B, CHAR_DIM, 21, 79]
    rare_color_logits = model_output['rare_color_logits'][valid_screen]  # [B, COLOR_DIM, 21, 79]
    msg_logits = model_output['msg_logits'][valid_screen]  # [B, 256, MSG_VSIZE]
    hero_presence_logits = model_output['hero_presence_logits'][valid_screen]  # [B]
    hero_centroid = model_output['hero_centroid'][valid_screen]  # [B, 2]

    valid_B = valid_screen.sum().item()  # Number of valid samples
    assert valid_B > 0, "No valid samples for loss calculation. Check valid_screen mask."
    
    # ============= Raw Reconstruction Losses =============
    raw_losses = {}
    
    # Glyph reconstruction (chars + colors)
    H, W = glyph_chars.shape[1], glyph_chars.shape[2]
    gc = glyph_chars[valid_screen]
    col_t = glyph_colors[valid_screen]
    fg = (gc != ord(' '))                 # [valid_B,H,W]
    common_fg = torch.zeros_like(fg, dtype=torch.bool)  # common glyphs mask
    for c in COMMON_GLYPHS:
        common_fg |= (gc == c[0]) & (col_t == c[1])
    hero_fg = torch.zeros_like(fg, dtype=torch.bool)  # hero glyphs mask
    hero_fg |= (gc == HERO_CHAR)
    rare_fg = fg & ~common_fg & ~hero_fg  # rare glyphs mask
    common_fg = fg & common_fg & ~hero_fg  # common glyphs mask
    occ_t = fg.float().unsqueeze(1)           # [valid_B,1,H,W]
    rare_occ_t = rare_fg.float().unsqueeze(1)  # [valid_B,1,H,W]
    common_occ_t = common_fg.float().unsqueeze(1)  # [valid_B,1,H,W]
    char_t = torch.clamp(gc - 32, 0, CHAR_DIM - 1)  # [valid_B,H,W]
    bag_t = make_pair_bag(glyph_chars[valid_screen], glyph_colors[valid_screen])  # [valid_B,GLYPH_DIM]
    hero_p_t, hero_c_t = hero_presence_and_centroid(glyph_chars[valid_screen]) # [valid_B], [valid_B,2]
    
    # ---- hero presence + centroid ----
    hero_bce = F.binary_cross_entropy_with_logits(hero_presence_logits, hero_p_t, pos_weight=torch.full_like(hero_presence_logits, 10))  # [valid_B]
    # centroid error only when hero exists (mask by target presence)
    hero_mse = ((hero_centroid - hero_c_t) ** 2).sum(dim=1) # [valid_B,2] -> [valid_B]
    hero_mse = hero_mse * hero_p_t
    hero_loss = hero_bce + 5.0 * hero_mse
    raw_losses['hero_loc'] = hero_loss.mean()  # Average over valid samples

    # occupancy focal BCE (mean over pixels)
    #occ_loss_per_sample = _bce_focal_with_logits(occupy_logits.squeeze(1), occ_t.squeeze(1), alpha_pos=focal_loss_alpha, gamma=focal_loss_gamma, reduction='none').sum(dim=[1, 2])  # [valid_B]
    fg_rate = 0.25
    with torch.no_grad():
        pos = occ_t.sum()
        neg = occ_t.numel() - pos
        pos_weight = (neg / pos).clamp(max=10.0)
    occ_loss_per_sample = F.binary_cross_entropy_with_logits(occupy_logits, occ_t, reduction='none', pos_weight=pos_weight).sum(dim=[1,2,3])  # [valid_B]
    assert occ_loss_per_sample.shape == (valid_B,), f"occupy loss shape mismatch: {occ_loss_per_sample.shape} != ({valid_B},)"
    
    rare_fg_rate = 0.05
    with torch.no_grad():
        rare_pos = rare_occ_t.sum()
        rare_neg = occ_t.sum() - rare_pos
        rare_pos_weight = (rare_neg / rare_pos).clamp(max=10.0)
    rare_occ_loss_per_sample = F.binary_cross_entropy_with_logits(rare_occ_logits, rare_occ_t, reduction='none', pos_weight=rare_pos_weight)
    rare_occ_loss_per_sample = (rare_occ_loss_per_sample * occ_t).sum(dim=[1,2,3])  # [valid_B]
    
    
    # total variation loss (TV) for occupancy map
    p = torch.sigmoid(occupy_logits)          # [B,1,H,W]
    dx = p[..., :, 1:] - p[..., :, :-1]
    dy = p[..., 1:, :] - p[..., :-1, :]
    tv = (dx.abs().mean() + dy.abs().mean()) * H * W

    # Dice (alternative or in addition; keep weight small)
    probs = p.squeeze(1); target = occ_t.squeeze(1)
    inter = (probs * target).sum(dim=(1,2))
    dice = (2*inter + 1e-5) / (probs.sum(dim=(1,2)) + target.sum(dim=(1,2)) + 1e-5)
    dice_loss = (1 - dice.mean()) * H * W

    # CE losses (masked by foreground)
    if fg.any():
        IGNORE = -100
        common_char_t_mask = torch.where(common_fg, char_t, torch.full_like(char_t, IGNORE))
        common_col_t_mask  = torch.where(common_fg, col_t,  torch.full_like(col_t,  IGNORE))
        rare_char_t_mask = torch.where(rare_fg, char_t, torch.full_like(char_t, IGNORE))
        rare_col_t_mask  = torch.where(rare_fg, col_t,  torch.full_like(col_t,  IGNORE))

        # CE computed only on FG via ignore_index
        rare_char_loss_per_sample = F.cross_entropy(
            rare_char_logits, rare_char_t_mask, ignore_index=IGNORE, label_smoothing=0.05, reduction='none'
        )
        assert rare_char_loss_per_sample.shape == (valid_B, H, W), f"rare_char_loss_per_sample shape mismatch: {rare_char_loss_per_sample.shape} != ({valid_B}, {H}, {W})"
        
        rare_color_loss_per_sample = F.cross_entropy(
            rare_color_logits, rare_col_t_mask, ignore_index=IGNORE, label_smoothing=0.05, reduction='none'
        )
        assert rare_color_loss_per_sample.shape == (valid_B, H, W), f"rare_color_loss_per_sample shape mismatch: {rare_color_loss_per_sample.shape} != ({valid_B}, {H}, {W})"

        common_char_loss_per_sample = F.cross_entropy(
            char_logits, common_char_t_mask, ignore_index=IGNORE, label_smoothing=0.05, reduction='none'
        )
        assert common_char_loss_per_sample.shape == (valid_B, H, W), f"common_char_loss_per_sample shape mismatch: {common_char_loss_per_sample.shape} != ({valid_B}, {H}, {W})"

        common_color_loss_per_sample = F.cross_entropy(
            color_logits, common_col_t_mask, ignore_index=IGNORE, label_smoothing=0.05, reduction='none'
        )
        assert common_color_loss_per_sample.shape == (valid_B, H, W), f"common_color_loss_per_sample shape mismatch: {common_color_loss_per_sample.shape} != ({valid_B}, {H}, {W})"

        # Sum over spatial dimensions for each sample
        rare_char_loss_per_sample = rare_char_loss_per_sample.sum(dim=[1, 2])  # [valid_B]
        rare_color_loss_per_sample = rare_color_loss_per_sample.sum(dim=[1, 2])  # [valid_B]
        common_char_loss_per_sample = common_char_loss_per_sample.sum(dim=[1, 2])  # [valid_B]
        common_color_loss_per_sample = common_color_loss_per_sample.sum(dim=[1, 2])
    else:
        rare_char_loss_per_sample = torch.zeros(valid_B, device=glyph_chars.device)
        rare_color_loss_per_sample = torch.zeros(valid_B, device=glyph_colors.device)
        common_char_loss_per_sample = torch.zeros(valid_B, device=glyph_chars.device)
        common_color_loss_per_sample = torch.zeros(valid_B, device=glyph_colors.device)

    # Average over valid samples only
    raw_losses['occupy'] = occ_loss_per_sample.mean() + 0.01 * tv + 0.1 * dice_loss  # Average over valid samples
    raw_losses['rare_occupy'] = rare_occ_loss_per_sample.mean()
    raw_losses['common_char'] = common_char_loss_per_sample.mean()
    raw_losses['common_color'] = common_color_loss_per_sample.mean()
    raw_losses['rare_char'] = rare_char_loss_per_sample.mean()
    raw_losses['rare_color'] = rare_color_loss_per_sample.mean()

    # stats
    pre = model_output['stats_encoder'].preprocessor
    pre = pre.to(blstats.device)
    pre.eval()
    with torch.no_grad():
        t = pre.targets(blstats[valid_screen])
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
    nll_cont = nll_cont.mean()
    hung_loss = F.cross_entropy(sp_filtered['hunger'], t['hunger'])
    dung_loss = F.cross_entropy(sp_filtered['dungeon'], t['dungeon'])
    level_loss = F.cross_entropy(sp_filtered['level'], t['level'])
    cond_loss = F.binary_cross_entropy_with_logits(sp_filtered['cond'], t['cond'])
    stats_loss = nll_cont + hung_loss + dung_loss + level_loss + cond_loss
    raw_losses['stats'] = stats_loss

    # message CE (mask PAD + after EOS)
    msg_t = msg_tokens[valid_screen]
    msg_loss = _message_ce_loss(msg_logits, msg_t, pad_id=MSG_PAD, eos_id=MSG_EOS)
    raw_losses['msg'] = msg_loss
    
    # ---- bag presence BCE (weighted) ----
    bag_logits = model_output["bag_logits"][valid_screen]
    # weights: emphasize '@' and de-emphasize very common chars
    with torch.no_grad():
        w = torch.ones(valid_B, GLYPH_DIM, device=bag_t.device) * 0.01
        # downweight common chars (all colors)
        common_idx = []
        for ch in COMMON_GLYPHS:
            ci = (ch[0] - 32)
            if 0 <= ci < CHAR_DIM:
                base = ci * COLOR_DIM + ch[1]
                common_idx.append(base)
        if common_idx:
            w[:, common_idx] = 0

        w[:, 0] = 0.0  # ignore space glyph
        hero_index = HERO_CHAR - 32
        w[:, hero_index * COLOR_DIM:(hero_index + 1) * COLOR_DIM] = 0.0  # ignore hero glyphs
        for ci in range(CHAR_DIM):
            w[:, ci * COLOR_DIM] = 0.0  # ignore all chars with no colors
        w[bag_t.bool()] = 1.0  # emphasize bag presence

    bag_bce = F.binary_cross_entropy_with_logits(bag_logits, bag_t, weight=w, reduction='none').sum(dim=1).mean()
    
    raw_losses['bag'] = bag_bce

    # Sum raw reconstruction losses
    total_raw_loss = sum(raw_losses[k] * raw_modality_weights.get(k, 1.0) for k in raw_losses)
    raw_losses['tv'] = tv  # Add total variation loss
    raw_losses['dice'] = dice_loss
    
    # KL divergence
    # Sigma_q = lowrank_factors @ lowrank_factors.T + torch.diag(torch.exp(logvar))
    # KL divergence for low-rank approximation
    eps = 1e-6
    var = torch.exp(logvar).clamp_min(eps)  # [valid_B, LATENT_DIM]
    mu2 = mu.square().sum(dim=1)  # mu^T * mu # [valid_B,]
    d = mu.size(1)
    if lowrank_factors is not None:
        U = lowrank_factors  # [valid_B, LATENT_DIM, LOW_RANK]
        UUT_bar = torch.zeros(d, d, device=mu.device, dtype=mu.dtype)  # [LATENT_DIM, LATENT_DIM]
        UUT_bar += torch.einsum('bik,bjk->ij', U, U) / valid_B # [LATENT_DIM, LATENT_DIM]
        Sigma_q_bar = UUT_bar + torch.diag(var.mean(dim=0))  # [LATENT_DIM, LATENT_DIM]
        tr_Sigma_q = var.sum(dim=1) + (U * U).sum(dim=(1,2))  # Trace of Sigma_q
        Dinvu = U / var.unsqueeze(-1)  # [valid_B, LATENT_DIM, LOW_RANK]
        I_plus = torch.bmm(U.transpose(1, 2), Dinvu)  # [valid_B, LATENT_DIM, LATENT_DIM]
        eye_d = torch.eye(d, device=mu.device, dtype=mu.dtype).unsqueeze(0)  # [1, LATENT_DIM, LATENT_DIM]
        I_plus = I_plus + eye_d  # [valid_B, LATENT_DIM, LATENT_DIM]
        sign2, logdet_small = torch.linalg.slogdet(I_plus)  # log(det(I + U^T * diag(1/var) * U))
        if not torch.all(sign2 > 0):
            raise ValueError("Matrix I + U^T * diag(1/var) * U is not positive definite.")
        log_det_Sigma_q = logdet_small + logvar.sum(dim=1)  # log(det(Sigma_q)) [valid_B,]
    else:
        # If no low-rank factors, just use diagonal covariance
        Sigma_q_bar = torch.diag(var.mean(dim=0))  # [LATENT_DIM, LATENT_DIM]
        tr_Sigma_q = var.sum(dim=1)  # Trace of Sigma_q
        log_det_Sigma_q = logvar.sum(dim=1)  # log(det(Sigma_q)) [valid_B,]
    # KL divergence: D_KL(q(z|x) || p(z)) =
    # 0.5 * (tr(Sigma_q) + mu^T * mu - d - log(det(Sigma_q)))
    # where d is the dimensionality of the latent space (LATENT_DIM)
    
    kl_per_sample = 0.5 * (tr_Sigma_q + mu2 - d - log_det_Sigma_q)  # [valid_B,]
    kl_loss = kl_per_sample.mean()  # Average over batch

    # Cov(mu) = E[mu * mu^T] - E[mu] * E[mu]^T
    mu_bar = mu.mean(dim=0)  # Average mean vector over batch
    mu_c = mu - mu_bar
    S_mu = (mu_c.t() @ mu_c) / valid_B  # Covariance matrix of the posterior distribution
    assert S_mu.shape == (d, d), f"Covariance matrix of mu shape mismatch: {S_mu.shape} != {(d, d)}"
    total_S = Sigma_q_bar + S_mu  # Total covariance matrix of the posterior distribution
    # Ensure the covariance matrix is symmetric
    total_S = (total_S + total_S.T) / 2 # [LATENT_DIM, LATENT_DIM]
    
    # Dimension-wise KL divergence with free bits regularization
    diag_S_bar = torch.diag(total_S).clamp_min(eps)  # Diagonal of the covariance matrix, [LATENT_DIM]
    dwkl_per_dim = 0.5 * (diag_S_bar + mu_bar.pow(2) - 1.0 - diag_S_bar.log())  # [LATENT_DIM]
    dwkl_fb_per_dim = torch.clamp(dwkl_per_dim - free_bits, min=0)  # Free bits regularization
    dwkl = dwkl_per_dim.sum()  # Sum over dimensions
    dwkl_fb = dwkl_fb_per_dim.sum()  # Free bits regularization sum
    
    # total correlation term - use FP32 for linear algebra operations
    with torch.amp.autocast('cuda', enabled=False):  # Disable autocast for numerical stability
        # Convert to FP32 for linear algebra operations
        diag_S_bar_fp32 = diag_S_bar.float()
        total_S_fp32 = total_S.float()
        
        inv_sqrt = torch.diag(diag_S_bar_fp32.sqrt().reciprocal())  # Inverse square root of diagonal covariance
        R = inv_sqrt @ total_S_fp32 @ inv_sqrt  # R = inv_sqrt * total_S * inv_sqrt
        R = 0.5 * (R + R.t()) + eps * torch.eye(d, device=mu.device, dtype=torch.float32)  # Ensure symmetry and positive definiteness
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
        'kl_diagnosis': kl_diagnosis
    }

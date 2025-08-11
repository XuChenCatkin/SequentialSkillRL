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
* inv_oclasses: LongTensor[B,55] - inventory object classes (0-18)
* inv_strs    : LongTensor[B,55,80] - inventory string descriptions (0-127 per char)

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
    6. InventoryEncoder: [B,55] & [B,55,80] → [B,24] - inventory features

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

# ------------------------- hyper‑params ------------------------------ #
CHAR_DIM = 96      # ASCII code space for characters shown on the map (32-127)
CHAR_EMB = 16      # char‑id embedding dim (0-15)
COLOR_DIM = 16      # colour id space for colours
COLOR_EMB = 4       # colour‑id embedding dim
GLYPH_EMB = CHAR_EMB + COLOR_EMB  # glyph embedding dim
LATENT_DIM = 96     # z‑dim for VAE
BLSTATS_DIM = 27   # raw scalar stats (hp, gold, …)
MSG_VOCAB = 128     # Nethack takes 32-127 byte for char of messages
GLYPH_DIM = CHAR_DIM * COLOR_DIM  # glyph dim for char + color
PADDING_IDX = [32, 0]  # padding index for glyphs (blank space char, black color)
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
# Inventory constants 
INV_MAX_SIZE = 55      # Maximum inventory size in NetHack
INV_STR_LEN = 80       # Maximum string length for inventory descriptions
INV_OCLASS_DIM = 19    # Object class cardinality (0-18)

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

# Inventory object class mappings
INV_OCLASS_MAP = {
    # 0 random object, not appear in inventory
    1:ord(']'), # Illegal object
    2:ord(')'), # Weapon
    3:ord('['), # Armor
    4:ord('='), # Ring
    5:ord('"'), # Amulet
    6:ord('('), # Tool
    7:ord('%'), # Food
    8:ord('!'), # Potion
    9:ord('?'), # Scroll
    10:ord('+'), # Spellbook
    11:ord('/'), # Wand
    12:ord('$'), # Coin
    13:ord('*'), # Gem
    14:ord('`'), # Rock
    15:ord('0'), # Iron Ball
    16:ord('_'), # Chains
    17:ord('.'), # Venom
    18: 128      # MAXOCLASSES, used for padding, use same as PADDING_IDX
}

def _gn_groups(c: int) -> int:
    """Pick a sensible GroupNorm group count that divides c."""
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1

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
            Z[bad] = self.empty_vec

        return Z.reshape(B, -1)                     # [B, K*C]

class GlyphCNN(nn.Module):
    """
    NetHack map encoder:
      - char & color embeddings (space can be true padding)
      - +1 foreground mask channel
      - +2 CoordConv channels (x,y in [-1,1])
      - Conv backbone mixing standard & anisotropic dilated convs
      - Masked attention pooling (K queries) or masked mean

    Forward returns a vector suitable for a VAE encoder MLP or heads.
    """
    def __init__(self, dropout=0.0, attn_queries=4, feat_channels=96):
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
        # glyph_chars are in the range of [32, 127] (ASCII codes)
        # shift to [0, 95] for model indices before embedding
        glyph_chars = torch.clamp(glyph_chars - 32, 0, CHAR_DIM - 1)  # Normalize to [0, 95]
        B, H, W = glyph_chars.shape
        device = glyph_chars.device
        fg_mask = (glyph_chars != 0)
        fg_mask = fg_mask.to(device=device).unsqueeze(1).float()  # [B, 1, H, W]
        char_embedding = self.char_emb(glyph_chars)  # [B, H, W] -> [B, H, W, 16]
        char_embedding = char_embedding.permute(0, 3, 1, 2)  # [B, H, W, 16] -> [B, 16, H, W]
        color_embedding = self.col_emb(glyph_colors)  # [B, H, W] -> [B, H, W, 4]
        color_embedding = color_embedding.permute(0, 3, 1, 2)  # [B, H, W, 4] -> [B, 4, H, W]
        glyph_emb = torch.cat([char_embedding, color_embedding], dim=1).contiguous()  # [B, 20, H, W]
        masked_glyph_emb = glyph_emb * fg_mask  # Apply foreground mask, [B, 20, H, W]
        feats = [masked_glyph_emb, fg_mask]
        cy, cx = self._coords(B, H, W, device, masked_glyph_emb.dtype)
        feats += [cy, cx]
        complete_glyph_emb = torch.cat(feats, dim=1)  # [B, 20+1+2, H, W]
        x = self.stem(complete_glyph_emb)  # [B, 64, H, W]
        x = self.body(x)  # [B, C, H, W]
        x = self.pool(x, fg_mask)  # [B, K*C] - pooled features
        x = self.norm_out(x)
        return x, char_embedding, color_embedding
    
    def _filter_topk_topp(self, logits: torch.Tensor, top_k: int, top_p: float):
        # logits: [B, C, H, W]
        B, C, H, W = logits.shape

        # ---- top-k ----
        if top_k > 0 and top_k < C:
            # kth value per pixel; keep anything >= kth (ties kept)
            kth_vals = torch.topk(logits, top_k, dim=1).values[:, -1:, :, :]  # [B,1,H,W]
            logits = logits.masked_fill(logits < kth_vals, float('-inf'))

        # ---- top-p (nucleus) ----
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, dim=1, descending=True)  # [B,C,H,W]
            probs = F.softmax(sorted_logits, dim=1)
            cumprobs = probs.cumsum(dim=1)

            remove = cumprobs > top_p                      # remove those past nucleus
            remove[:, 1:, :, :] = remove[:, :-1, :, :].clone()  # keep at least the first
            remove[:, 0, :, :] = False

            # also keep anything that was already -inf masked
            remove = remove | torch.isneginf(sorted_logits)

            to_remove = torch.zeros_like(logits, dtype=torch.bool)
            to_remove.scatter_(1, sorted_idx, remove)
            logits = logits.masked_fill(to_remove, float('-inf'))

        return logits

    def _sample_per_pixel(self, logits: torch.Tensor):
        # logits: [B, C, H, W]
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        # Guard: if a row sums to 0 (all -inf -> NaNs), fall back to argmax one-hot
        probs = torch.nan_to_num(probs, nan=0.0)
        zero_row = probs.sum(dim=1, keepdim=True) == 0
        if zero_row.any():
            argmax = logits.argmax(dim=1, keepdim=True)  # [B,1,H,W]
            one_hot = torch.zeros_like(probs).scatter_(1, argmax, 1.0)
            probs = torch.where(zero_row, one_hot, probs)

        flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        idx = torch.multinomial(flat, 1).squeeze(-1)               # [B*H*W]
        return idx.view(B, H, W)                                   # [B,H,W]

    def sample_from_logits(self, occupy_logits: torch.Tensor, glyph_logits: torch.Tensor, color_logits: torch.Tensor, temperature: float = 1.0, top_k: int = 5, top_p: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample glyphs from logits.
        Args:
            occupy_logits (torch.Tensor): Logits for occupied pixels, [B, 1, H, W].
            glyph_logits (torch.Tensor): Logits for glyph characters, [B, CHAR_DIM, H, W].
            color_logits (torch.Tensor): Logits for glyph colors, [B, COLOR_DIM, H, W].
            temperature (float): Temperature for sampling.
            top_k (int): Top-k filtering.
            top_p (float): Top-p (nucleus) filtering.
        """
        # Sample from categorical distributions
        # Apply temperature scaling and top-k/top-p filtering if needed
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        else:
            glyph_logits = glyph_logits / temperature
            color_logits = color_logits / temperature
        
        device = glyph_logits.device
        B, _, H, W = occupy_logits.shape
        
        occ_p = torch.sigmoid(occupy_logits).squeeze(1)  # [B,H,W]
        fg_mask = (occ_p > 0.5)  # foreground mask, [B,H,W]

        filtered_glyph_logits = self._filter_topk_topp(glyph_logits, top_k, top_p)
        filtered_color_logits = self._filter_topk_topp(color_logits, top_k, top_p)
        
        # Sample from the filtered logits
        glyph_samples = self._sample_per_pixel(filtered_glyph_logits)
        color_samples = self._sample_per_pixel(filtered_color_logits)

        # Convert samples back to original range
        glyph_samples = glyph_samples + 32  # Shift back to [32, 127]
        chars_out  = torch.full((B, H, W), 32, dtype=torch.long, device=device)
        colors_out = torch.full((B, H, W), 0, dtype=torch.long, device=device)
        chars_out[fg_mask]  = glyph_samples[fg_mask]
        colors_out[fg_mask] = color_samples[fg_mask]
        return fg_mask, chars_out, colors_out

    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return self.out_dim
    
class GlyphBag(nn.Module):
    """Encodes glyphs into a bag-of-glyphs representation (sorted)."""
    def __init__(self, max_len=64, logger: logging.Logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        if max_len <= 0:
            raise ValueError("max_len must be a positive integer.")
        if max_len > GLYPH_DIM:
            self.logger.warning(f"max_len ({max_len}) is larger than GLYPH_DIM ({GLYPH_DIM}), truncating to {GLYPH_DIM}.")
            max_len = GLYPH_DIM
        self.max_len = max_len  # max number of unique glyphs in the bag
        self.net = nn.RNN(GLYPH_EMB, 32, batch_first=True)  # simple RNN for bag encoding
            
        
    def encode_glyphs_to_bag(self, glyph_chars: torch.Tensor, glyph_colors: torch.Tensor) -> torch.Tensor:
        """Encodes glyphs into a bag-of-glyphs representation (sorted)."""
        # Create a bag of glyphs tensor
        B, H, W = glyph_chars.size()
        bag = torch.zeros(B, self.max_len, 2, dtype=torch.long, device=glyph_chars.device)
        
        # Encode each glyph as a unique id
        g_ids = torch.concat([glyph_chars.unsqueeze(1), glyph_colors.unsqueeze(1)], dim=1)  # [B, 2, H, W]
        g_ids = g_ids.view(B, 2, -1)  # flatten to [B, 2, H*W]
        # for each batch, create a sorted bag of glyphs
        for b in range(B):
            # g_ids[b] has shape [2, H*W], find unique glyph pairs along dim=1
            unique_glyphs, counts = torch.unique(g_ids[b], return_counts=True, dim=1)
            
            # Filter out blank space (32, 0) from unique_glyphs
            blank_space_mask = (unique_glyphs[0] == 32) & (unique_glyphs[1] == 0)
            non_blank_mask = ~blank_space_mask
            if non_blank_mask.any():
                unique_glyphs = unique_glyphs[:, non_blank_mask]
                counts = counts[non_blank_mask]
            else:
                # If all glyphs are blank space, keep empty tensors
                unique_glyphs = torch.empty((2, 0), dtype=unique_glyphs.dtype, device=unique_glyphs.device)
                counts = torch.empty((0,), dtype=counts.dtype, device=counts.device)
            
            # Sort by the glyph pairs - need to sort by both char and color
            # First sort by character (row 0), then by color (row 1) for ties
            if unique_glyphs.size(1) > 0:
                sorted_indices = torch.arange(unique_glyphs.size(1), device=unique_glyphs.device)
                perm_sec = torch.argsort(unique_glyphs[1], stable=True)  # sort by color
                sorted_indices = sorted_indices[perm_sec]  # apply color sort
                perm_pri = torch.argsort(unique_glyphs[0][sorted_indices], stable=True)  # sort by char
                sorted_indices = sorted_indices[perm_pri]  # apply char sort
                sorted_glyphs = unique_glyphs[:, sorted_indices]
                sorted_counts = counts[sorted_indices]
            else:
                # No non-blank glyphs found
                sorted_glyphs = unique_glyphs
                sorted_counts = counts
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Batch {b}: unique glyphs {unique_glyphs.size(1)}, counts {counts.size(0)}")
                self.logger.debug(f"Batch {b}: sorted glyphs {sorted_glyphs}, counts {sorted_counts}")
            if sorted_glyphs.size(1) > self.max_len:
                self.logger.warning(f"Batch {b}: Too many unique glyphs ({sorted_glyphs.size(1)}), truncating to {self.max_len}.")
            # Fill the bag with glyph ids (up to max_len)
            num_glyphs = min(sorted_glyphs.size(1), self.max_len)
            if num_glyphs > 0:
                bag[b, :num_glyphs] = sorted_glyphs[:, :num_glyphs].t()
            bag[b, num_glyphs:] = torch.tensor(PADDING_IDX, dtype=torch.long, device=bag.device)  # pad the rest
        return bag  # [B, max_len, 2] of glyph ids

    def forward(self, glyph_encoder: GlyphCNN, glyph_chars: torch.Tensor, glyph_colors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode glyphs to bag
        bag = self.encode_glyphs_to_bag(glyph_chars, glyph_colors)
        char_bag = bag[:, :, 0]  # [B, max_len] of char ids
        color_bag = bag[:, :, 1]  # [B, max_len] of color ids
        char_bag = torch.clamp(char_bag - 32, 0, CHAR_DIM - 1)  # Ensure char ids are in range [0, CHAR_DIM-1]
        char_emb = glyph_encoder.char_emb(char_bag)  # [B, max_len] -> [B, max_len, CHAR_EMB]
        color_emb = glyph_encoder.col_emb(color_bag)  # [B, max_len] -> [B, max_len, COLOR_EMB]
        emb = torch.cat([char_emb, color_emb], dim=2)  # [B, max_len, GLYPH_EMB]
        # Use RNN to encode the bag into a fixed-size vector
        padding_indices = torch.tensor(PADDING_IDX, dtype=torch.long, device=bag.device)  # [2]
        lengths = (bag != padding_indices).all(dim=-1).sum(dim=1).cpu()  # [B] - count non-padding glyphs
        
        # Initialize features tensor
        B = emb.size(0)
        features = torch.zeros(B, 32, device=emb.device)  # [B, 32] - output size of RNN
        
        # Find samples with non-zero lengths
        valid_mask = lengths > 0
        
        if valid_mask.any():
            # Extract embeddings and lengths for valid samples
            valid_emb = emb[valid_mask]  # [valid_B, max_len, GLYPH_EMB]
            valid_lengths = lengths[valid_mask]  # [valid_B]
            
            # Process valid embeddings through RNN
            packed_emb = nn.utils.rnn.pack_padded_sequence(valid_emb, lengths=valid_lengths, batch_first=True, enforce_sorted=False)
            _, h = self.net(packed_emb)  # h: [1, valid_B, 32]
            valid_features = h.squeeze(0)  # [valid_B, 32]
            
            # Assign valid features back to the full tensor
            features[valid_mask] = valid_features
        
        # Features for samples with lengths == 0 remain as zeros (already initialized)
        
        return features, emb, bag # [B, 32], [B, max_len, GLYPH_EMB], [B, max_len, 2]
        
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return 32

class BlstatsPreprocessor(nn.Module):
    """
    Preprocesses blstats (27-dimensional) for the VAE model.
    Uses a mixed approach: continuous normalization + discrete embeddings.
    
    Based on analysis from trial.ipynb:
    - blstats[0]: x_coordinate (0-78)
    - blstats[1]: y_coordinate (0-20)  
    - blstats[2-8]: attributes (strength, dex, con, int, wis, cha)
    - blstats[9]: score, blstats[13]: gold, blstats[19]: exp_points, blstats[20]: time
    - blstats[10-11]: hp/max_hp, blstats[14-15]: energy/max_energy
    - blstats[21]: hunger_state (0-6), blstats[23]: dungeon_num, blstats[24]: level_num
    - blstats[25]: condition_mask (bitfield) - special handling for multiple conditions
    - ignore game time (blstats[20]) and alignment (blstats[26] - already in hero_info)
    """
    def __init__(self, stats_dim=27):
        super().__init__()
        self.stats_dim = stats_dim
        self.useful_dims = [i for i in range(stats_dim) if i not in [20, 26]]  # Ignore time and alignment
        # Define discrete stats that need embeddings
        self.discrete_indices = [21, 23, 24]  # hunger_state, dungeon_number, level_number
        self.condition_index = 25  # Special handling for condition mask
        self.continuous_indices = [i for i in self.useful_dims if i not in self.discrete_indices + [self.condition_index]]
        
        # Embedding layers for discrete stats
        self.hunger_emb = nn.Embedding(7, 3)    # Hunger states (0-6)
        self.dungeon_emb = nn.Embedding(11, 4)  # Dungeon numbers (0-10)
        self.level_emb = nn.Embedding(51, 4)    # Level numbers (0-50)
        
        # Condition mask processing - decode each bit as separate binary features
        self.num_conditions = NUM_CONDITIONS
        self.condition_bits = CONDITION_BITS
        
        # BatchNorm for continuous stats
        self.batch_norm = nn.BatchNorm1d(len(self.continuous_indices) - 2)  # -2 for removed max_hp/max_energy
        
    def forward(self, blstats):
        """
        Args:
            blstats: [B, 27] tensor of raw blstats
        Returns:
            processed_stats: [B, processed_dim] tensor
        """

        # Process continuous stats
        continuous_stats = blstats[:, self.continuous_indices].float()  # [B, 21]
        
        # Process continuous stats - let BatchNorm handle most normalization
        continuous_processed = continuous_stats.clone()
        
        # Only apply essential transformations before BatchNorm
        # 1. Coordinate normalization (these have known, bounded ranges)
        continuous_processed[:, 0] = continuous_stats[:, 0] / MAX_X_COORD  # x coordinate (0-78) -> [0, 1]
        continuous_processed[:, 1] = continuous_stats[:, 1] / MAX_Y_COORD  # y coordinate (0-20) -> [0, 1]
        
        # 2. Log transform for heavily skewed distributions (but don't bound with tanh)
        log_indices = [9, 13, 19]  # score, gold, exp_points
        for idx in log_indices:
            if idx in self.continuous_indices:
                local_idx = self.continuous_indices.index(idx)
                # Apply log1p to handle skewness, let BatchNorm handle scaling
                continuous_processed[:, local_idx] = torch.log1p(torch.clamp(continuous_processed[:, local_idx], min=0))
        
        # 3. Keep other stats as-is - BatchNorm will normalize them appropriately
        # No need for manual normalization of attributes, depth, armor_class, etc.
        
        # Create health and energy ratios
        hp_ratio = continuous_stats[:, 10] / torch.clamp(continuous_stats[:, 11], min=1)  # hp/max_hp
        energy_ratio = continuous_stats[:, 14] / torch.clamp(continuous_stats[:, 15], min=1)  # energy/max_energy
        
        # Replace health and energy stats with ratios and remove max health/energy
        hp_idx = self.continuous_indices.index(10)  # hp
        max_hp_idx = self.continuous_indices.index(11)  # max_hp
        energy_idx = self.continuous_indices.index(14)  # energy
        max_energy_idx = self.continuous_indices.index(15)  # max_energy
        
        continuous_processed[:, hp_idx] = hp_ratio
        continuous_processed[:, energy_idx] = energy_ratio
        
        # Remove max health and max energy from continuous stats
        # Create mask to keep all indices except max_hp and max_energy
        keep_indices = [i for i in range(continuous_processed.size(1)) if i not in [max_hp_idx, max_energy_idx]]
        continuous_processed = continuous_processed[:, keep_indices]  # [B, 19]
        
        # Apply BatchNorm
        continuous_normalized = self.batch_norm(continuous_processed)  # [B, 19]
        
        # Process discrete stats
        # if the stats are out of bounds, log the issue
        if blstats[:, 21].max() > 6 or blstats[:, 21].min() < 0:
            logging.warning(f"Hunger state out of bounds: {blstats[:, 21].min()} to {blstats[:, 21].max()}")
        if blstats[:, 23].max() > 10 or blstats[:, 23].min() < 0:
            logging.warning(f"Dungeon number out of bounds: {blstats[:, 23].min()} to {blstats[:, 23].max()}")
        if blstats[:, 24].max() > 50 or blstats[:, 24].min() < 0:
            logging.warning(f"Level number out of bounds: {blstats[:, 24].min()} to {blstats[:, 24].max()}")
        hunger_state = torch.clamp(blstats[:, 21].long(), 0, 6)
        dungeon_num = torch.clamp(blstats[:, 23].long(), 0, 10)
        level_num = torch.clamp(blstats[:, 24].long(), 0, 50)
        
        hunger_emb = self.hunger_emb(hunger_state)
        dungeon_emb = self.dungeon_emb(dungeon_num)
        level_emb = self.level_emb(level_num)
        
        # Process condition mask (bitfield)
        condition_mask = blstats[:, self.condition_index].long()  # [B]
        condition_features = []
        for bit_value in self.condition_bits:
            # Extract each bit as a binary feature
            is_condition_active = ((condition_mask & bit_value) > 0).float()  # [B]
            condition_features.append(is_condition_active.unsqueeze(-1))  # [B, 1]
        condition_vector = torch.cat(condition_features, dim=-1)  # [B, 13]
        
        # Combine all features
        processed_stats = torch.cat([
            continuous_normalized,  # [B, 19]
            hunger_emb,            # [B, 3]
            dungeon_emb,           # [B, 4]
            level_emb,             # [B, 4]
            condition_vector       # [B, 13]
        ], dim=1) # [B, 19 + 3 + 4 + 4 + 13] = [B, 43]
        
        return processed_stats
    
    def get_output_dim(self):
        """Returns the output dimension of processed stats"""
        continuous_dim = len(self.continuous_indices) - 2  # -2 for removed max_hp/max_energy
        discrete_dim = 3 + 4 + 4  # hunger + dungeon + level embeddings (no alignment)
        condition_dim = self.num_conditions  # Binary features for each condition bit
        return continuous_dim + discrete_dim + condition_dim  # 19 + 11 + 13 = 43

class StatsMLP(nn.Module):
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
        return features, processed
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return 16

class MessageGRU(nn.Module):
    def __init__(self, vocab: int=MSG_VOCAB+2, emb: int=32, hid: int=12):
        super().__init__()
        self.emb_dim = emb
        self.hid_dim = hid
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)  # bidirectional GRU
        self.out = nn.Linear(hid * 2, hid)  # output layer to reduce to hid_dim
        self.sos = MSG_VOCAB  # start-of-sequence token
        self.eos = MSG_VOCAB + 1  # end-of-sequence token
        
    def forward(self, msg_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        msg_tokens: [B, 256] padded with 0s
        returns: [B, hid_dim] encoding, [B, 256, emb_dim] embeddings
        """
        # calculate lengths for packing
        lengths = (msg_tokens != 0).sum(dim=1)  # [B,] - count non-padding tokens
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
        return out, x  # return the final hidden state and the embeddings
    
    def teacher_forcing_decorator(self, msg_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        msg_tokens: [B, 256] padded with 0s
        returns: [B, 256] shifted and decorated msg_tokens, [B, 256, emb_dim] shifted embeddings
        """
        # calculate lengths for packing
        lengths = (msg_tokens != 0).sum(dim=1)  # [B,] - count non-padding tokens
        bempty = (lengths == 0)
        padded_msg_tokens = msg_tokens.clone()  # clone to avoid modifying original
        padded_msg_tokens[bempty, 0] = ord(' ')  # set space token for empty messages
        lengths[bempty] = 1  # set length to 1 for empty messages
        shifted_tokens = torch.zeros_like(padded_msg_tokens)
        shifted_tokens[:, 0] = self.sos  # set start-of-sequence token
        shifted_tokens[:, 1:] = padded_msg_tokens[:, :-1]  # shift right by 1
        # Get embeddings for shifted tokens
        shifted_embeddings = self.emb(shifted_tokens)  # [B, 256, emb_dim]
        return shifted_tokens, shifted_embeddings  # return shifted tokens and embeddings

    def sample_from_logits(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 5, top_p: float = 0.9) -> torch.Tensor:
        """
        Sample from the output logits using temperature, top-k, and top-p sampling.
        """
        # Apply temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k is not None and top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, top_k_indices, True)
            logits = logits.masked_fill(~mask, float('-inf'))

        # Top-p sampling
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            # Convert back to original indices and mask
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample from the final logits
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def get_output_channels(self) -> int:
        """Returns the number of output channels."""
        return self.hid_dim
    
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
        embeddings = {
            'role': role_emb,
            'race': race_emb,
            'gend': gend_emb,
            'align': align_emb
        }
        return hero_vec, embeddings  # [B, 16]
    
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
                 logger: logging.Logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.dropout_rate = dropout_rate
        self.enable_dropout_on_latent = enable_dropout_on_latent
        self.enable_dropout_on_decoder = enable_dropout_on_decoder
        
        self.glyph_cnn = GlyphCNN(dropout=self.dropout_rate)
        self.glyph_bag = GlyphBag(logger=self.logger)  # for bag of glyphs
        self.stats_mlp = StatsMLP()
        self.msg_gru   = MessageGRU()
        self.hero_emb = HeroEmbedding()  # for hero embedding
        
        self.include_glyph_bag = bInclude_glyph_bag
        self.include_hero = bInclude_hero

        fusion_in = self.glyph_cnn.get_output_channels() + \
                    self.stats_mlp.get_output_channels() + \
                    self.msg_gru.get_output_channels() + \
                    (self.glyph_bag.get_output_channels() if bInclude_glyph_bag else 0) + \
                    (self.hero_emb.get_output_channels() if bInclude_hero else 0) # cnn + stats + msg + bag of glyphs + hero embedding + inventory
        
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
        if self.dropout_rate > 0.0 and self.enable_dropout_on_decoder:
            self.decode_shared = nn.Sequential(
                nn.Linear(LATENT_DIM, 256), nn.ReLU(),
                nn.Dropout(self.dropout_rate),
            )
        else:
            self.decode_shared = nn.Sequential(
                nn.Linear(LATENT_DIM, 256), nn.ReLU(),
            )
        
        # ------------- Decode embeddings ----------------
        
        # from 256 to 20 * 21 * 79 using conv transpose
        # First reshape 256 -> (64, 1, 1), then upsample to (20, 21, 79)
        self.decode_glyph_emb = nn.Sequential(
            # Start: [B, 256] -> [B, 64, 1, 1]
            nn.Linear(256, 64 * 1 * 1),
            nn.Unflatten(1, (64, 1, 1)),  # [B, 64, 1, 1]
            
            # Upsample to target size: 21 x 79
            # Step 1: [B, 64, 1, 1] -> [B, 32, 3, 10] 
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 10), stride=1, padding=0),  # -> [B, 32, 3, 10]
            nn.ReLU(),
            
            # Step 2: [B, 32, 3, 10] -> [B, 20, 21, 79]
            # kernel_size needed: (21-3+1, 79-10+1) = (19, 70)
            nn.ConvTranspose2d(32, GLYPH_EMB, kernel_size=(19, 70), stride=1, padding=0)  # -> [B, 20, 21, 79]
        )
        
        # glyph bag embeddings
        # Note: glyph bag reconstruction (if required) is derived from char/color logits
        # No separate head needed since glyph bag is just unique combinations of char+color
        
        # stats embeddings
        self.decode_stats_emb = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.stats_mlp.preprocessor.get_output_dim())  # [B, 256] -> [B, 43]
        )
        
        # Message decoder - use unidirectional GRU with proper hidden state size
        self.decode_msg_latent2hidden = nn.Linear(256, self.msg_gru.hid_dim)  # [B, 256] -> [B, hid_dim]
        self.decode_msg_gru = nn.GRU(self.msg_gru.emb_dim, self.msg_gru.hid_dim, batch_first=True)  # [B, T, emb_dim] -> [B, T, hid_dim]
        self.decode_msg_hidden2emb = nn.Linear(self.msg_gru.hid_dim, self.msg_gru.emb_dim)  # [B, T, hid_dim] -> [B, T, emb_dim]
        
        # ------------- Decode logits ----------------
        # These will take the embedding as starting point
        
        # Separate heads for char and color reconstruction
        # these will return logits for each pixel
        self.decode_occupy = nn.Sequential(
            nn.Linear(CHAR_EMB, 8), nn.ReLU(),  # [B, 21, 79, 16] -> [B, 21, 79, 8]
            nn.Linear(8, 1)  # [B, 21, 79, 8] -> [B, 21, 79, 1] (occupy logits)
        )
        self.decode_chars = nn.Sequential(
            nn.Linear(CHAR_EMB, 32), nn.ReLU(),  # [B, 21, 79, 16] -> [B, 21, 79, 32]
            nn.Linear(32, CHAR_DIM)  # [B, 21, 79, 32] -> [B, 21, 79, CHAR_DIM] (char logits)
        )
        self.decode_colors = nn.Sequential(
            nn.Linear(COLOR_EMB, 8), nn.ReLU(),  # [B, 21, 79, 4] -> [B, 21, 79, 8]
            nn.Linear(8, COLOR_DIM)  # [B, 21, 79, 8] -> [B, 21, 79, COLOR_DIM] (color logits)
        )

        # Note: glyph bag reconstruction is derived from char/color logits
        # No separate head needed since glyph bag is just unique combinations of char+color
        
        # stats - decode to 24 relevant dimensions (excluding time and unknowns)
        # Split into discrete and continuous components
        self.decode_stats_continuous = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(self.stats_mlp.preprocessor.get_output_dim(), 64), nn.ReLU(),  # [B, 43] -> [B, 64]
            nn.Linear(64, 19),  # [B, 64] -> [B, 19] for normalized continuous stats (matches BatchNorm input)
        )
        
        # Discrete stats: hunger_state (0-6), dungeon_number (0-10), level_number (0-50)
        self.decode_hunger_state = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.stats_mlp.preprocessor.get_output_dim(), 32), nn.ReLU(),
            nn.Linear(32, 7),  # [B, 32] -> [B, 7] for hunger states
        )
        
        self.decode_dungeon_number = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.stats_mlp.preprocessor.get_output_dim(), 32), nn.ReLU(),
            nn.Linear(32, 11),  # [B, 32] -> [B, 11] for dungeon numbers
        )
        
        self.decode_level_number = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.stats_mlp.preprocessor.get_output_dim(), 32), nn.ReLU(),
            nn.Linear(32, 51),  # [B, 32] -> [B, 51] for level numbers
        )
        
        # Condition mask decoder - outputs 13 binary logits (one per condition bit)
        self.decode_condition_mask = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.stats_mlp.preprocessor.get_output_dim(), 32), nn.ReLU(),
            nn.Linear(32, 13),  # [B, 32] -> [B, 13] for condition bits (binary classification per bit)
        )
        
        # messages - use extended vocabulary to include SOS/EOS tokens
        self.decode_msg = nn.Linear(self.msg_gru.emb_dim, self.msg_gru.vocab)  # [B, T, emb_dim] -> [B, T, vocab_size]
        
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
        glyph_feat, char_emb, color_emb = self.glyph_cnn(glyph_chars, glyph_colors)  # [B,64], [B,16,H,W], [B,4,H,W]
        stats_feat, stats_emb = self.stats_mlp(blstats)
        
        # Message encoding
        msg_feat, msg_emb_no_shift = self.msg_gru(msg_tokens)  # [B,hid_dim], [B,256,emb_dim]
        msg_token_shift, msg_emb_shift = self.msg_gru.teacher_forcing_decorator(msg_tokens)  # [B,256] -> [B,256,emb_dim]
        
        features = [glyph_feat, stats_feat, msg_feat]
        
        if self.include_glyph_bag:
            glyph_bag_feat, glyph_bag_emb, glyph_bag = self.glyph_bag(self.glyph_cnn, glyph_chars, glyph_colors) # [B,32], [B,max_len,20], [B,max_len, 2]
            features.append(glyph_bag_feat)
        else:
            glyph_bag_emb = None
            glyph_bag = None
        
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
            hero_feat, hero_emb_dict = self.hero_emb(role, race, gend, align)  # [B,16], dict
            features.append(hero_feat)
        else:
            hero_emb_dict = None
        
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
            'target_char_emb': char_emb,  # [B, 16, 21, 79]
            'target_color_emb': color_emb,  # [B, 4, 21, 79]
            'target_glyph_bag_emb': glyph_bag_emb,  # [B, max_len, 20] or None
            'target_stats_emb': stats_emb,  # [B, 43]
            'target_msg_emb': msg_emb_no_shift,  # [B, T, emb_dim]
            'target_glyph_bag': glyph_bag,  # [B, max_len, 2]
            'target_hero_emb': hero_emb_dict,  # dict of hero embeddings or None
            
            'msg_token_shift': msg_token_shift,  # [B, 256]
            'msg_emb_shift': msg_emb_shift,  # [B, 256, emb_dim]
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

    def decode(self, z, msg_token_shift=None, msg_emb_shift=None, training_mode=False, max_length=256, temperature=1.0,
               top_k=5, top_p=0.9):
        """
        Decodes the latent variable z into observations.
        
        Args:
            z: [B, LATENT_DIM] - latent variable
            msg_token_shift: [B, 256] - shifted message tokens (optional, for teacher forcing)
            msg_emb_shift: [B, 256, emb_dim] - shifted message embeddings (optional, for teacher forcing)
            training_mode: bool - whether to use training mode (True for training, False for generation)
            max_length: int - maximum sequence length for autoregressive generation
            temperature: float - temperature for sampling (default 1.0)
            top_k: int - top-k sampling (default 5)
            top_p: float - top-p sampling (default 0.9)
        
        Returns:
            dict with decoded observations and embeddings
        """
        
        # Get batch size and device
        B = z.size(0)
        device = z.device
        
        # Decode
        shared_features = self.decode_shared(z)  # [B, 256]
        
        # Decode embeddings
        glyph_emb_decoded = self.decode_glyph_emb(shared_features)  # [B, 20, 21, 79]
        stats_emb_decoded = self.decode_stats_emb(shared_features)  # [B, 43]
        
        # Message decoding - support both teacher forcing and autoregressive generation
        msg_hidden_init = self.decode_msg_latent2hidden(shared_features)  # [B, 256] -> [B, hid_dim]
        
        if training_mode and msg_emb_shift is not None:
            # TRAINING MODE: Use teacher forcing with ground truth shifted sequences
            msg_lengths = (msg_token_shift != 0).sum(dim=1)  # [B,] - count non-padding tokens
            packed_msg_emb_shift = nn.utils.rnn.pack_padded_sequence(
                msg_emb_shift, lengths=msg_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            # Decode messages using GRU
            packed_msg_hidden_decoded, _ = self.decode_msg_gru(packed_msg_emb_shift, msg_hidden_init.unsqueeze(0))
            # Unpack the output - pad back to original length
            msg_hidden_decoded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_msg_hidden_decoded, batch_first=True, total_length=msg_emb_shift.size(1)
            )
            # Convert hidden states back to embeddings
            msg_emb_decoded = self.decode_msg_hidden2emb(msg_hidden_decoded)  # [B, 256, hid_dim] -> [B, 256, emb_dim]
            # Message decoding - GRU approach with proper teacher forcing
            msg_logits = self.decode_msg(msg_emb_decoded)  # [B, 256, emb_dim] -> [B, 256, MSG_VOCAB]
            generated_tokens = None  # Not generated in teacher forcing mode
            
        else:
            # GENERATION MODE: Autoregressive generation without teacher forcing
            generated_tokens = torch.zeros(B, max_length, dtype=torch.long, device=device)
            # Assume we have a start-of-sequence token (index 1) and end-of-sequence token (index 2)
            generated_tokens[:, 0] = self.msg_gru.sos  # Start with SOS token

            msg_emb_decoded = torch.zeros(B, max_length, self.msg_gru.emb_dim, device=device)
            msg_logits = torch.zeros(B, max_length, self.msg_gru.vocab, device=device)  # [B, T, vocab_size]
            # Initialize hidden state
            hidden = msg_hidden_init.unsqueeze(0)  # [1, B, hid_dim] for GRU
            
            # Generate step by step
            for t in range(max_length - 1):
                # Get embedding for current token
                current_token = generated_tokens[:, t]  # [B]
                current_emb = self.msg_gru.emb(current_token).unsqueeze(1)  # [B, 1, emb_dim]
                
                # Run GRU for one step
                gru_out, hidden = self.decode_msg_gru(current_emb, hidden)  # gru_out: [B, 1, hid_dim]
                
                # Convert hidden to embedding
                step_emb = self.decode_msg_hidden2emb(gru_out.squeeze(1))  # [B, emb_dim]
                msg_emb_decoded[:, t] = step_emb
                
                # Get logits and sample next token
                step_logits = self.decode_msg(step_emb.unsqueeze(1))  # [B, 1, vocab_size]
                msg_logits[:, t] = step_logits.squeeze(1)  # [B, vocab_size]
                
                # sampling
                next_token = self.msg_gru.sample_from_logits(
                    step_logits.squeeze(1), 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p
                ).squeeze(-1)  # [B, 1] -> [B]
                
                generated_tokens[:, t + 1] = next_token
                
                # Early stopping if all sequences have generated EOS
                if torch.all(next_token == self.msg_gru.eos):
                    break
            
        # logits
        # char logits takes first 16 channels of glyph_emb_decoded
        # color logits takes last 4 channels of glyph_emb_decoded
        occupy_logits = self.decode_occupy(glyph_emb_decoded[:, :CHAR_EMB, :, :].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 16, 21, 79] -> [B, 1, 21, 79]
        char_logits = self.decode_chars(glyph_emb_decoded[:, :CHAR_EMB, :, :].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 16, 21, 79] -> [B, 96, 21, 79]
        color_logits = self.decode_colors(glyph_emb_decoded[:, CHAR_EMB:, :, :].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 4, 21, 79] -> [B, 16, 21, 79]
        if training_mode:
            # During training, we return logits directly
            generated_occupy = None
            generated_chars = None
            generated_colors = None
        else:
            # During generation, sample from logits
            generated_occupy, generated_chars, generated_colors = self.glyph_cnn.sample_from_logits(
                occupy_logits,
                char_logits,
                color_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode stats - separate into continuous and discrete components
        stats_continuous_normalized = self.decode_stats_continuous(stats_emb_decoded)  # [B, 19]
        
        hunger_state_logits = self.decode_hunger_state(stats_emb_decoded)      # [B, 7]
        dungeon_number_logits = self.decode_dungeon_number(stats_emb_decoded)  # [B, 11]
        level_number_logits = self.decode_level_number(stats_emb_decoded)      # [B, 51]
        condition_mask_logits = self.decode_condition_mask(stats_emb_decoded)  # [B, 13]

        return {
            'glyph_emb_decoded': glyph_emb_decoded,  # [B, 20, 21, 79]
            'stats_emb_decoded': stats_emb_decoded,  # [B, 43]
            'msg_emb_decoded': msg_emb_decoded,  # [B, 256, emb_dim]
            'occupy_logits': occupy_logits,  # [B, 1, 21, 79]
            'char_logits': char_logits, # [B, 256, 21, 79]
            'color_logits': color_logits,  # [B, 16, 21, 79]
            'stats_continuous_normalized': stats_continuous_normalized,  # [B, 19]
            'hunger_state_logits': hunger_state_logits,  # [B, 7]
            'dungeon_number_logits': dungeon_number_logits,  # [B, 11]
            'level_number_logits': level_number_logits,  # [B, 51]
            'condition_mask_logits': condition_mask_logits,  # [B, 13]
            'msg_logits': msg_logits, # [B, 256, MSG_VOCAB]
            'generated_occupy': generated_occupy,  # [B, 21, 79]
            'generated_chars': generated_chars,  # [B, 21, 79]
            'generated_colors': generated_colors,  # [B, 21, 79]
            'generated_tokens': generated_tokens,  # [B, max_length] or None if teacher forcing
        }

    def forward(self, glyph_chars, glyph_colors, blstats, msg_tokens, hero_info=None, training_mode=True, temperature=1.0, top_k=5, top_p=0.9):
        
        fg = glyph_chars != ord(' ')  # foreground mask for glyphs
        bad = (glyph_colors == 0) & fg  # bad pixels where color is 0 and glyph is not space
        if bad.any():
            i = bad.nonzero()[0].tolist()
            raise RuntimeError(f"Found color==0 on non-space at index {i} — cannot use 0 as padding_idx.")
        
        encoded_vars = self.encode(glyph_chars, glyph_colors, blstats, msg_tokens, hero_info)
        
        mu = encoded_vars['mu']  # [B, LATENT_DIM]
        logvar = encoded_vars['logvar']  # [B, LATENT_DIM]
        lowrank_factors = encoded_vars['lowrank_factors']
        
        msg_token_shift = encoded_vars['msg_token_shift']  # [B, 256]
        msg_emb_shift = encoded_vars['msg_emb_shift']  # [B, 256, emb_dim]
        
        z = self._reparameterise(mu, logvar, lowrank_factors)
        
        # Decode with teacher forcing (training mode)
        decoded_vars = self.decode(
            z, 
            msg_token_shift=msg_token_shift,
            msg_emb_shift=msg_emb_shift,
            training_mode=training_mode,  # Use teacher forcing during training
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return {
            **encoded_vars,  # Contains mu, logvar, embeddings, etc.
            **decoded_vars   # Contains all decoded outputs including generated_tokens
        }
        
    def get_dropout_config(self):
        """
        Get dropout configuration for logging and debugging
        
        Returns:
            Dict containing dropout configuration
        """
        return {
            'dropout_rate': self.dropout_rate,
            'enable_dropout_on_latent': self.enable_dropout_on_latent,
            'enable_dropout_on_decoder': self.enable_dropout_on_decoder,
            'training_mode': self.training,
            'dropout_active': self.training and self.dropout_rate > 0.0
        }

# ------------------------- loss helpers ------------------------------ #

def vae_loss(
    model_output, 
    glyph_chars, 
    glyph_colors, 
    blstats, 
    msg_tokens,
    valid_screen,
    raw_modality_weights={'occupy': 10.0, 'char': 1.0, 'color': 1.0, 'stats': 1.0, 'msg': 1.0},
    emb_modality_weights={'char_emb': 0.05, 'color_emb': 0.05, 'stats_emb': 1.0, 'msg_emb': 0.1},
    focal_loss_alpha=0.75,
    focal_loss_gamma=2.0,
    weight_emb=1.0, 
    weight_raw=0.1, 
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
    occupy_logits = model_output['occupy_logits']  # [B, 1, 21, 79]
    char_logits = model_output['char_logits']  # [B, CHAR_DIM, 21, 79]
    color_logits = model_output['color_logits']  # [B, COLOR_DIM, 21, 79]
    msg_logits = model_output['msg_logits']  # [B, 256, MSG_VOCAB+2]
    
    # Embedding reconstructions (continuous targets)
    glyph_emb_decoded = model_output['glyph_emb_decoded']  # [B, 20, 21, 79]
    stats_emb_decoded = model_output['stats_emb_decoded']  # [B, 43]
    msg_emb_decoded = model_output['msg_emb_decoded']  # [B, 256, emb_dim]
    
    valid_B = valid_screen.sum().item()  # Number of valid samples
    assert valid_B > 0, "No valid samples for loss calculation. Check valid_screen mask."
    
    # ============= Raw Reconstruction Losses =============
    raw_losses = {}
    
    # Glyph reconstruction (chars + colors)
    H, W = glyph_chars.shape[1], glyph_chars.shape[2]
    fg_mask = (glyph_chars != ord(' '))  # [B, H, W]
    ooc_t = fg_mask.float().unsqueeze(1)  # [B, 1, H, W]
    masked_num = fg_mask.sum(dim=[1, 2]).clamp(min=1)
    glyph_chars_ce = torch.clamp(glyph_chars - 32, 0, CHAR_DIM - 1)  # CE targets
    
    # Occupancy focal loss
    occ_bce = F.binary_cross_entropy_with_logits(occupy_logits, ooc_t, reduction='none').squeeze(1)  # [B, H, W]
    p = torch.sigmoid(occupy_logits).squeeze(1)
    ooc_hw = ooc_t.squeeze(1)
    p_t = torch.where(ooc_hw > 0.5, p, 1.0 - p)
    alpha_pos = torch.as_tensor(focal_loss_alpha, device=occupy_logits.device, dtype=occupy_logits.dtype)
    alpha_neg = torch.as_tensor(1.0 - focal_loss_alpha, device=occupy_logits.device, dtype=occupy_logits.dtype)
    alpha_t = torch.where(ooc_hw > 0.5, alpha_pos, alpha_neg)
    focal_weight = alpha_t * (1.0 - p_t).pow(focal_loss_gamma)
    focal_loss_per_sample = (focal_weight * occ_bce).sum(dim=(1, 2)) # [B]

    # CE losses (masked by foreground)
    char_loss_per_sample = F.cross_entropy(char_logits, glyph_chars_ce, reduction='none')  # [B, H, W]
    masked_char_loss_per_sample = char_loss_per_sample * fg_mask.float()
    color_loss_per_sample = F.cross_entropy(color_logits, glyph_colors, reduction='none')  # [B, H, W]
    masked_color_loss_per_sample = color_loss_per_sample * fg_mask.float()

    # Sum over spatial dimensions for each sample
    masked_char_loss_per_sample = masked_char_loss_per_sample.sum(dim=[1, 2])  # [B]
    masked_color_loss_per_sample = masked_color_loss_per_sample.sum(dim=[1, 2])  # [B]

    # Average over valid samples only
    raw_losses['occupy'] = focal_loss_per_sample[valid_screen].mean()  # Average over valid samples
    raw_losses['char'] = masked_char_loss_per_sample[valid_screen].mean()
    raw_losses['color'] = masked_color_loss_per_sample[valid_screen].mean()

    # Stats reconstruction
    stats_continuous_normalized = model_output['stats_continuous_normalized']  # [B, 19]
    hunger_state_logits = model_output['hunger_state_logits']
    dungeon_number_logits = model_output['dungeon_number_logits']
    level_number_logits = model_output['level_number_logits']
    condition_mask_logits = model_output['condition_mask_logits']

    # Discrete targets (clamped)
    hunger_target = torch.clamp(blstats[:, 21].long(), 0, 6)
    dungeon_target = torch.clamp(blstats[:, 23].long(), 0, 10)
    level_target = torch.clamp(blstats[:, 24].long(), 0, 50)

    # Condition mask target vector
    condition_mask_target = blstats[:, 25].long()
    condition_target_vector = []
    for bit_value in CONDITION_BITS:
        is_condition_active = ((condition_mask_target & bit_value) > 0).float()  # [B]
        condition_target_vector.append(is_condition_active)
    condition_target_vector = torch.stack(condition_target_vector, dim=1)  # [B, 13]

    # Continuous targets in BN-normalized space: first 19 dims of target_stats_emb
    target_stats_emb = model_output['target_stats_emb']  # [B, 43]
    continuous_targets_bn = target_stats_emb[:, :19]

    stat_raw_losses = {}
    stats_continuous_loss = F.mse_loss(stats_continuous_normalized, continuous_targets_bn, reduction='none').sum(dim=1)
    stat_raw_losses['stats_continuous'] = stats_continuous_loss[valid_screen].mean()
    stat_raw_losses['stats_hunger'] = F.cross_entropy(hunger_state_logits[valid_screen], hunger_target[valid_screen], reduction='mean')
    stat_raw_losses['stats_dungeon'] = F.cross_entropy(dungeon_number_logits[valid_screen], dungeon_target[valid_screen], reduction='mean')
    stat_raw_losses['stats_level'] = F.cross_entropy(level_number_logits[valid_screen], level_target[valid_screen], reduction='mean')
    condition_loss = F.binary_cross_entropy_with_logits(condition_mask_logits, condition_target_vector, reduction='none').sum(dim=1)
    stat_raw_losses['stats_condition'] = condition_loss[valid_screen].mean()

    raw_losses['stats'] = (stat_raw_losses['stats_continuous'] +
                           stat_raw_losses['stats_hunger'] +
                           stat_raw_losses['stats_dungeon'] +
                           stat_raw_losses['stats_level'] +
                           stat_raw_losses['stats_condition'])

    # Message reconstruction (token CE with EOS injection as before)
    msg_lengths = (msg_tokens != 0).sum(dim=1)
    msg_tokens_with_eos = msg_tokens.clone()
    mask_has_space = msg_lengths < msg_tokens.size(1)
    msg_tokens_with_eos[mask_has_space, msg_lengths[mask_has_space]] = MSG_VOCAB + 1
    msg_loss_per_token = F.cross_entropy(
        msg_logits.view(-1, msg_logits.size(-1)),
        msg_tokens_with_eos.view(-1),
        reduction='none',
        ignore_index=0
    )  # [B*seq_len]
    
    # Reshape and sum over sequence length for each sample
    msg_loss_per_sample = msg_loss_per_token.view(msg_logits.size(0), -1).sum(dim=1)  # [B] - sum over seq_len
    raw_losses['msg'] = msg_loss_per_sample[valid_screen].mean()  # Average over valid samples
    
    # ============= Embedding Reconstruction Losses =============
    emb_losses = {}
    target_char_emb = model_output['target_char_emb']  # [B, 16, 21, 79]
    target_color_emb = model_output['target_color_emb']  # [B, 4, 21, 79]
    target_stats_emb = model_output['target_stats_emb']  # [B, 43]

    char_emb_recon, color_emb_recon = torch.split(glyph_emb_decoded, [CHAR_EMB, COLOR_EMB], dim=1)
    char_diff = (char_emb_recon[valid_screen] - target_char_emb[valid_screen].detach()).pow(2).sum(dim=1)  # [valid_B, 21, 79]
    color_diff = (color_emb_recon[valid_screen] - target_color_emb[valid_screen].detach()).pow(2).sum(dim=1) # [valid_B, 21, 79]
    w = 1.0 * fg_mask[valid_screen].float() + 0.05 * (1.0 - fg_mask[valid_screen].float())  # [valid_B, 21, 79]

    emb_losses['char_emb'] = (char_diff * w).sum() / valid_B  # Average over valid samples
    emb_losses['color_emb'] = (color_diff * w).sum() / valid_B  # Average over valid samples
    emb_losses['stats_emb'] = F.mse_loss(stats_emb_decoded[valid_screen], target_stats_emb[valid_screen].detach(), reduction='sum') / valid_B

    # Message embedding loss with mask on shifted tokens (exclude padding)
    # Use non-shifted embeddings with EOS as targets to align with next-token prediction
    target_msg_emb = model_output['target_msg_emb']  # [B, T, emb_dim]
    msg_mask = (msg_tokens_with_eos != 0).float()  # [B, T]
    diff_sq = (msg_emb_decoded - target_msg_emb.detach()).pow(2).sum(dim=2)  # [B, T]
    diff_sq = diff_sq * msg_mask  # mask padding (including positions beyond EOS)
    diff_sq_valid = diff_sq[valid_screen]
    emb_losses['msg_emb'] = diff_sq_valid.sum() / valid_B

    # ============= Combine Losses =============
    
    # Sum raw reconstruction losses
    total_raw_loss = sum(raw_losses[k] * raw_modality_weights.get(k, 1.0) for k in raw_losses)
    
    # Sum embedding losses (when implemented)
    total_emb_loss = sum(emb_losses[k] * emb_modality_weights.get(k, 1.0) for k in emb_losses)

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
        I_plus = torch.malmul(U.transpose(1, 2), Dinvu)  # [valid_B, LATENT_DIM, LATENT_DIM]
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
    
    # total correlation term
    inv_sqrt = torch.diag(diag_S_bar.sqrt().reciprocal())  # Inverse square root of diagonal covariance
    R = inv_sqrt @ total_S @ inv_sqrt  # R = inv_sqrt * total_S * inv_sqrt
    R = 0.5 * (R + R.t()) + eps * torch.eye(d, device=mu.device, dtype=mu.dtype)  # Ensure symmetry and positive definiteness
    sign, logabsdet_R = torch.linalg.slogdet(R)  # log(det(R))
    if not torch.all(sign > 0):
        raise ValueError("Covariance matrix R is not positive definite.")
    total_correlation = -0.5 * logabsdet_R  # Total correlation term
    
    mutual_information = kl_loss - total_correlation - dwkl  # Mutual information term
    
    with torch.no_grad():
        # Compute the eigenvalues and eigenvectors
        eigenvalues = torch.linalg.eigvalsh(total_S)
        kl_diagnosis = {
            'mutual_info': float(mutual_information.item()),
            'total_correlation': float(total_correlation.item()),
            'dimension_wise_kl': dwkl_per_dim,
            'dimension_wise_kl_sum': float(dwkl.item()),
            'eigenvalues': eigenvalues
        }
    
    # Total weighted loss
    total_loss = (weight_raw * total_raw_loss + 
                  weight_emb * total_emb_loss + 
                  mi_beta * mutual_information + 
                  tc_beta * total_correlation +
                  dw_beta * dwkl_fb)  # Free bits regularization
    
    return {
        'total_loss': total_loss,
        'total_raw_loss': total_raw_loss,
        'total_emb_loss': total_emb_loss,
        'kl_loss': kl_loss,
        'raw_losses': raw_losses,
        'emb_losses': emb_losses,
        'raw_stats_losses': stat_raw_losses,
        'kl_diagnosis': kl_diagnosis
    }

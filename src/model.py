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
LATENT_DIM = 64     # z‑dim for VAE
BLSTATS_DIM = 27   # raw scalar stats (hp, gold, …)
MSG_VOCAB = 128     # Nethack takes 0-127 byte for char of messages
GLYPH_DIM = CHAR_DIM * COLOR_DIM  # glyph dim for char + color
PADDING_IDX = [128, COLOR_DIM]  # padding index for glyphs
LOW_RANK = 4      # low-rank factorisation rank for covariance
# NetHack map dimensions (from NetHack source: include/config.h)
MAP_HEIGHT = 21     # ROWNO - number of rows in the map
MAP_WIDTH = 79      # COLNO-1 - playable columns (COLNO=80, but rightmost is UI)
MAX_X_COORD = 78    # Maximum x coordinate (width - 1)
MAX_Y_COORD = 20    # Maximum y coordinate (height - 1)
ROLE_CAD = 13     # role cardinality for MiniHack (e.g. 4 for 'knight')
RACE_CAD = 5      # race cardinality for MiniHack (e.g. 0 for 'human')
GEND_CAD = 2      # gender cardinality for MiniHack (e.g. 0 for 'female')
ALIGN_CAD = 3     # alignment cardinality for MiniHack (e.g. 0 for 'lawful')
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

class GlyphCNN(nn.Module):
    """3-layer conv on [B, C= (char_emb+color_emb), H, W]."""
    def __init__(self):
        super().__init__()
        self.char_emb = nn.Embedding(CHAR_DIM + 1, CHAR_EMB)
        self.col_emb  = nn.Embedding(COLOR_DIM + 1, COLOR_EMB)
        self.net = nn.Sequential(
            nn.Conv2d(GLYPH_EMB, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # -> [B,64,1,1]
        )
        
    def forward(self, glyph_chars: torch.Tensor, glyph_colors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        char_embedding = self.char_emb(glyph_chars)  # [B, H, W] -> [B, H, W, 16]
        char_embedding = char_embedding.permute(0, 3, 1, 2)  # [B, H, W, 16] -> [B, 16, H, W]
        color_embedding = self.col_emb(glyph_colors)  # [B, H, W] -> [B, H, W, 4]
        color_embedding = color_embedding.permute(0, 3, 1, 2)  # [B, H, W, 4] -> [B, 4, H, W]
        glyph_emb = torch.cat([char_embedding, color_embedding], dim=1)  # [B, 20, H, W]
        x = self.net(glyph_emb)
        feature = x.view(x.size(0), -1)  # [B,64]
        return feature, char_embedding, color_embedding
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return 64
    
class GlyphBag(GlyphCNN):
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
            unique_glyphs, counts = torch.unique(g_ids[b], return_counts=True, dim=1)
            sorted_indices = torch.argsort(unique_glyphs)
            sorted_glyphs = unique_glyphs[sorted_indices]
            sorted_counts = counts[sorted_indices]
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Batch {b}: unique glyphs {unique_glyphs.size(0)}, counts {counts.size(0)}")
                self.logger.debug(f"Batch {b}: sorted glyphs {sorted_glyphs}, counts {sorted_counts}")
            if len(sorted_glyphs) > self.max_len:
                self.logger.warning(f"Batch {b}: Too many unique glyphs ({len(sorted_glyphs)}), truncating to {self.max_len}.")
            # Fill the bag with glyph ids (up to max_len)
            bag[b, :len(sorted_glyphs)] = sorted_glyphs[:self.max_len]
            bag[b, len(sorted_glyphs):] = PADDING_IDX  # pad the rest
        return bag  # [B, max_len, 2] of glyph ids

    def forward(self, glyph_chars: torch.Tensor, glyph_colors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode glyphs to bag
        bag = self.encode_glyphs_to_bag(glyph_chars, glyph_colors)
        char_bag = bag[:, :, 0]  # [B, max_len] of char ids
        color_bag = bag[:, :, 1]  # [B, max_len] of color ids
        char_emb = super().char_emb(char_bag)  # [B, max_len] -> [B, max_len, CHAR_EMB]
        color_emb = super().col_emb(color_bag)  # [B, max_len] -> [B, max_len, COLOR_EMB]
        emb = torch.cat([char_emb, color_emb], dim=2)  # [B, max_len, GLYPH_EMB]
        # Use RNN to encode the bag into a fixed-size vector
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths=(bag != PADDING_IDX).sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.net(packed_emb)  # h: [1,B,32]
        features = h.squeeze(0)  # [B,32]
        return features, emb, bag # [B, max_len, GLYPH_EMB], [B, max_len, 2]
        
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
        msg_tokens_with_eos = msg_tokens.clone()
        bHasSpace = (lengths < msg_tokens.size(1))  # check if there is space for EOS
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
        shifted_tokens = torch.zeros_like(msg_tokens)
        shifted_tokens[:, 0] = self.sos  # set start-of-sequence token
        shifted_tokens[:, 1:] = msg_tokens[:, :-1]  # shift right by 1
        bHasSpace = (lengths < msg_tokens.size(1) - 1)  # check if there is space for EOS
        shifted_tokens[bHasSpace, lengths[bHasSpace]+1] = self.eos  # set end-of-sequence token at the end of each message
        # Get embeddings for shifted tokens
        shifted_embeddings = self.emb(shifted_tokens)  # [B, 256, emb_dim]
        return shifted_tokens, shifted_embeddings  # return shifted tokens and embeddings
        
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
    

class InventoryEncoder(GlyphCNN):
    """
    Encodes inventory using object classes (inv_oclasses) and string descriptions (inv_strs).
    We use char embeddings from GlyphCNN for object classes.
    
    Architecture:
    - Object classes: [B, 55] -> [B, hid] using a simple RNN
    - String descriptions: [B, 55, 80] -> [B, hid * 2] using a bidirectional GRU
    - Fusion: Concatenate object class and string embeddings, then pass through a linear layer to output [B, output_dim].
    
    This approach avoids using inv_glyphs which can be inconsistent and instead uses
    semantic object class information and natural language descriptions.
    """
    def __init__(self, vocab: int=MSG_VOCAB + 2, str_emb: int=16, hid: int=16, output_dim: int=24):
        super().__init__()
        self.output_dim = output_dim
        self.vocab = vocab
        self.str_emb_dim = str_emb
        self.hid_dim = hid
        self.sos = MSG_VOCAB  # start-of-sequence token for strings
        self.eos = MSG_VOCAB + 1  # end-of-sequence token
        
        self.oclass_net = nn.RNN(CHAR_EMB, hid, batch_first=True)  # simple RNN for object class encoding
        
        # Simple character-level processing
        self.char_emb = nn.Embedding(vocab, str_emb, padding_idx=0)
        self.str_gru = nn.GRU(str_emb, hid, batch_first=True, bidirectional=True)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hid * 3, hid),
            nn.ReLU(),
            nn.Linear(hid, output_dim)
        )
        
    def forward(self, inv_oclasses: torch.Tensor, inv_strs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inv_oclasses: [B, 55] - object class indices
            inv_strs: [B, 55, 80] - string descriptions (character codes)
        Returns:
            inv_features: [B, output_dim] - encoded inventory features
        """
        B = inv_oclasses.size(0)
        # map inv_oclasses to char
        inv_class = map(lambda x: INV_OCLASS_MAP.get(x, 128), inv_oclasses.view(-1).tolist())  # flatten and map
        inv_class = torch.tensor(list(inv_class), dtype=torch.long, device=inv_oclasses.device).view(B, INV_MAX_SIZE)  # [B, 55]
        
        # Process object classes
        inv_class_emb = super().char_emb(inv_class)  # [B, 55] -> [B, 55, CHAR_EMB]
        packed_inv_class_emb = nn.utils.rnn.pack_padded_sequence(
            inv_class_emb, (inv_oclasses != 18).sum(dim=1).cpu(),
            batch_first=True, enforce_sorted=False
        )  # Pack the sequences for RNN
        _, h_class_n = self.oclass_net(packed_inv_class_emb)  # h_class_n: [1, B, hid]
        h_class = h_class_n.squeeze(0)  # [B, hid]
        
        # Process string descriptions
        inv_strs_with_eos = inv_strs.clone().view(B * INV_MAX_SIZE, INV_STR_LEN)  # Flatten to [B*55, 80]
        lengths = (inv_strs_with_eos != 0).sum(dim=-1)  # [B*55,] - count non-padding characters
        bHasSpace = (lengths < INV_STR_LEN)  # check if there is space for EOS
        inv_strs_with_eos[bHasSpace, lengths[bHasSpace]] = self.eos  # set end-of-sequence token at the end of each string
        str_emb = self.char_emb(inv_strs_with_eos)  # [B*55, 80, str_emb]
        packed_str_emb = nn.utils.rnn.pack_padded_sequence(
            str_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )  # Pack the sequences for GRU
        
        # GRU over characters
        packed_out, h_n = self.gru(packed_str_emb)  # hidden: [2, B*55, hid]
        # Use final hidden state (concat forward and backward)
        str_final = torch.cat([h_n[0], h_n[1]], dim=1)  # [B*55, hid * 2]
        str_final = str_final.view(B, INV_MAX_SIZE, self.hid_dim * 2)  # [B, 55, hid * 2]
        
        # Average pool over inventory slots (mask out empty slots)
        str_mask = (inv_strs.sum(dim=-1) > 0).float().unsqueeze(-1)  # [B, 55, 1]
        str_masked = str_final * str_mask  # [B, 55, hid * 2]
        str_pooled = str_masked.sum(dim=1) / (str_mask.sum(dim=1) + 1e-8)  # [B, hid * 2]
        
        # Fusion
        combined = torch.cat([h_class, str_pooled], dim=-1)  # [B, hid] + [B, hid * 2] -> [B, hid * 3]
        inv_features = self.fusion(combined)  # [B, output_dim]
        
        return inv_features, inv_class_emb, str_emb  # return features, class embeddings, string embeddings
        # inv_class_emb: [B, 55, CHAR_EMB], str_emb: [B, 55, 80, str_emb]
        
    def teacher_forcing_decorator(self, inv_oclasses: torch.Tensor, inv_strs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inv_oclasses: [B, 55] - object class indices
            inv_strs: [B, 55, 80] - string descriptions (character codes)
        Returns:
            shifted_oclasses: [B, 55] - shifted object class indices with sos/eos
            shifted_strs: [B, 55, 80] - shifted string descriptions with sos/eos
            shifted_oclass_emb: [B, 55, CHAR_EMB] - shifted object class embeddings
            shifted_str_emb: [B, 55, 80, str_emb] - shifted string embeddings
        """
        B = inv_oclasses.size(0)
        # Shift object classes
        shifted_oclasses = torch.zeros_like(inv_oclasses)
        shifted_oclasses[:, 0] = self.sos  # set start-of-sequence token
        shifted_oclasses[:, 1:] = inv_oclasses[:, :-1]  # shift right by 1
        lengths_oclasses = (inv_oclasses != 18).sum(dim=1)  # count non-padding object classes
        bHasSpace = lengths_oclasses < INV_MAX_SIZE - 1  # check if there is space for EOS
        shifted_oclasses[bHasSpace, lengths_oclasses[bHasSpace] + 1] = self.eos  # set end-of-sequence token at the end of each object class sequence
        # Shift string descriptions
        flat_inv_strs = inv_strs.view(B * INV_MAX_SIZE, INV_STR_LEN)  # Flatten to [B*55, 80]
        shifted_strs = flat_inv_strs.clone()
        shifted_strs[:, 0] = self.sos  # set start-of-sequence token
        shifted_strs[:, 1:] = flat_inv_strs[:, :-1]  # shift right by 1
        lengths_strs = (flat_inv_strs != 0).sum(dim=-1)  # [B*55,] - count non-padding characters
        bHasSpace = lengths_strs < INV_STR_LEN - 1  # check if there is space for EOS
        shifted_strs[bHasSpace, lengths_strs[bHasSpace] + 1] = self.eos  # set end-of-sequence token
        # Get embeddings for shifted object classes and strings
        shifted_oclass_emb = super().char_emb(shifted_oclasses)  # [B, 55] -> [B, 55, CHAR_EMB]
        shifted_str_emb = self.char_emb(shifted_strs)  # [B*55, 80] -> [B*55, 80, str_emb]
        shifted_str_emb = shifted_str_emb.view(B, INV_MAX_SIZE, INV_STR_LEN, self.str_emb_dim)
        return shifted_oclasses, shifted_strs, shifted_oclass_emb, shifted_str_emb  # return shifted object classes and strings, and their embeddings
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels."""
        return self.output_dim

class MiniHackVAE(nn.Module):
    def __init__(self, 
                 bInclude_glyph_bag=True, 
                 bInclude_hero=True,
                 bInclude_inventory=True,
                 logger: logging.Logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.glyph_cnn = GlyphCNN()
        self.glyph_bag = GlyphBag(logger=self.logger)  # for bag of glyphs
        self.stats_mlp = StatsMLP()
        self.msg_gru   = MessageGRU()
        self.hero_emb = HeroEmbedding()  # for hero embedding
        self.inv_encoder = InventoryEncoder()  # for inventory encoding
        
        self.include_glyph_bag = bInclude_glyph_bag
        self.include_hero = bInclude_hero
        self.include_inventory = bInclude_inventory

        fusion_in = self.glyph_cnn.get_output_channels() + \
                    self.stats_mlp.get_output_channels() + \
                    self.msg_gru.get_output_channels() + \
                    (self.glyph_bag.get_output_channels() if bInclude_glyph_bag else 0) + \
                    (self.hero_emb.get_output_channels() if bInclude_hero else 0) + \
                    (self.inv_encoder.get_output_channels() if bInclude_inventory else 0)  # cnn + stats + msg + bag of glyphs + hero embedding + inventory
        self.to_latent = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 256), nn.ReLU(),
        )
        self.mu_head     = nn.Linear(256, LATENT_DIM)
        self.logvar_diag_head = nn.Linear(256, LATENT_DIM)  # diagonal part
        self.lowrank_factor_head = nn.Linear(256, LATENT_DIM * LOW_RANK)  # low-rank factors

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
            # Step 1: [B, 64, 1, 1] -> [B, 32, 7, 26] (roughly 1/3 of target)
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=1, padding=0),  # -> [B, 32, 7, 26]
            nn.ReLU(),
            
            # Step 2: [B, 32, 7, 26] -> [B, 20, 21, 79]
            nn.ConvTranspose2d(32, GLYPH_EMB, kernel_size=(15, 54), stride=1, padding=0)  # -> [B, 20, 21, 79]
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
        
        # Inventory embedding decoder
        self.decode_inv_latent2hidden = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.inv_encoder.hid_dim * 3)  # [B, 256] -> [B, hid_dim * 3] (for oclass + str)
        )
        self.decode_inv_class_rnn_emb = nn.RNN(CHAR_EMB, self.inv_encoder.hid_dim, batch_first=True)  # RNN for object class decoding, [B, 55, CHAR_EMB] -> [B, 55, hid_dim]
        self.decode_inv_class_hidden2emb = nn.Linear(self.inv_encoder.hid_dim, self.glyph_cnn.char_emb.embedding_dim)  # [B, 55, hid_dim] -> [B, 55, CHAR_EMB]
        self.decode_inv_str_gru_emb = nn.GRU(self.inv_encoder.str_emb_dim, self.inv_encoder.hid_dim, batch_first=True, bidirectional=True)  # GRU for string decoding, [B*55, 80, emb_dim] -> [B*55, 80, hid_dim * 2]
        self.decode_inv_str_hidden2emb = nn.Linear(self.inv_encoder.hid_dim * 2, self.inv_encoder.str_emb_dim)  # [B*55, 80, hid_dim * 2] -> [B*55, 80, str_emb]
        
        # ------------- Decode logits ----------------
        # These will take the embedding as starting point
        
        # Separate heads for char and color reconstruction
        # these will return logits for each pixel
        self.decode_chars = nn.Conv2d(CHAR_EMB, CHAR_DIM, kernel_size=1)  # [B, 16, 21, 79] -> [B, 256, 21, 79]
        self.decode_colors = nn.Conv2d(COLOR_EMB, COLOR_DIM, kernel_size=1)  # [B, 4, 21, 79] -> [B, 16, 21, 79]
        
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
        
        # messages - use extended vocabulary to include SOS/EOS tokens
        self.decode_msg = nn.Linear(self.msg_gru.emb_dim, self.msg_gru.vocab)  # [B, T, emb_dim] -> [B, T, vocab_size]
        
        self.decode_inv_class = nn.Linear(self.glyph_cnn.char_emb.embedding_dim, INV_OCLASS_DIM)  # [B, 55, CHAR_EMB] -> [B, 55, INV_OCLASS_DIM]
        self.decode_inv_str = nn.Linear(self.inv_encoder.str_emb_dim, self.inv_encoder.vocab)  # [B, 55, 80, str_emb] -> [B, 55, 80, MSG_VOCAB + 2]
        
        # Dynamic prediction heads
        # It takes latent z, action a, HDP HMM state h and hero info c
        # TODO: Implement dynamic prediction heads
        

        

    def _reparameterise(self, mu, logvar, lowrank_factors=None):
        diag_std = torch.exp(0.5*logvar)
        eps1 = torch.randn_like(diag_std)
        z = mu + diag_std*eps1
        if lowrank_factors is not None:
            # If low-rank factors are provided, combine them with the diagonal std
            eps2 = torch.randn_like(lowrank_factors)
            # Assuming lowrank_factors is [B, LATENT_DIM, RANK]
            lowrank_std = torch.bmm(lowrank_factors, eps2.unsqueeze(-1)).squeeze(-1)  # [B, LATENT_DIM]
            z += lowrank_std
        return z    # [B, LATENT_DIM]

    def forward(self, glyph_chars, glyph_colors, blstats, msg_tokens, hero_info=None, inv_oclasses=None, inv_strs=None):
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
            glyph_bag_feat, glyph_bag_emb, glyph_bag = self.glyph_bag(glyph_chars, glyph_colors) # [B,32], [B,max_len,20], [B,max_len, 2]
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
        
        # Inventory encoding (optional)
        if self.include_inventory and inv_oclasses is not None and inv_strs is not None:
            inv_features, inv_oclasses_emb_no_shift, inv_str_emb_no_shift = self.inv_encoder(inv_oclasses, inv_strs)  # [B, output_dim]
            features.append(inv_features)
            inv_oclasses_shift, inv_strs_shift, inv_oclasses_emb_shift, inv_strs_emb_shift = self.inv_encoder.teacher_forcing_decorator(inv_oclasses, inv_strs)  # [B, 55], [B, 55, 80], [B, 55, CHAR_EMB], [B, 55, 80, str_emb]
        else:
            inv_features = None
            inv_oclasses_shift = None
            inv_strs_shift = None
            inv_oclasses_emb_shift = None
            inv_strs_emb_shift = None
        
        fused = torch.cat(features, dim=-1)
        
        # Encode to latent space
        h = self.to_latent(fused)
        mu = self.mu_head(h)
        logvar_diag = self.logvar_diag_head(h)
        lowrank_factors = self.lowrank_factor_head(h).view(B, LATENT_DIM, LOW_RANK)
        z = self._reparameterise(mu, logvar_diag, lowrank_factors)

        # Decode
        shared_features = self.decode_shared(z)  # [B, 256]
        
        # Decode embeddings
        glyph_emb_decoded = self.decode_glyph_emb(shared_features)  # [B, 20, 21, 79]
        stats_emb_decoded = self.decode_stats_emb(shared_features)  # [B, 43]
        
        # Message decoding - autoregressive GRU decoding with proper hidden state
        msg_hidden_init = self.decode_msg_latent2hidden(shared_features)  # [B, 256] -> [B, hid_dim]
        # Use teacher forcing - pass the shifted embeddings through GRU
        msg_lengths = (msg_token_shift != 0).sum(dim=1)  # [B,] - count non-padding tokens
        packed_msg_emb_shift = nn.utils.rnn.pack_padded_sequence(
            msg_emb_shift, lengths=msg_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        # Decode messages using GRU
        packed_msg_hidden_decoded, _ = self.decode_msg_gru(packed_msg_emb_shift, msg_hidden_init.unsqueeze(0))  # hidden: [1, B, hid_dim]
        # Unpack the output
        msg_hidden_decoded, _ = nn.utils.rnn.pad_packed_sequence(packed_msg_hidden_decoded, batch_first=True)
        # Convert hidden states back to embeddings
        msg_emb_decoded = self.decode_msg_hidden2emb(msg_hidden_decoded)  # [B, 256, hid_dim] -> [B, 256, emb_dim]
        
        if self.include_inventory and inv_oclasses is not None and inv_strs is not None:
            # Decode inventory embeddings
            inv_hidden = self.decode_inv_latent2hidden(shared_features)  # [B, 256] -> [B, hid_dim * 3]
            # Decode object classes using RNN
            inv_oclass_lengths = (inv_oclasses_shift != 18).sum(dim=1)  # [B,] - count non-padding object classes
            packed_inv_oclass_emb_shift = nn.utils.rnn.pack_padded_sequence(
                inv_oclasses_emb_shift, lengths=inv_oclass_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            packed_inv_oclass_decoded, _ = self.decode_inv_class_rnn_emb(packed_inv_oclass_emb_shift, inv_hidden[:, :self.inv_encoder.hid_dim].unsqueeze(0))  # hidden: [1, B, hid_dim]
            inv_oclasses_hidden_decoded, _ = nn.utils.rnn.pad_packed_sequence(packed_inv_oclass_decoded, batch_first=True)  # [B, 55, hid_dim]
            inv_oclasses_emb_decoded = self.decode_inv_class_hidden2emb(inv_oclasses_hidden_decoded)  # [B, 55, hid_dim] -> [B, 55, CHAR_EMB]
            
            inv_str_lengths = (inv_strs_shift.view(B * INV_MAX_SIZE, -1) != 0).sum(dim=-1)  # [B*55, ] - count non-padding characters
            packed_inv_str_emb_shift = nn.utils.rnn.pack_padded_sequence(
                inv_strs_emb_shift.view(B * INV_MAX_SIZE, INV_STR_LEN, self.inv_encoder.str_emb_dim), 
                lengths=inv_str_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            packed_inv_str_decoded, _ = self.decode_inv_str_gru_emb(packed_inv_str_emb_shift, inv_hidden[:, self.inv_encoder.hid_dim:].unsqueeze(0))  # hidden: [1, B*55, hid_dim * 2]
            inv_str_hidden_decoded, _ = nn.utils.rnn.pad_packed_sequence(packed_inv_str_decoded, batch_first=True)  # [B*55, 80, hid_dim * 2]
            inv_str_emb_decoded = self.decode_inv_str_hidden2emb(inv_str_hidden_decoded)  # [B*55, 80, hid_dim * 2] -> [B, 55, 80, str_emb]
            inv_str_emb_decoded = inv_str_emb_decoded.view(B, INV_MAX_SIZE, INV_STR_LEN, self.inv_encoder.str_emb_dim)
        else:
            inv_oclasses_emb_decoded = None
            inv_str_emb_decoded = None
            
        
        # logits
        # char logits takes first 16 channels of glyph_emb_decoded
        # color logits takes last 4 channels of glyph_emb_decoded
        char_logits = self.decode_chars(glyph_emb_decoded[:, :CHAR_EMB, :, :])  # [B, 16, 21, 79] -> [B, 256, 21, 79]
        color_logits = self.decode_colors(glyph_emb_decoded[:, CHAR_EMB:, :, :])  # [B, 4, 21, 79] -> [B, 16, 21, 79]
        
        # Decode stats - separate into continuous and discrete components
        stats_continuous_normalized = self.decode_stats_continuous(stats_emb_decoded)  # [B, 19]
        
        # Invert BatchNorm transformation: output = normalized * std + mean
        # Need to access the BatchNorm from the preprocessor to get running statistics
        running_var = self.stats_mlp.preprocessor.batch_norm.running_var
        running_mean = self.stats_mlp.preprocessor.batch_norm.running_mean
        stats_continuous = stats_continuous_normalized * running_var.sqrt() + running_mean
        
        hunger_state_logits = self.decode_hunger_state(stats_emb_decoded)      # [B, 7]
        dungeon_number_logits = self.decode_dungeon_number(stats_emb_decoded)  # [B, 11]
        level_number_logits = self.decode_level_number(stats_emb_decoded)      # [B, 51]
        
        # Message decoding - GRU approach with proper teacher forcing
        msg_logits = self.decode_msg(msg_emb_decoded)  # [B, 256, emb_dim] -> [B, 256, MSG_VOCAB]
        
        if self.include_inventory and inv_oclasses is not None and inv_strs is not None:
            # Decode inventory logits
            inv_oclasses_logits = self.decode_inv_class(inv_oclasses_emb_decoded)  # [B, 55, CHAR_EMB] -> [B, 55, INV_OCLASS_DIM]
            inv_strs_logits = self.decode_inv_str(inv_str_emb_decoded)  # [B, 55, 80, str_emb] -> [B, 55, 80, MSG_VOCAB + 2]
        else:
            inv_oclasses_logits = None
            inv_strs_logits = None

        return {
            'glyph_emb_decoded': glyph_emb_decoded,  # [B, 20, 21, 79]
            'stats_emb_decoded': stats_emb_decoded,  # [B, 43]
            'msg_emb_decoded': msg_emb_decoded,  # [B, 256, emb_dim]
            'inv_oclasses_emb_decoded': inv_oclasses_emb_decoded,  # [B, 55, CHAR_EMB]
            'inv_str_emb_decoded': inv_str_emb_decoded,  # [B, 55, 80, str_emb]
            'char_logits': char_logits, # [B, 256, 21, 79]
            'color_logits': color_logits,  # [B, 16, 21, 79]
            'stats_continuous': stats_continuous,  # [B, 19]
            'hunger_state_logits': hunger_state_logits,  # [B, 7]
            'dungeon_number_logits': dungeon_number_logits,  # [B, 11]
            'level_number_logits': level_number_logits,  # [B, 51]
            'msg_logits': msg_logits, # [B, 256, MSG_VOCAB]
            'inv_oclasses_logits': inv_oclasses_logits,  # [B, 55, INV_OCLASS_DIM]
            'inv_strs_logits': inv_strs_logits,  # [B, 55, 80, MSG_VOCAB + 2]
            'target_char_emb': char_emb,  # [B, 16, H, W]
            'target_color_emb': color_emb,  # [B, 4, H, W]  
            'target_stats_emb': stats_emb,  # [B, emb_dim]
            'target_msg_emb': msg_emb_no_shift,  # [B, T, emb_dim]
            'target_glyph_bag_emb': glyph_bag_emb,  # [B, max_len, 20]
            'target_glyph_bag': glyph_bag,  # [B, max_len, 2] - for glyph bag reconstruction loss
            'target_hero_emb': hero_emb_dict,  # dict of embeddings
            'target_inv_oclassess_emb': inv_oclasses_emb_no_shift,  # [B, 55, CHAR_EMB]
            'target_inv_strs_emb': inv_str_emb_no_shift,  # [B, 55, 80, str_emb]
            'mu': mu, # [B, LATENT_DIM]
            'logvar': logvar_diag, # [B, LATENT_DIM]
            'lowrank_factors': lowrank_factors # [B, LATENT_DIM, LOW_RANK]
        }

# ------------------------- loss helpers ------------------------------ #

def vae_loss(model_output, glyph_chars, glyph_colors, blstats, msg_tokens, hero_info=None,
             inv_oclasses=None, inv_strs=None, weight_emb=1.0, weight_raw=0.1, kl_beta=1.0):
    """VAE loss with separate embedding and raw reconstruction losses."""
    
    # Extract outputs from model
    mu = model_output['mu']
    logvar = model_output['logvar']
    lowrank_factors = model_output['lowrank_factors']
    
    # Raw reconstruction logits
    char_logits = model_output['char_logits']  # [B, 256, 21, 79]
    color_logits = model_output['color_logits']  # [B, 16, 21, 79]
    msg_logits = model_output['msg_logits']  # [B, 256, MSG_VOCAB+2]
    
    # Embedding reconstructions (continuous targets)
    glyph_emb_decoded = model_output['glyph_emb_decoded']  # [B, 20, 21, 79]
    stats_emb_decoded = model_output['stats_emb_decoded']  # [B, 43]
    msg_emb_decoded = model_output['msg_emb_decoded']  # [B, 256, emb_dim]
    
    # ============= Raw Reconstruction Losses =============
    raw_losses = {}
    
    # Glyph reconstruction (chars + colors)
    raw_losses['char'] = F.cross_entropy(char_logits, glyph_chars, reduction='mean')
    raw_losses['color'] = F.cross_entropy(color_logits, glyph_colors, reduction='mean')
    
    # Stats reconstruction - separate losses for continuous and discrete components
    # Extract model outputs
    stats_continuous = model_output['stats_continuous']  # [B, 19] - already inverted to original scale
    hunger_state_logits = model_output['hunger_state_logits']
    dungeon_number_logits = model_output['dungeon_number_logits']
    level_number_logits = model_output['level_number_logits']
    
    # Extract relevant targets (24 dimensions, excluding time and unknowns)
    relevant_indices = [i for i in range(27) if i not in [20, 25, 26]]  # Exclude time and unknowns
    target_stats = blstats[:, relevant_indices]  # [B, 24]
    
    # Separate continuous and discrete targets
    discrete_indices = [21, 23, 24]  # hunger_state, dungeon_number, level_number (in original indexing)
    continuous_indices = [i for i in relevant_indices if i not in discrete_indices]  # 21 indices
    
    # Map discrete indices to the relevant_indices space
    discrete_targets = []
    discrete_targets.append(blstats[:, 21].long())  # hunger_state
    discrete_targets.append(blstats[:, 23].long())  # dungeon_number  
    discrete_targets.append(blstats[:, 24].long())  # level_number
    
    # For continuous targets, we need to prepare them the same way as preprocessor does
    # The decoder outputs stats_continuous on the original scale, so we need to prepare the targets
    continuous_targets_raw = blstats[:, continuous_indices]  # [B, 21]
    
    # Process continuous targets the same way as preprocessor (coordinate normalization, log1p, ratios)
    # but we need to prepare the targets to match what the decoder should output
    continuous_targets_processed = continuous_targets_raw.clone()
    continuous_targets_processed[:, 0] = continuous_targets_processed[:, 0] / MAX_X_COORD  # x coordinate
    continuous_targets_processed[:, 1] = continuous_targets_processed[:, 1] / MAX_Y_COORD  # y coordinate
    
    # Apply log1p to skewed features
    skewed_indices = [9, 13, 19]  # Based on preprocessor
    continuous_targets_processed[:, skewed_indices] = torch.log1p(continuous_targets_processed[:, skewed_indices])
    
    # Create health and energy ratios
    hp_ratio = continuous_targets_processed[:, 10] / torch.clamp(continuous_targets_processed[:, 11], min=1)
    energy_ratio = continuous_targets_processed[:, 14] / torch.clamp(continuous_targets_processed[:, 15], min=1)
    
    # Replace health and energy stats with ratios and remove max health/energy
    continuous_targets_processed[:, 10] = hp_ratio
    continuous_targets_processed[:, 14] = energy_ratio
    
    # Remove max health and max energy columns (indices 11 and 15)
    keep_indices = [i for i in range(continuous_targets_processed.size(1)) if i not in [11, 15]]
    continuous_targets_final = continuous_targets_processed[:, keep_indices]  # [B, 19]
    
    # Calculate losses - stats_continuous should match the processed targets (after inversion)
    raw_losses['stats_continuous'] = F.mse_loss(stats_continuous, continuous_targets_final, reduction='mean')
    raw_losses['stats_hunger'] = F.cross_entropy(hunger_state_logits, discrete_targets[0], reduction='mean')
    raw_losses['stats_dungeon'] = F.cross_entropy(dungeon_number_logits, discrete_targets[1], reduction='mean')
    raw_losses['stats_level'] = F.cross_entropy(level_number_logits, discrete_targets[2], reduction='mean')
    
    # Total stats loss
    raw_losses['stats'] = (raw_losses['stats_continuous'] + 
                          raw_losses['stats_hunger'] + 
                          raw_losses['stats_dungeon'] + 
                          raw_losses['stats_level'])
    
    # Message reconstruction
    msg_lengths = (msg_tokens != 0).sum(dim=1)  # [B,] - count non-padding tokens
    msg_tokens_with_eos = msg_tokens.clone()
    # Add EOS tokens where there's space
    mask = msg_lengths < msg_tokens.size(1)
    msg_tokens_with_eos[mask, msg_lengths[mask]] = MSG_VOCAB + 1  # EOS token
    raw_losses['msg'] = F.cross_entropy(msg_logits.view(-1, msg_logits.size(-1)), msg_tokens_with_eos.view(-1), reduction='mean', ignore_index=0)
    
    # Inventory reconstruction (optional)
    inv_oclasses_logits = model_output['inv_oclasses_logits']
    inv_strs_logits = model_output['inv_strs_logits']
    if inv_oclasses_logits is not None and inv_strs_logits is not None and inv_oclasses is not None and inv_strs is not None:
        # Object class reconstruction
        raw_losses['inv_oclasses'] = F.cross_entropy(
            inv_oclasses_logits.view(-1, INV_OCLASS_DIM), 
            inv_oclasses.view(-1), 
            reduction='mean', 
            ignore_index=18  # Padding index
        )
        
        # String reconstruction
        raw_losses['inv_strs'] = F.cross_entropy(
            inv_strs_logits.view(-1, MSG_VOCAB + 2), 
            inv_strs.view(-1), 
            reduction='mean', 
            ignore_index=0  # Padding index
        )
    
    # ============= Embedding Reconstruction Losses =============
    emb_losses = {}
    
    # Target embeddings (from encoder)
    target_char_emb = model_output['target_char_emb']  # [B, 16, H, W]
    target_color_emb = model_output['target_color_emb']  # [B, 4, H, W]
    target_stats_emb = model_output['target_stats_emb']  # [B, emb_dim] 
    target_msg_emb = model_output['target_msg_emb']  # [B, 256, emb_dim]
    
    # Embedding reconstruction losses
    # Split glyph embeddings back to char/color
    char_emb_recon, color_emb_recon = torch.split(glyph_emb_decoded, [CHAR_EMB, COLOR_EMB], dim=1)  # Split into char and color embeddings
    
    emb_losses['char_emb'] = F.mse_loss(char_emb_recon, target_char_emb.detach(), reduction='mean')
    emb_losses['color_emb'] = F.mse_loss(color_emb_recon, target_color_emb.detach(), reduction='mean')
    
    # Stats embedding loss
    # stats_emb_decoded is [B, 45], target_stats_emb is [B, 45]
    emb_losses['stats_emb'] = F.mse_loss(stats_emb_decoded, target_stats_emb.detach(), reduction='mean')
    
    # Message embedding loss
    emb_losses['msg_emb'] = F.mse_loss(msg_emb_decoded, target_msg_emb.detach(), reduction='mean')
    
    # Inventory embedding loss (optional)
    inv_oclasses_emb_decoded = model_output['inv_oclasses_emb_decoded'] # [B, 55, CHAR_EMB]
    inv_str_emb_decoded = model_output['inv_str_emb_decoded'] # [B, 55, 80, str_emb]
    if inv_oclasses_emb_decoded is not None and inv_str_emb_decoded is not None:
        emb_losses['inv_oclasses_emb'] = F.mse_loss(inv_oclasses_emb_decoded, model_output['target_inv_oclassess_emb'].detach(), reduction='mean')
        emb_losses['inv_strs_emb'] = F.mse_loss(inv_str_emb_decoded, model_output['target_inv_strs_emb'].detach(), reduction='mean')
    
    # ============= Combine Losses =============
    
    # Sum raw reconstruction losses
    total_raw_loss = sum(raw_losses.values())
    
    # Sum embedding losses (when implemented)
    total_emb_loss = sum(emb_losses.values())
    
    # KL divergence
    # Sigma_q = lowrank_factors @ lowrank_factors.T + torch.diag(torch.exp(logvar))
    # KL divergence for low-rank approximation
    if lowrank_factors is not None:
        Sigma_q = torch.bmm(lowrank_factors, lowrank_factors.transpose(1, 2)) + torch.diag_embed(torch.exp(logvar))
    else:
        # If no low-rank factors, just use diagonal covariance
        Sigma_q = torch.diag_embed(torch.exp(logvar))
    # KL divergence: D_KL(q(z|x) || p(z)) =
    # 0.5 * (tr(Sigma_q) + mu^T * mu - d - log(det(Sigma_q)))
    # where d is the dimensionality of the latent space (LATENT_DIM)
    d = mu.size(1)
    tr_Sigma_q = torch.trace(Sigma_q)  # Trace of Sigma_q
    mu2 = (mu.T @ mu).view(-1) # mu^T * mu. [B,]
    log_det_Sigma_q = torch.logdet(Sigma_q)  # log(det(Sigma_q))
    kl_loss = 0.5 * (tr_Sigma_q + mu2 - d - log_det_Sigma_q)
    kl_loss = kl_loss.mean()  # Average over batch
    # Total weighted loss
    total_loss = (weight_raw * total_raw_loss + 
                  weight_emb * total_emb_loss + 
                  kl_beta * kl_loss)
    
    return {
        'total_loss': total_loss,
        'total_raw_loss': total_raw_loss,
        'total_emb_loss': total_emb_loss,
        'kl_loss': kl_loss,
        'raw_losses': raw_losses,
        'emb_losses': emb_losses
    }



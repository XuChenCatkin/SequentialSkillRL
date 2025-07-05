# MiniHack Variational Auto‑Encoder (glyph‐chars + colours + messages)
# ---------------------------------------------------------------------
# Author: Xu Chen
# Date  : 30/06/2025
"""
A light-weight VAE for MiniHack observations.

Inputs (per time-step)
---------------------
* glyph_chars : LongTensor[B,H,W]-ASCII code 0-255 per cell
* glyph_colors: LongTensor[B,H,W]-colour id 0-15  per cell
* blstats     : FloatTensor[B, N_b]-raw scalar stats (hp, gold, …)
* message_txt : LongTensor[B,T_msg]-pre-tokenised message line

Outputs
-------
* recon_glyph_logits   -  pixel-wise categorical reconstruction
* recon_msg_logits     -  token-wise categorical reconstruction
* mu, logvar           -  latent Gaussian params (for KL)

The model is split into four blocks:
    1. GlyphCNN  -> 64-d feature
    2. StatsMLP  -> 32-d feature
    3. MsgGRU    -> 12-d feature
    4. FusionMLP -> mu, logvar  (dim=LATENT_DIM)

Back-prop uses the standard VAE loss (BCE + KL).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# ------------------------- hyper‑params ------------------------------ #
CHAR_DIM = 256      # ASCII code space for characters
CHAR_EMB = 16      # char‑id embedding dim
COLOR_DIM = 16      # colour id space for colours
COLOR_EMB = 4       # colour‑id embedding dim
GLYPH_EMB = CHAR_EMB + COLOR_EMB  # glyph embedding dim
LATENT_DIM = 64     # z‑dim for VAE
BLSTATS_DIM = 27   # raw scalar stats (hp, gold, …)
MSG_VOCAB = 256     # Nethack takes 0-255 byte for char of messages
GLYPH_DIM = CHAR_DIM * COLOR_DIM  # glyph dim for char + color
PADDING_IDX = GLYPH_DIM  # padding index for glyphs
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
    "hunger_state",      # 21: Hunger state (0-5)
    "carrying_capacity", # 22: How much you can carry (0-999)
    "dungeon_number",    # 23: Which dungeon you're in (0-10)
    "level_number",      # 24: Level within the dungeon (0-50)
    "unknown_25",        # 25: Unknown field
    "unknown_26",        # 26: Unknown field (often -1)
]

class GlyphCNN(nn.Module):
    """3-layer conv on [B, C= (char_emb+color_emb), H, W]."""
    def __init__(self):
        super().__init__()
        self.char_emb = nn.Embedding(CHAR_DIM, CHAR_EMB)
        self.col_emb  = nn.Embedding(COLOR_DIM, COLOR_EMB)
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
        self.emb = nn.Embedding(GLYPH_DIM + 1, GLYPH_EMB, padding_idx=PADDING_IDX)  # +1 for padding
        self.net = nn.RNN(GLYPH_EMB, 32, batch_first=True)  # simple RNN for bag encoding
            
        
    def encode_glyphs_to_bag(self, glyph_chars: torch.Tensor, glyph_colors: torch.Tensor) -> torch.Tensor:
        """Encodes glyphs into a bag-of-glyphs representation (sorted)."""
        # Create a bag of glyphs tensor
        B, H, W = glyph_chars.size()
        bag = torch.zeros(B, self.max_len, dtype=torch.long, device=glyph_chars.device)
        
        # Encode each glyph as a unique id
        g_ids = (glyph_chars << 4) + glyph_colors
        g_ids = g_ids.view(B, -1)  # flatten to [B, H*W]
        # for each batch, create a sorted bag of glyphs
        for b in range(B):
            unique_glyphs, counts = torch.unique(g_ids[b], return_counts=True)
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
        return bag  # [B, max_len] of glyph ids

    def forward(self, glyph_chars: torch.Tensor, glyph_colors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode glyphs to bag
        bag = self.encode_glyphs_to_bag(glyph_chars, glyph_colors)
        emb = self.emb(bag)  # [B, max_len] -> [B, max_len, GLYPH_EMB]
        # Use RNN to encode the bag into a fixed-size vector
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths=(bag != PADDING_IDX).sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.net(packed_emb)  # h: [1,B,32]
        features = h.squeeze(0)  # [B,32]
        return features, emb, bag # [B, max_len, GLYPH_EMB], [B, max_len]
        
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
    - blstats[21]: hunger_state (0-5), blstats[23]: dungeon_num, blstats[24]: level_num
    - ignore game time and last 2 stats (blstats[25-26])
    """
    def __init__(self, stats_dim=27):
        super().__init__()
        self.stats_dim = stats_dim
        self.useful_dims = [i for i in range(stats_dim) if i not in [20, 25, 26]]  # Ignore time and last two stats
        # Define discrete stats that need embeddings
        self.discrete_indices = [21, 23, 24]  # hunger_state, dungeon_number, level_number
        self.continuous_indices = [i for i in self.useful_dims if i not in self.discrete_indices] # size 21
        
        # Embedding layers for discrete stats
        self.hunger_emb = nn.Embedding(6, 3)    # Hunger states (0-5)
        self.dungeon_emb = nn.Embedding(11, 4)  # Dungeon numbers (0-10)
        self.level_emb = nn.Embedding(51, 4)    # Level numbers (0-50)
        
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
        hunger_state = torch.clamp(blstats[:, 21].long(), 0, 5)
        dungeon_num = torch.clamp(blstats[:, 23].long(), 0, 10)
        level_num = torch.clamp(blstats[:, 24].long(), 0, 50)
        
        hunger_emb = self.hunger_emb(hunger_state)
        dungeon_emb = self.dungeon_emb(dungeon_num)
        level_emb = self.level_emb(level_num)
        
        # Combine all features
        processed_stats = torch.cat([
            continuous_normalized,  # [B, 19]
            hunger_emb,            # [B, 3]
            dungeon_emb,           # [B, 4]
            level_emb              # [B, 4]
        ], dim=1) # [B, 19 + 3 + 4 + 4] = [B, 30]
        
        return processed_stats
    
    def get_output_dim(self):
        """Returns the output dimension of processed stats"""
        continuous_dim = len(self.continuous_indices) - 2  # -2 for removed max_hp/max_energy
        discrete_dim = 3 + 4 + 4  # hunger + dungeon + level embeddings
        return continuous_dim + discrete_dim  # 19 + 11 = 30

class StatsMLP(nn.Module):
    def __init__(self, stats: int=BLSTATS_DIM, emb_dim: int=64):
        super().__init__()
        self.preprocessor = BlstatsPreprocessor(stats_dim=stats)
        input_dim = self.preprocessor.get_output_dim()  # 30 dimensions after preprocessing
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        
    def forward(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self.preprocessor(stats)  # [B, 27] -> [B, 30]
        features = self.net(processed)  # [B, 30] -> [B, 16]
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
    

class MiniHackVAE(nn.Module):
    def __init__(self, bInclude_glyph_bag=True, bInclude_hero=True, logger: logging.Logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.glyph_cnn = GlyphCNN()
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
                    (self.hero_emb.get_output_channels() if bInclude_hero else 0)  # cnn + stats + msg + bag of glyphs + hero embedding
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
        #    - glyphs_bag (pixel-wise categorical)
        #    - stats (pixel-wise categorical)
        #    - messages (token-wise categorical)
        #    - heros (token-wise categorical)
        # 2. For reconstruction of frozen embeddings (optional):
        #    - glyph_char_embeddings (pixel-wise continuous)
        #    - glyph_color_embeddings (pixel-wise continuous)
        #    - glyph_bag_embeddings (pixel-wise continuous)
        #    - stats_embeddings (pixel-wise continuous)
        #    - messages_embeddings (pixel-wise continuous)
        #    - heros_embeddings (pixel-wise continuous)
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
        # Note: glyph bag reconstruction is derived from char/color logits
        # No separate head needed since glyph bag is just unique combinations of char+color
        
        # stats embeddings
        self.decode_stats_emb = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.stats_mlp.preprocessor.get_output_dim())  # [B, 256] -> [B, 30]
        )
        
        # Message decoder - use unidirectional GRU with proper hidden state size
        self.decode_msg_latent2hidden = nn.Linear(256, self.msg_gru.hid_dim)  # [B, 256] -> [B, hid_dim]
        self.decode_msg_gru = nn.GRU(self.msg_gru.emb_dim, self.msg_gru.hid_dim, batch_first=True)  # [B, T, emb_dim] -> [B, T, hid_dim]
        self.decode_msg_hidden2emb = nn.Linear(self.msg_gru.hid_dim, self.msg_gru.emb_dim)  # [B, T, hid_dim] -> [B, T, emb_dim]
        
        # hero embeddings
        self.decode_role_emb = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(),
            nn.Linear(32, ROLE_EMB)  # [B, emb_dim]
        )
        
        self.decode_race_emb = nn.Sequential(
            nn.Linear(256, 16), nn.ReLU(),
            nn.Linear(16, RACE_EMB)  # [B, emb_dim]
        )
        
        self.decode_gend_emb = nn.Sequential(
            nn.Linear(256, 8), nn.ReLU(),
            nn.Linear(8, GEND_EMB)  # [B, emb_dim]
        )
        
        self.decode_align_emb = nn.Sequential(
            nn.Linear(256, 8), nn.ReLU(),
            nn.Linear(8, ALIGN_EMB)  # [B, emb_dim]
        )
        
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
            nn.Linear(self.stats_mlp.preprocessor.get_output_dim(), 64), nn.ReLU(),  # [B, 30] -> [B, 64]
            nn.Linear(64, 19),  # [B, 64] -> [B, 19] for normalized continuous stats (matches BatchNorm input)
        )
        
        # Discrete stats: hunger_state (0-5), dungeon_number (0-10), level_number (0-50)
        self.decode_hunger_state = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.stats_mlp.preprocessor.get_output_dim(), 32), nn.ReLU(),
            nn.Linear(32, 6),  # [B, 32] -> [B, 6] for hunger states
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
        
        # hero
        self.decode_role = nn.Sequential(
            nn.ReLU(),
            nn.Linear(ROLE_EMB, ROLE_CAD),  # [B, ROLE_EMB] -> [B, ROLE_CAD]
        )
        
        self.decode_race = nn.Sequential(
            nn.ReLU(),
            nn.Linear(RACE_EMB, RACE_CAD), # [B. RACE_EMB] -> [B, RACE_CAD]
        )
        
        self.decode_gend = nn.Sequential(
            nn.ReLU(),
            nn.Linear(GEND_EMB, GEND_CAD),  # [B, GEND_EMB] -> [B, GEND_CAD]
        )
        
        self.decode_align = nn.Sequential(
            nn.ReLU(),
            nn.Linear(ALIGN_EMB, ALIGN_CAD),  # [B, ALIGN_EMB] -> [B, ALIGN_CAD]
        )
        
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

    def forward(self, glyph_chars, glyph_colors, blstats, msg_tokens, hero_info=None):
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
            glyph_bag_feat, glyph_bag_emb, glyph_bag = self.glyph_bag(glyph_chars, glyph_colors) # [B,32], [B,max_len,20], [B,max_len]
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
        lowrank_factors = self.lowrank_factor_head(h).view(B, LATENT_DIM, LOW_RANK)
        z = self._reparameterise(mu, logvar_diag, lowrank_factors)

        # Decode
        shared_features = self.decode_shared(z)  # [B, 256]
        
        # Decode embeddings
        glyph_emb_decoded = self.decode_glyph_emb(shared_features)  # [B, 20, 21, 79]
        stats_emb_decoded = self.decode_stats_emb(shared_features)  # [B, 30]
        
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
        
        if self.include_hero:
            role_emb_decoded = self.decode_role_emb(shared_features) # [B, ROLE_EMB]
            race_emb_decoded = self.decode_race_emb(shared_features) # [B, RACE_EMB]
            gend_emb_decoded = self.decode_gend_emb(shared_features) # [B, GEND_EMB]
            align_emb_decoded = self.decode_align_emb(shared_features) # [B, ALIGN_EMB]
        else:
            role_emb_decoded = None
            race_emb_decoded = None
            gend_emb_decoded = None
            align_emb_decoded = None
        
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
        
        hunger_state_logits = self.decode_hunger_state(stats_emb_decoded)      # [B, 6]
        dungeon_number_logits = self.decode_dungeon_number(stats_emb_decoded)  # [B, 11]
        level_number_logits = self.decode_level_number(stats_emb_decoded)      # [B, 51]
        
        # Message decoding - GRU approach with proper teacher forcing
        msg_logits = self.decode_msg(msg_emb_decoded)  # [B, 256, emb_dim] -> [B, 256, MSG_VOCAB]
        
        # Derive glyph bag logits from char and color logits (no separate head needed)
        if self.include_glyph_bag:
            # We need the target glyph bag to compute logits - extract it from input
            glyph_bag_logits = self._derive_glyph_bag_logits(char_logits, color_logits, glyph_bag)
        else:
            glyph_bag_logits = None
            
        if self.include_hero:
            role_logits = self.decode_role(role_emb_decoded)  # [B, ROLE_EMB] -> [B, ROLE_CAD]
            race_logits = self.decode_race(race_emb_decoded)  # [B, RACE_EMB] -> [B, RACE_CAD]
            gend_logits = self.decode_gend(gend_emb_decoded)  # [B, GEND_EMB] -> [B, GEND_CAD]
            align_logits = self.decode_align(align_emb_decoded)  # [B, ALIGN_EMB] -> [B, ALIGN_CAD]
        else:
            role_logits = None
            race_logits = None
            gend_logits = None
            align_logits = None

        return {
            'glyph_emb_decoded': glyph_emb_decoded,  # [B, 20, 21, 79]
            'stats_emb_decoded': stats_emb_decoded,  # [B, 30]
            'msg_emb_decoded': msg_emb_decoded,  # [B, 256, emb_dim]
            'role_emb_decoded': role_emb_decoded,  # [B, ROLE_EMB]
            'race_emb_decoded': race_emb_decoded,  # [B, RACE_EMB]
            'gend_emb_decoded': gend_emb_decoded,  # [B, GEND_EMB]
            'align_emb_decoded': align_emb_decoded,  # [B, ALIGN_EMB]
            'char_logits': char_logits, # [B, 256, 21, 79]
            'color_logits': color_logits,  # [B, 16, 21, 79]
            'glyph_bag_logits': glyph_bag_logits,  # [B, GLYPH_DIM + 1]
            'stats_continuous': stats_continuous,  # [B, 19]
            'hunger_state_logits': hunger_state_logits,  # [B, 6]
            'dungeon_number_logits': dungeon_number_logits,  # [B, 11]
            'level_number_logits': level_number_logits,  # [B, 51]
            'msg_logits': msg_logits, # [B, 256, MSG_VOCAB]
            'role_logits': role_logits,  # [B, ROLE_CAD]
            'race_logits': race_logits,  # [B, RACE_CAD]
            'gend_logits': gend_logits,  # [B, GEND_CAD]
            'align_logits': align_logits,  # [B, ALIGN_CAD]
            'target_char_emb': char_emb,  # [B, 16, H, W]
            'target_color_emb': color_emb,  # [B, 4, H, W]  
            'target_stats_emb': stats_emb,  # [B, emb_dim]
            'target_msg_emb': msg_emb_no_shift,  # [B, T, emb_dim]
            'target_glyph_bag_emb': glyph_bag_emb if glyph_bag_emb is not None else None,  # [B, max_len, 20]
            'target_glyph_bag': glyph_bag,  # [B, max_len] - for glyph bag reconstruction loss
            'target_hero_emb': hero_emb_dict,  # dict of embeddings
            'mu': mu, # [B, LATENT_DIM]
            'logvar': logvar_diag, # [B, LATENT_DIM]
            'lowrank_factors': lowrank_factors # [B, LATENT_DIM, LOW_RANK]
        }

    def _derive_glyph_bag_logits(self, char_logits, color_logits, target_glyph_bag=None):
        """
        Derive glyph bag logits from char and color logits.
        
        Args:
            char_logits: [B, 256, H, W] - character logits
            color_logits: [B, 16, H, W] - color logits  
            target_glyph_bag: [B, max_len] - target glyph bag for computing unique combinations
            
        Returns:
            glyph_bag_logits: [B, max_len, GLYPH_DIM + 1] - logits for glyph bag reconstruction
        """
        B, _, H, W = char_logits.shape
        
        if target_glyph_bag is None:
            # If no target provided, we can't compute bag logits
            return None
            
        max_len = target_glyph_bag.size(1)
        
        # Get probabilities from logits
        char_probs = F.softmax(char_logits, dim=1)  # [B, 256, H, W]
        color_probs = F.softmax(color_logits, dim=1)  # [B, 16, H, W]
        
        # Compute glyph probabilities: P(glyph_id) = P(char) * P(color)
        # First, reshape to [B, 256, 16, H, W] for outer product
        char_expanded = char_probs.unsqueeze(2)  # [B, 256, 1, H, W]
        color_expanded = color_probs.unsqueeze(1)  # [B, 1, 16, H, W]
        
        # Outer product to get joint probabilities
        glyph_probs = char_expanded * color_expanded  # [B, 256, 16, H, W]
        
        # Reshape to [B, GLYPH_DIM, H, W] where GLYPH_DIM = 256 * 16
        glyph_probs = glyph_probs.view(B, GLYPH_DIM, H, W)
        
        # For each position in the glyph bag, extract the probability of that glyph
        glyph_bag_probs = torch.zeros(B, max_len, GLYPH_DIM + 1, device=char_logits.device)
        
        for b in range(B):
            for pos in range(max_len):
                glyph_id = target_glyph_bag[b, pos].item()
                if glyph_id == PADDING_IDX:
                    # Padding token
                    glyph_bag_probs[b, pos, -1] = 1.0
                else:
                    # Sum probabilities across all spatial locations for this glyph
                    glyph_bag_probs[b, pos, glyph_id] = glyph_probs[b, glyph_id].sum()
        
        # Convert back to logits (add small epsilon to avoid log(0))
        glyph_bag_logits = torch.log(glyph_bag_probs + 1e-8)
        
        return glyph_bag_logits

# ------------------------- loss helpers ------------------------------ #

def vae_loss(model_output, glyph_chars, glyph_colors, blstats, msg_tokens, hero_info=None,
             weight_emb=1.0, weight_raw=0.1, kl_beta=1.0):
    """VAE loss with separate embedding and raw reconstruction losses."""
    
    # Extract outputs from model
    mu = model_output['mu']
    logvar = model_output['logvar']
    
    # Raw reconstruction logits
    char_logits = model_output['char_logits']  # [B, 256, 21, 79]
    color_logits = model_output['color_logits']  # [B, 16, 21, 79]
    stats_logits = model_output['stats_logits']  # [B, BLSTATS_DIM]
    msg_logits = model_output['msg_logits']  # [B, 256, MSG_VOCAB+2]
    
    # Embedding reconstructions (continuous targets)
    glyph_emb_decoded = model_output['glyph_emb_decoded']  # [B, 20, 21, 79]
    stats_emb_decoded = model_output['stats_emb_decoded']  # [B, 30]
    msg_emb_decoded = model_output['msg_emb_decoded']  # [B, 256, emb_dim]
    
    # Optional outputs (may be None)
    glyph_bag_logits = model_output['glyph_bag_logits']
    target_glyph_bag = model_output.get('target_glyph_bag', None)
    role_logits = model_output['role_logits']
    race_logits = model_output['race_logits'] 
    gend_logits = model_output['gend_logits']
    align_logits = model_output['align_logits']
    
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
    
    # Optional reconstructions
    if glyph_bag_logits is not None and target_glyph_bag is not None:
        # Reshape logits for cross entropy: [B, max_len, GLYPH_DIM + 1] -> [B * max_len, GLYPH_DIM + 1]
        glyph_bag_logits_flat = glyph_bag_logits.view(-1, glyph_bag_logits.size(-1))
        target_glyph_bag_flat = target_glyph_bag.view(-1)
        raw_losses['glyph_bag'] = F.cross_entropy(glyph_bag_logits_flat, target_glyph_bag_flat, reduction='mean')
        
    if hero_info is not None and role_logits is not None:
        role, race, gend, align = hero_info[:, 0], hero_info[:, 1], hero_info[:, 2], hero_info[:, 3]
        raw_losses['role'] = F.cross_entropy(role_logits, role, reduction='mean')
        raw_losses['race'] = F.cross_entropy(race_logits, race, reduction='mean') 
        raw_losses['gend'] = F.cross_entropy(gend_logits, gend, reduction='mean')
        raw_losses['align'] = F.cross_entropy(align_logits, align, reduction='mean')
    
    # ============= Embedding Reconstruction Losses =============
    emb_losses = {}
    
    # Target embeddings (from encoder)
    target_char_emb = model_output['target_char_emb']  # [B, 16, H, W]
    target_color_emb = model_output['target_color_emb']  # [B, 4, H, W]
    target_stats_emb = model_output['target_stats_emb']  # [B, emb_dim] 
    target_msg_emb = model_output['target_msg_emb']  # [B, 256, emb_dim]
    
    # Embedding reconstruction losses
    # Split glyph embeddings back to char/color
    char_emb_recon = glyph_emb_decoded[:, :16, :, :]  # First 16 channels [B, 16, H, W]
    color_emb_recon = glyph_emb_decoded[:, 16:, :, :]  # Last 4 channels [B, 4, H, W]
    
    emb_losses['char_emb'] = F.mse_loss(char_emb_recon, target_char_emb, reduction='mean')
    emb_losses['color_emb'] = F.mse_loss(color_emb_recon, target_color_emb, reduction='mean')
    
    # Stats embedding loss
    # stats_emb_decoded is [B, 30], target_stats_emb is [B, 30]
    emb_losses['stats_emb'] = F.mse_loss(stats_emb_decoded, target_stats_emb, reduction='mean')
    
    # Message embedding loss
    emb_losses['msg_emb'] = F.mse_loss(msg_emb_decoded, target_msg_emb, reduction='mean')
    
    # Hero embedding losses
    target_hero_emb = model_output.get('target_hero_emb', None)
    if target_hero_emb is not None:
        role_emb_decoded = model_output.get('role_emb_decoded', None)
        race_emb_decoded = model_output.get('race_emb_decoded', None)
        gend_emb_decoded = model_output.get('gend_emb_decoded', None)
        align_emb_decoded = model_output.get('align_emb_decoded', None)
        
        if role_emb_decoded is not None:
            emb_losses['role_emb'] = F.mse_loss(role_emb_decoded, target_hero_emb['role'], reduction='mean')
        if race_emb_decoded is not None:
            emb_losses['race_emb'] = F.mse_loss(race_emb_decoded, target_hero_emb['race'], reduction='mean')
        if gend_emb_decoded is not None:
            emb_losses['gend_emb'] = F.mse_loss(gend_emb_decoded, target_hero_emb['gend'], reduction='mean')
        if align_emb_decoded is not None:
            emb_losses['align_emb'] = F.mse_loss(align_emb_decoded, target_hero_emb['align'], reduction='mean')
    
    # ============= Combine Losses =============
    
    # Sum raw reconstruction losses
    total_raw_loss = sum(raw_losses.values())
    
    # Sum embedding losses (when implemented)
    total_emb_loss = sum(emb_losses.values())
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total weighted loss
    total_loss = (weight_raw * total_raw_loss + 
                  weight_emb * total_emb_loss + 
                  kl_beta * kl_loss)
    
    return {
        'total_loss': total_loss,
        'raw_loss': total_raw_loss,
        'emb_loss': total_emb_loss,
        'kl_loss': kl_loss,
        'raw_losses': raw_losses,
        'emb_losses': emb_losses
    }



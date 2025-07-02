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

# ------------------------- hyper‑params ------------------------------ #
CHAR_DIM = 256      # ASCII code space for characters
CHAR_EMB = 16      # char‑id embedding dim
COLOR_DIM = 16      # colour id space for colours
COLOR_EMB = 4       # colour‑id embedding dim
GLYPH_EMB = CHAR_EMB + COLOR_EMB  # glyph embedding dim
LATENT_DIM = 64     # z‑dim for VAE
MSG_VOCAB = 256     # byte‑pair vocabulary size for message tokens
GLYPH_DIM = CHAR_DIM * COLOR_DIM  # glyph dim for char + color
PADDING_IDX = GLYPH_DIM  # padding index for glyphs

# ------------------------- 1. Glyph encoder -------------------------- #
class GlyphCNN(nn.Module):
    """3-layer conv on [B, C= (char_emb+color_emb), H, W]."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(GLYPH_EMB, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # -> [B,64,1,1]
        )

    def forward(self, glyph_emb: torch.Tensor) -> torch.Tensor:  # [B,C,H,W]
        x = self.net(glyph_emb)
        return x.view(x.size(0), -1)  # [B,64]
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return 64

# ------------------------- 2. Stats MLP ------------------------------ #
class StatsMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats)  # [B,32]
    
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.net."""
        return 32

# ------------------------- 3. Message GRU ---------------------------- #
class MessageGRU(nn.Module):
    def __init__(self, vocab: int=MSG_VOCAB, emb: int=8, hid: int=12):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
    def forward(self, msg_tokens: torch.Tensor) -> torch.Tensor:
        # msg_tokens: [B,T]
        emb = self.emb(msg_tokens)
        _, h = self.gru(emb)      # h: [1,B,12]
        return h.squeeze(0)       # [B,12]
    def get_output_channels(self) -> int:
        """Returns the number of output channels after self.gru."""
        return 12

# ------------------------- 4. Fusion / VAE --------------------------- #
class MiniHackVAE(nn.Module):
    def __init__(self, stats_dim: int):
        super().__init__()
        # embeddings for glyph char + colour
        self.char_emb = nn.Embedding(CHAR_DIM,  CHAR_EMB)
        self.col_emb  = nn.Embedding(COLOR_DIM, COLOR_EMB)
        self.glyph_emb = nn.Embedding(GLYPH_DIM + 1, GLYPH_EMB, padding_idx=PADDING_IDX)  # +1 for padding

        self.glyph_cnn = GlyphCNN()
        self.stats_mlp = StatsMLP(stats_dim)
        self.msg_gru   = MessageGRU()

        fusion_in = self.glyph_cnn.get_output_channels() + \
                    self.stats_mlp.get_output_channels() + \
                    self.msg_gru.get_output_channels() + \
                    GLYPH_EMB # cnn + stats + msg + bag of glyphs
        self.to_latent = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 256), nn.ReLU(),
        )
        self.mu_head     = nn.Linear(256, LATENT_DIM)
        self.logvar_head = nn.Linear(256, LATENT_DIM)

        # simple Gaussian decoder for illustration
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 64*77*79 + MSG_VOCAB*200)  # flattened outputs
        )

    # --------------- helpers --------------- #
    def _reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps
    
    def _encode_glyphs_to_bag(self, glyph_chars, glyph_colors, max_len=64):
        """Encodes glyphs into a bag-of-glyphs representation."""
        # Create a bag of glyphs tensor
        B, H, W = glyph_chars.size()
        bag = torch.zeros(B, max_len, dtype=torch.long, device=glyph_chars.device)
        
        # Encode each glyph as a unique id
        g_ids = glyph_chars << 4 + glyph_colors

    # --------------- forward --------------- #
    def forward(self, glyph_chars, glyph_colors, blstats, msg_tokens):
        # glyphs
        g_emb = torch.cat([
            self.char_emb(glyph_chars),
            self.col_emb(glyph_colors)
        ], dim=1)                # [B, C=20, H, W]
        glyph_feat = self.glyph_cnn(g_emb)
        
        g_ids = glyph_chars << 4 + glyph_colors  # [B, H, W]
        

        stats_feat = self.stats_mlp(blstats)
        msg_feat   = self.msg_gru(msg_tokens)

        fused = torch.cat([glyph_feat, stats_feat, msg_feat], dim=-1)
        h = self.to_latent(fused)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self._reparameterise(mu, logvar)

        recon_flat = self.dec(z)
        return recon_flat, mu, logvar

# ------------------------- loss helpers ------------------------------ #

def vae_loss(recon_flat, glyph_chars, msg_tokens, mu, logvar):
    """Simple BCE + KL for demonstration. Split recon tensor back."""
    B = glyph_chars.size(0)
    glyph_logits = recon_flat[:, :64*77*79].view(B, 64, 77, 79)
    msg_logits   = recon_flat[:, 64*77*79:].view(B, 200, MSG_VOCAB)

    recon_glyph_loss = F.cross_entropy(
        glyph_logits, glyph_chars, reduction='mean')
    recon_msg_loss = F.cross_entropy(
        msg_logits.permute(0,2,1), msg_tokens, reduction='mean')

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_glyph_loss + recon_msg_loss + kl

"""
Complete VAE training pipeline for NetHack Learning Dataset
Supports both the simple NetHackVAE and the sophisticated MiniHackVAE from src/model.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import re  # Add regex import for status line parsing
warnings.filterwarnings('ignore')

# Add utils to path for importing env_utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import env_utils

# Add src to path for importing the existing MultiModalHackVAE
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import MultiModalHackVAE, vae_loss
import torch.optim as optim
import sqlite3
import random
from pathlib import Path
import nle.dataset as nld
from nle.nethack import tty_render
import matplotlib.pyplot as plt
import time

# Import our utility functions
import sys
sys.path.append('./utils')
from env_utils import separate_tty_components, get_current_message, get_status_lines, get_game_map


class NetHackDataCollector:
    """Collects and preprocesses NetHack data for VAE training"""
    
    def __init__(self, dbfilename: str = "ttyrecs.db"):
        self.dbfilename = dbfilename
        self.collected_data = []
        self.game_hero_info = {} # game_id -> hero_info tensor [4,]
        
        # NetHack mappings for hero info parsing
        self.role_mapping = {
            'archaeologist': 0, 'barbarian': 1, 'caveman': 2, 'cavewoman': 2,
            'healer': 3, 'knight': 4, 'monk': 5, 'priest': 6, 'priestess': 6,
            'ranger': 7, 'rogue': 8, 'samurai': 9, 'tourist': 10, 'valkyrie': 11,
            'wizard': 12
        }
        
        self.race_mapping = {
            'human': 0, 
            'humanity': 0,  # 'humanity' is a common term in NetHack
            'hum': 0,
            'elf': 1, 
            'elven': 1,
            'elvenkind': 1,
            'dwarf': 2, 
            'dwarven': 2,
            'dwarvenkind': 2,
            'dwa': 2,
            'gnome': 3, 
            'gnomish': 3,
            'gnomehood': 3,
            'gno': 3,
            'orc': 4,
            'orcish': 4,
            'orcdom': 4
        }
        
        self.gender_mapping = {
            'male': 0, 'female': 1, 'neuter': 2  # Note: 0=male, 1=female, 2=neuter in NetHack
        }
        
        self.alignment_mapping = {
            'lawful': 1, 'neutral': 0, 'chaotic': -1
        }
        
    def get_hero_info(self, game_id: int, message: str) -> Optional[torch.Tensor]:
        """
        Get hero info tensor for a specific game ID.
        
        Args:
            game_id: Unique identifier for the game
            message: Welcome message from the game

        Returns:
            Hero info tensor [4,] with [role, race, gender, alignment] or None if not found
        """
        hero_info = self.game_hero_info.get(game_id)
        if hero_info is not None:
            return hero_info
        # If not found, try to parse from message
        return self.parse_hero_info_from_message(game_id, message)

    def parse_hero_info_from_message(self, game_id: int, message: str) -> Optional[torch.Tensor]:
        """
        Parse hero information from NetHack welcome message and return tensor, also cache it into dict.
        
        Args:
            game_id: Unique game identifier
            message: Welcome message like "Hello Agent, welcome to NetHack! You are a neutral female human Monk"
            
        Returns:
            Hero info tensor [4,] with [role, race, gender, alignment] or None if parsing fails
        """
        if not message or 'welcome to nethack' not in message.lower():
            return None
        
        import re
        message_lower = message.lower()
        words = re.findall(r'\b\w+\b', message_lower)  # Extract word characters only
        
        # Extract alignment
        alignment = None
        for align_name, align_value in self.alignment_mapping.items():
            if align_name in words:
                alignment = align_value
                break
        
        # Extract gender  
        gender = None
        for gender_name, gender_value in self.gender_mapping.items():
            if gender_name in words:
                gender = gender_value
                break
        
        # Extract race
        race = None
        for race_name, race_value in self.race_mapping.items():
            if race_name in words:
                race = race_value
                break
        
        # Extract role
        role = None
        for role_name, role_value in self.role_mapping.items():
            if role_name in words:
                role = role_value
                if 'man' or 'priest' in role_name and not gender:
                    gender = self.gender_mapping['male']
                elif 'woman' or 'priestess' in role_name and not gender:
                    gender = self.gender_mapping['female']
                elif not gender:
                    gender = self.gender_mapping['neuter']  # Default to neuter if not specified
                break
        
        # Only return if we found all required components
        if all(x is not None for x in [alignment, gender, race, role]):
            hero_info = torch.tensor([role, race, gender, alignment], dtype=torch.int8)
            self.game_hero_info[game_id] = hero_info
            return hero_info
        
        return None
    
    def collect_data_batch(self, dataset_name: str, adapter: callable, max_batches: int = 1000, 
                          batch_size: int = 32, seq_length: int = 32) -> List[Dict]:
        """
        Collect data from the dataset for VAE training
        
        Args:
            dataset_name: Name of dataset in database
            max_samples: Maximum number of samples to collect
            batch_size: Batch size for dataset loader
            seq_length: Sequence length for dataset loader
            
        Returns:
            List of processed data samples
        """
        print(f"Collecting {max_batches} batches from {dataset_name} for VAE training...")
        print(f"  - Batch size: {batch_size}, Sequence length: {seq_length}")
        
        # Create dataset loader
        dataset = nld.TtyrecDataset(
            dataset_name,
            batch_size=batch_size,
            seq_length=seq_length,
            dbfilename=self.dbfilename,
            shuffle=True
        )
        
        collected_batches = []
        batch_count = 0
        
        for batch_idx, minibatch in enumerate(dataset):
            if batch_count >= max_batches:
                break
                
            print(f"Processing batch {batch_idx + 1}...")
            
            num_games, num_time = minibatch['tty_chars'].shape[:2]
            message_chars_minibatch = torch.empty((num_games, num_time, 80), dtype=torch.int8)
            game_chars_minibatch = torch.empty((num_games, num_time, 21, 79), dtype=torch.int8)
            game_colors_minibatch = torch.empty((num_games, num_time, 21, 79), dtype=torch.int8)
            status_chars_minibatch = torch.empty((num_games, num_time, 2, 80), dtype=torch.int8)
            hero_info_minibatch = torch.empty((num_games, num_time, 4), dtype=torch.int8)
            blstats_minibatch = torch.empty((num_games, num_time, 27), dtype=torch.float8)

            # Process each game in the batch
            for game_idx in range(minibatch['tty_chars'].shape[0]):
                # Process each timestep
                for time_idx in range(minibatch['tty_chars'].shape[1]):
                    
                    game_id = int(minibatch['gameids'][game_idx, time_idx])
                    timestamp = float(minibatch['timestamps'][game_idx, time_idx])
                    score = float(minibatch['scores'][game_idx, time_idx])
                    
                    # Get TTY data
                    tty_chars = minibatch['tty_chars'][game_idx, time_idx]
                    tty_colors = minibatch['tty_colors'][game_idx, time_idx]
                    
                    # Separate TTY components using our utility function
                    tty_components = separate_tty_components(tty_chars, tty_colors)
                    
                    # Extract text information
                    current_message = get_current_message(tty_chars)
                    status_lines = get_status_lines(tty_chars)
                    
                    hero_info = self.get_hero_info(game_id, current_message)
                    if hero_info is None:
                        raise Exception(f"⚠️ No hero info found for game {game_id}, skipping sample")

                    # Fill in the minibatch tensors
                    message_chars_minibatch[game_idx, time_idx] = tty_components['message_chars']
                    game_chars_minibatch[game_idx, time_idx] = tty_components['game_chars']
                    game_colors_minibatch[game_idx, time_idx] = tty_components['game_colors']
                    status_chars_minibatch[game_idx, time_idx] = tty_components['status_chars']
                    hero_info_minibatch[game_idx, time_idx] = hero_info
                    blstats_minibatch[game_idx, time_idx] = adapter(tty_components['game_chars'], status_lines, score, timestamp)

            minibatch['message_chars'] = message_chars_minibatch
            minibatch['game_chars'] = game_chars_minibatch
            minibatch['game_colors'] = game_colors_minibatch
            minibatch['status_chars'] = status_chars_minibatch
            minibatch['hero_info'] = hero_info_minibatch
            minibatch['blstats'] = blstats_minibatch
            collected_batches.append(minibatch)

        print(f"✅ Successfully collected {len(collected_batches)} batches from {batch_count} processed batches")
        return collected_batches
class BLStatsAdapter:
    """
    Converts TTY data to MultiModalHackVAE expected format

    TTY data format (from NetHack Learning Dataset):
    - chars: (24, 80) array of ASCII characters
    - colors: (24, 80) array of color indices
    - cursor: (y, x) position

    MultiModalHackVAE expected format:
    - glyph_chars: LongTensor[B,H,W] - ASCII character codes (32-127) per map cell
    - glyph_colors: LongTensor[B,H,W] - color indices (0-15) per map cell  
    - blstats: FloatTensor[B,27] - game statistics (hp, gold, position, etc.)
    - msg_tokens: LongTensor[B,256] - tokenized message text (0-127 + SOS/EOS)
    - hero_info: optional dict with role, race, gender, alignment
    """
    
    def __init__(self):
        # NetHack map region is typically at rows 1-21 (top row is messages, bottom rows are status)
        self.map_start_row = 1
        self.map_end_row = 22  # 21 rows of map
        self.map_height = 21
        self.map_width = 79
        
        self.hunger_mapping = {
            None: 1,          # NOT_HUNGRY (normal state when nothing shown)
            'satiated': 0,    # SATIATED
            'hungry': 2,      # HUNGRY
            'weak': 3,        # WEAK
            'fainting': 4,    # FAINTING
            'fainted': 5,     # FAINTED
            'starved': 6      # STARVED
        }
        
        # Condition bit mapping from nle/include/botl.h
        # These are the exact masks and display names from NetHack source
        self.condition_bits = {
            'stone': 0x00000001,    # BL_MASK_STONE
            'slime': 0x00000002,    # BL_MASK_SLIME
            'strngl': 0x00000004,   # BL_MASK_STRNGL (strangling)
            'foodpois': 0x00000008, # BL_MASK_FOODPOIS
            'termill': 0x00000010,  # BL_MASK_TERMILL (terminal illness)
            'blind': 0x00000020,    # BL_MASK_BLIND
            'deaf': 0x00000040,     # BL_MASK_DEAF
            'stun': 0x00000080,     # BL_MASK_STUN
            'conf': 0x00000100,     # BL_MASK_CONF (confused)
            'hallu': 0x00000200,    # BL_MASK_HALLU (hallucinating)
            'lev': 0x00000400,      # BL_MASK_LEV (levitating)
            'fly': 0x00000800,      # BL_MASK_FLY (flying)
            'ride': 0x00001000      # BL_MASK_RIDE (riding)
        }
        
        # Alternative names that might appear in status lines
        self.condition_aliases = {
            'confused': 'conf',
            'stunned': 'stun',
            'hallucinating': 'hallu',
            'levitating': 'lev',
            'flying': 'fly',
            'strangling': 'strngl',
            'poisoned': 'foodpois',
            'sick': 'foodpois',  # Food poisoning often shows as "sick"
            'ill': 'termill',
            'riding': 'ride'
        }
        
        # Capacity/encumbrance mapping from nle/src/botl.c
        # enc_stat array: ["", "Burdened", "Stressed", "Strained", "Overtaxed", "Overloaded"]
        self.capacity_mapping = {
            None: 0,          # UNENCUMBERED (no encumbrance shown)
            'burdened': 1,    # SLT_ENCUMBER 
            'stressed': 2,    # MOD_ENCUMBER
            'strained': 3,    # HVY_ENCUMBER
            'overtaxed': 4,   # EXT_ENCUMBER
            'overloaded': 5   # OVERLOADED
        }
    
    def _find_player_position(self, chars: np.ndarray) -> Tuple[int, int]:
        """
        Find player position by locating '@' symbol on the game map
        
        Args:
            chars: TTY characters array (24, 80)
            
        Returns:
            Tuple of (x, y) coordinates, or cursor position if '@' not found
        """
        # Search for '@' symbol (ASCII 64) in the map region (rows 1-21)
        for y in range(self.map_start_row, self.map_end_row):
            for x in range(self.map_width):
                if chars[y, x] == ord('@'):
                    # Return 0-indexed coordinates relative to map
                    map_x = x
                    map_y = y - self.map_start_row  # Adjust for map offset
                    return (map_x, map_y)
        
        # If '@' not found, return center of map as fallback
        return (self.map_width // 2, self.map_height // 2)
    
    def _parse_status_comprehensive(self, status_lines: List[str]) -> Dict:
        """Comprehensive status line parsing with multiple format support"""
        if not status_lines:
            return {}
        
        full_status = ' '.join(status_lines).lower()
        stats = {}
        
        # Parse ability scores from first status line (Line 22)
        # Format: "Agent the Digger               St:15 Dx:9 Co:7 In:13 Wi:18 Ch:12 Neutral S:0"
        ability_patterns = {
            'strength': r'st[:\s]*(\d+)',
            'dexterity': r'dx[:\s]*(\d+)',
            'constitution': r'co[:\s]*(\d+)',
            'intelligence': r'in[:\s]*(\d+)',
            'wisdom': r'wi[:\s]*(\d+)',
            'charisma': r'ch[:\s]*(\d+)',
        }
        
        for ability, pattern in ability_patterns.items():
            match = re.search(pattern, full_status)
            if match:
                stats[ability] = int(match.group(1))
        
        # Parse score from first status line
        score_patterns = [
            r's[:\s]*(\d+)',  # S:0
            r'score[:\s]*(\d+)',  # Score:150
        ]
        for pattern in score_patterns:
            match = re.search(pattern, full_status)
            if match:
                stats['status_score'] = int(match.group(1))
                break
        
        # HP parsing with multiple formats (usually in second status line)
        hp_patterns = [
            r'hp[:\s]*(\d+)\((\d+)\)',      # HP:15(20)
            r'(\d+)\((\d+)\)\s*hp',         # 15(20)Hp
            r'hitpoints[:\s]*(\d+)/(\d+)',  # Hitpoints:15/20
        ]
        for pattern in hp_patterns:
            match = re.search(pattern, full_status)
            if match:
                stats['hp'] = int(match.group(1))
                stats['max_hp'] = int(match.group(2))
                break
        
        # Power/Energy parsing
        pw_patterns = [
            r'pw[:\s]*(\d+)\((\d+)\)',     # Pw:5(5)
            r'(\d+)\((\d+)\)\s*pw',        # 5(5)Pw
            r'power[:\s]*(\d+)/(\d+)',     # Power:5/5
        ]
        for pattern in pw_patterns:
            match = re.search(pattern, full_status)
            if match:
                stats['power'] = int(match.group(1))
                stats['max_power'] = int(match.group(2))
                break
        
        # Other key stats with robust pattern matching
        patterns = {
            'ac': r'ac[:\s]*(-?\d+)',                    # AC:2 or AC -2
            'exp_level': r'(?:xp|exp)[:\s]*(\d+)',       # Xp:3 or Exp:3
            'exp_points': r'(?:xp|exp)[:\s]*\d+/(\d+)',  # Xp:3/45
            'gold': r'(?:\$|au)[:\s]*(\d+)',             # $:45 or Au:45
            'time': r'\bt[:\s]*(\d+)',                     # T:1234 (word boundary to avoid St:15)
            'dungeon_level': r'dlvl[:\s]*(\d+)',         # Dlvl:1
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, full_status)
            if match:
                stats[key] = int(match.group(1))
        
        # Only set if explicitly shown, otherwise defaults to NOT_HUNGRY (1)
        hunger_states = ['satiated', 'hungry', 'weak', 'fainting', 'fainted', 'starved']
        stats['hunger_state'] = None  # Will map to NOT_HUNGRY (1)
        for hunger in hunger_states:
            if hunger in full_status:
                stats['hunger_state'] = hunger
                break
        
        # Capacity/encumbrance parsing
        # Only appears when encumbered (capacity > UNENCUMBERED)
        capacity_states = ['burdened', 'stressed', 'strained', 'overtaxed', 'overloaded']
        stats['capacity_state'] = None  # Will map to UNENCUMBERED (0)
        for capacity in capacity_states:
            if capacity in full_status:
                stats['capacity_state'] = capacity
                break

        # Monster level parsing (when polymorphed)
        # Format: "HD:n" where n is the monster level
        # Only appears when Upolyd (polymorphed) is true
        hd_pattern = r'hd[:\s]*(\d+)'
        match = re.search(hd_pattern, full_status)
        if match:
            stats['monster_level'] = int(match.group(1))
        else:
            stats['monster_level'] = 0  # 0 when not polymorphed
        
        # Alignment parsing from first status line
        # Format: "Agent the Digger               St:15 Dx:9 Co:7 In:13 Wi:18 Ch:12 Neutral S:0"
        # Can be "Lawful", "Neutral", or "Chaotic"
        alignment_patterns = [
            r'\b(lawful)\b',    # Lawful = 1
            r'\b(neutral)\b',   # Neutral = 0  
            r'\b(chaotic)\b'    # Chaotic = -1
        ]
        
        stats['alignment'] = None  # Will default to neutral (0)
        for pattern in alignment_patterns:
            match = re.search(pattern, full_status)
            if match:
                alignment_name = match.group(1).lower()
                if alignment_name == 'lawful':
                    stats['alignment'] = 1
                elif alignment_name == 'neutral':
                    stats['alignment'] = 0
                elif alignment_name == 'chaotic':
                    stats['alignment'] = -1
                break
        
        detected_conditions = []
        
        # Check for exact NetHack condition names
        for condition in self.condition_bits.keys():
            if condition in full_status:
                detected_conditions.append(condition)
        
        # Check for alternative names
        for alias, canonical in self.condition_aliases.items():
            if alias in full_status and canonical not in detected_conditions:
                detected_conditions.append(canonical)
        
        stats['conditions'] = detected_conditions
        
        return stats
    
    def __call__(self, game_chars: np.ndarray, status_lines: List[str], score: float = 0.0, timestamp: float = 0.0) -> np.ndarray:
        """
        Create accurate 27-dimensional blstats from TTY data
        
        1. Player position found by locating '@' symbol on map (not from status line)
        2. When hunger not shown -> assume NOT_HUNGRY state (hunger_state=1)
        3. When no conditions shown -> assume no conditions (condition_mask=0)
        4. Use reasonable defaults for stats not available in status line
        """
        # Parse status line first
        parsed_stats = self._parse_status_comprehensive(status_lines)
        
        # Create 27-dimensional blstats array
        blstats = np.zeros(27, dtype=np.int64)
        
        # Position (0, 1) - Find player '@' on map, not from status line
        player_x, player_y = self._find_player_position(game_chars)
        blstats[0] = player_x  # NLE_BL_X - x coordinate
        blstats[1] = player_y  # NLE_BL_Y - y coordinate
        
        # Use exact NLE blstats mapping from nle/include/nletypes.h
        blstats[2] = parsed_stats.get('strength', 16)     # NLE_BL_STR25 - strength 3..25
        blstats[3] = blstats[2]  # NLE_BL_STR125 - strength 3..125 (not available from tty_chars)
        blstats[4] = parsed_stats.get('dexterity', 16)    # NLE_BL_DEX - dexterity
        blstats[5] = parsed_stats.get('constitution', 16) # NLE_BL_CON - constitution
        blstats[6] = parsed_stats.get('intelligence', 16) # NLE_BL_INT - intelligence
        blstats[7] = parsed_stats.get('wisdom', 16)       # NLE_BL_WIS - wisdom
        blstats[8] = parsed_stats.get('charisma', 16)     # NLE_BL_CHA - charisma
        
        # Score (9) - NLE_BL_SCORE
        status_score = parsed_stats.get('status_score')
        if status_score is not None:
            blstats[9] = status_score
        else:
            blstats[9] = int(score)
        
        # HP (10, 11) - NLE_BL_HP, NLE_BL_HPMAX
        blstats[10] = parsed_stats.get('hp', 15)      # current hp
        blstats[11] = parsed_stats.get('max_hp', 15)  # max hp
        
        # Depth (12) - NLE_BL_DEPTH - dungeon depth level, not available in TTY chars
        blstats[12] = parsed_stats.get('dungeon_level', 1)
        
        # Gold (13) - NLE_BL_GOLD
        blstats[13] = parsed_stats.get('gold', 0)
        
        # Power/Energy (14, 15) - NLE_BL_ENE, NLE_BL_ENEMAX
        blstats[14] = parsed_stats.get('power', 5)    # current power
        blstats[15] = parsed_stats.get('max_power', 5) # max power
        
        # AC (16) - NLE_BL_AC - armor class
        blstats[16] = parsed_stats.get('ac', 10)
        
        # Monster level (17) - NLE_BL_HD - hit dice when polymorphed
        blstats[17] = parsed_stats.get('monster_level', 0)
        
        # Experience level (18) - NLE_BL_XP
        exp_level = parsed_stats.get('exp_level', 1)
        blstats[18] = exp_level
        
        # Experience points (19) - NLE_BL_EXP
        blstats[19] = parsed_stats.get('exp_points', 0)
        
        # Time (20) - NLE_BL_TIME
        blstats[20] = int(parsed_stats.get('time', timestamp))
        
        # Hunger state (21) - NLE_BL_HUNGER
        hunger_state = parsed_stats.get('hunger_state')
        blstats[21] = self.hunger_mapping.get(hunger_state, 1)  # NOT_HUNGRY
        
        # Capacity/encumbrance (22) - NLE_BL_CAP - carrying capacity
        capacity_state = parsed_stats.get('capacity_state')
        blstats[22] = self.capacity_mapping.get(capacity_state, 0)  # UNENCUMBERED
        
        # Dungeon number (23) - NLE_BL_DNUM, not available in TTY chars
        blstats[23] = 0  # Default to main dungeon
        
        # Dungeon level (24) - NLE_BL_DLEVEL
        blstats[24] = parsed_stats.get('dungeon_level', 1)
        
        # Condition mask (25) - NLE_BL_CONDITION
        condition_mask = 0
        for condition in parsed_stats.get('conditions', []):
            if condition in self.condition_bits:
                condition_mask |= self.condition_bits[condition]
        blstats[25] = condition_mask
        
        # Alignment (26) - NLE_BL_ALIGN
        alignment = parsed_stats.get('alignment')
        if alignment is not None:
            blstats[26] = alignment  # 1=lawful, 0=neutral, -1=chaotic
        else:
            blstats[26] = 0  # Default to neutral when not specified
        
        return blstats


class NetHackVAE(nn.Module):
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512):
        super(NetHackVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (2, 24, 80)
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),  # (32, 12, 40)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64, 6, 20)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (128, 3, 10)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (256, 3, 10)
            nn.ReLU(),
        )
        
        # Calculate flattened size after convolutions
        self.encoder_output_size = 256 * 3 * 10  # 7680
        
        # Latent space
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.encoder_output_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), # (128, 3, 10)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 6, 20)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (32, 12, 40)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),    # (2, 24, 80)
        )
    
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 3, 10)  # Reshape for conv transpose
        h = self.decoder(h)
        return h
    
    def forward(self, x):
        """Forward pass through VAE"""
        # Normalize input to [0, 1] range
        x_chars = x[:, 0:1] / 255.0  # Normalize chars
        x_colors = x[:, 1:2] / 15.0  # Normalize colors (assuming max 15)
        x_norm = torch.cat([x_chars, x_colors], dim=1)
        
        # Encode
        mu, logvar = self.encode(x_norm)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, mu, logvar, z


def vae_loss_function(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    VAE loss function: Reconstruction + KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input  
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
    """
    # Normalize inputs for comparison
    x_chars = x[:, 0:1] / 255.0
    x_colors = x[:, 1:2] / 15.0
    x_norm = torch.cat([x_chars, x_colors], dim=1)
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x_norm, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_vae_epoch(model, train_loader, optimizer, device, beta=1.0):
    """Train VAE for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Get data
        data = batch['input'].to(device)  # (batch_size, 2, 24, 80)
        
        # Forward pass
        recon_data, mu, logvar, z = model(data)
        
        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss_function(recon_data, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    num_batches = len(train_loader)
    return (total_loss / num_batches, 
            total_recon_loss / num_batches, 
            total_kl_loss / num_batches)


def evaluate_vae(model, test_loader, device, beta=1.0):
    """Evaluate VAE on test set"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch['input'].to(device)
            
            # Forward pass
            recon_data, mu, logvar, z = model(data)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss_function(recon_data, data, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    num_batches = len(test_loader)
    return (total_loss / num_batches,
            total_recon_loss / num_batches,
            total_kl_loss / num_batches)


def train_vae(train_samples, test_samples, 
              latent_dim=128, 
              batch_size=32, 
              num_epochs=50, 
              learning_rate=1e-3,
              beta=1.0,
              save_path="nethack_vae.pth"):
    """
    Complete VAE training pipeline
    
    Args:
        train_samples: Training data samples
        test_samples: Test data samples
        latent_dim: Dimension of latent space
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        beta: Weight for KL divergence in loss
        save_path: Path to save trained model
    """
    print("=" * 60)
    print("TRAINING VAE ON NETHACK DATA")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = NetHackDataset(train_samples, target_type='reconstruction')
    test_dataset = NetHackDataset(test_samples, target_type='reconstruction')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Training Setup:")
    print(f"  - Training samples: {len(train_samples)}")
    print(f"  - Test samples: {len(test_samples)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Beta (KL weight): {beta}")
    
    # Initialize model
    model = NetHackVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    test_losses = []
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_recon, train_kl = train_vae_epoch(
            model, train_loader, optimizer, device, beta
        )
        
        # Evaluate on test set
        test_loss, test_recon, test_kl = evaluate_vae(
            model, test_loader, device, beta
        )
        
        # Record losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {train_loss:.2f} (Recon: {train_recon:.2f}, KL: {train_kl:.2f}) | "
                  f"Test Loss: {test_loss:.2f} (Recon: {test_recon:.2f}, KL: {test_kl:.2f}) | "
                  f"Time: {epoch_time:.1f}s")
    
    # Save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'latent_dim': latent_dim,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
    }, save_path)
    
    print(f"\n✅ Training completed!")
    print(f"  - Final train loss: {train_losses[-1]:.2f}")
    print(f"  - Final test loss: {test_losses[-1]:.2f}")
    print(f"  - Model saved to: {save_path}")
    
    return model, train_losses, test_losses


def train_multimodalhack_vae(train_samples: List[Dict], test_samples: List[Dict], 
                      epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-3,
                      latent_dim: int = 64, save_path: str = "models/minihack_vae.pth",
                      device: str = None, include_inventory: bool = False) -> Tuple[MultiModalHackVAE, List[float], List[float]]:
    """
    Train MultiModalHackVAE on NetHack Learning Dataset

    Args:
        train_samples: List of training samples (TTY format)
        test_samples: List of testing samples (TTY format)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        latent_dim: Latent dimension for VAE
        save_path: Path to save trained model
        device: Device to use ('cuda' or 'cpu')
        include_inventory: Whether to include inventory processing
        
    Returns:
        Tuple of (trained_model, train_losses, test_losses)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔥 Training MiniHackVAE with {len(train_samples)} train samples, {len(test_samples)} test samples")
    print(f"   Device: {device}")
    print(f"   Latent dim: {latent_dim}")
    print(f"   Include inventory: {include_inventory}")
    
    # Create adapter and datasets
    adapter = TTYToMultiModalHackAdapter()
    train_dataset = NetHackDataset(train_samples, adapter)
    test_dataset = NetHackDataset(test_samples, adapter)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = MultiModalHackVAE(latent_dim=latent_dim, include_inventory=include_inventory)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    train_losses = []
    test_losses = []
    
    print(f"\n🎯 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value
                
                # Forward pass
                model_output = model(
                    glyph_chars=batch_device['glyph_chars'],
                    glyph_colors=batch_device['glyph_colors'], 
                    blstats=batch_device['blstats'],
                    msg_tokens=batch_device['msg_tokens'],
                    hero_info=batch_device['hero_info'],
                    inv_oclasses=batch_device['inv_oclasses'],
                    inv_strs=batch_device['inv_strs']
                )
                
                # Calculate loss
                loss_dict = vae_loss(
                    model_output=model_output,
                    glyph_chars=batch_device['glyph_chars'],
                    glyph_colors=batch_device['glyph_colors'],
                    blstats=batch_device['blstats'],
                    msg_tokens=batch_device['msg_tokens'],
                    hero_info=batch_device['hero_info'],
                    inv_oclasses=batch_device['inv_oclasses'],
                    inv_strs=batch_device['inv_strs'],
                    weight_emb=1.0,
                    weight_raw=0.1,
                    kl_beta=1.0
                )
                
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.2f}",
                    'raw': f"{loss_dict['total_raw_loss'].item():.2f}",
                    'emb': f"{loss_dict['total_emb_loss'].item():.2f}",
                    'kl': f"{loss_dict['kl_loss'].item():.2f}"
                })
        
        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Testing phase
        model.eval()
        epoch_test_loss = 0.0
        test_batch_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value
                
                # Forward pass
                model_output = model(
                    glyph_chars=batch_device['glyph_chars'],
                    glyph_colors=batch_device['glyph_colors'], 
                    blstats=batch_device['blstats'],
                    msg_tokens=batch_device['msg_tokens'],
                    hero_info=batch_device['hero_info'],
                    inv_oclasses=batch_device['inv_oclasses'],
                    inv_strs=batch_device['inv_strs']
                )
                
                # Calculate loss
                loss_dict = vae_loss(
                    model_output=model_output,
                    glyph_chars=batch_device['glyph_chars'],
                    glyph_colors=batch_device['glyph_colors'],
                    blstats=batch_device['blstats'],
                    msg_tokens=batch_device['msg_tokens'],
                    hero_info=batch_device['hero_info'],
                    inv_oclasses=batch_device['inv_oclasses'],
                    inv_strs=batch_device['inv_strs'],
                    weight_emb=1.0,
                    weight_raw=0.1,
                    kl_beta=1.0
                )
                
                epoch_test_loss += loss_dict['total_loss'].item()
                test_batch_count += 1
        
        avg_test_loss = epoch_test_loss / test_batch_count if test_batch_count > 0 else 0.0
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.2f}, Test Loss = {avg_test_loss:.2f}")
    
    # Save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'latent_dim': latent_dim,
        'include_inventory': include_inventory,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
    }, save_path)
    
    print(f"\n✅ MiniHackVAE training completed!")
    print(f"  - Final train loss: {train_losses[-1]:.2f}")
    print(f"  - Final test loss: {test_losses[-1]:.2f}")
    print(f"  - Model saved to: {save_path}")
    
    return model, train_losses, test_losses


def visualize_reconstructions(model, test_samples, device, num_samples=4, save_path="reconstructions.png"):
    """Visualize VAE reconstructions"""
    model.eval()
    
    # Get a few test samples
    test_dataset = NetHackDataset(test_samples[:num_samples], target_type='reconstruction')
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=False)
    
    with torch.no_grad():
        batch = next(iter(test_loader))
        data = batch['input'].to(device)
        
        # Get reconstructions
        recon_data, mu, logvar, z = model(data)
        
        # Convert back to CPU for plotting
        original = data.cpu().numpy()
        reconstructed = recon_data.cpu().numpy()
        
        # Plot comparisons
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        
        for i in range(num_samples):
            # Original chars
            axes[i, 0].imshow(original[i, 0], cmap='gray')
            axes[i, 0].set_title(f'Original Chars {i+1}')
            axes[i, 0].axis('off')
            
            # Original colors  
            axes[i, 1].imshow(original[i, 1], cmap='viridis')
            axes[i, 1].set_title(f'Original Colors {i+1}')
            axes[i, 1].axis('off')
            
            # Reconstructed chars
            axes[i, 2].imshow(reconstructed[i, 0], cmap='gray')
            axes[i, 2].set_title(f'Recon Chars {i+1}')
            axes[i, 2].axis('off')
            
            # Reconstructed colors
            axes[i, 3].imshow(reconstructed[i, 1], cmap='viridis')
            axes[i, 3].set_title(f'Recon Colors {i+1}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    print(f"Reconstructions saved to {save_path}")


def analyze_latent_space(model, samples, device, save_path="latent_analysis.png"):
    """Analyze the learned latent space"""
    model.eval()
    
    # Get latent representations for all samples
    dataset = NetHackDataset(samples, target_type='reconstruction')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    latent_vectors = []
    game_ids = []
    messages = []
    
    with torch.no_grad():
        for batch in dataloader:
            data = batch['input'].to(device)
            mu, logvar = model.encode(data)
            
            latent_vectors.append(mu.cpu().numpy())
            game_ids.extend(batch['metadata']['gameid'])
            messages.extend(batch['metadata']['current_message'])
    
    # Combine all latent vectors
    latent_vectors = np.vstack(latent_vectors)
    
    print(f"📊 Latent Space Analysis:")
    print(f"  - Latent vectors shape: {latent_vectors.shape}")
    print(f"  - Mean latent values: {np.mean(latent_vectors, axis=0)[:5]}...")
    print(f"  - Std latent values: {np.std(latent_vectors, axis=0)[:5]}...")
    
    # Simple 2D visualization using first two dimensions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                         c=game_ids, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Game ID')
    plt.xlabel('Latent Dimension 0')
    plt.ylabel('Latent Dimension 1')
    plt.title('Latent Space Visualization (First 2 Dimensions)')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Latent space visualization saved to {save_path}")
    
    return latent_vectors, game_ids, messages


if __name__ == "__main__":
    pass

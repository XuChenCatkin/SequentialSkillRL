"""
Complete VAE training pipeline for NetHack Learning Dataset
Supports both the simple NetHackVAE and the sophisticated MiniHackVAE from src/model.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Optional, Callable
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import re  # Add regex import for status line parsing
import json
import logging
from collections import Counter
from datetime import datetime
import tempfile  # Add tempfile import
from torch.optim.lr_scheduler import OneCycleLR
warnings.filterwarnings('ignore')

# Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: pip install wandb")

# HuggingFace integration imports
try:
    from huggingface_hub import HfApi, Repository, upload_file, create_repo, login
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    print("⚠️  HuggingFace Hub not available. Install with: pip install huggingface_hub")
    HF_AVAILABLE = False

# Scikit-learn availability check
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Add utils to path for importing env_utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import env_utils

# Add src to path for importing the existing MultiModalHackVAE
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import MultiModalHackVAE, vae_loss, LATENT_DIM, CHAR_DIM, COLOR_DIM
import torch.optim as optim
import sqlite3
import random
from pathlib import Path
import nle.dataset as nld
from nle.nethack import tty_render
import matplotlib.pyplot as plt
import time
import logging

# Import our utility functions
import sys
sys.path.append('./utils')
from env_utils import get_current_message, get_status_lines, get_game_map, detect_valid_map
from utils.analysis import visualize_reconstructions, analyze_latent_space


class NetHackDataCollector:
    """Collects and preprocesses NetHack data for VAE training"""
    
    def __init__(self, dbfilename: str = "ttyrecs.db"):
        self.dbfilename = dbfilename
        self.collected_data = []
        self.game_hero_info = {} # game_id -> hero_info tensor [4,]
        
        # NetHack mappings for hero info parsing
        self.role_mapping = {
            'archaeologist': 0, 'archeologist': 0,  # Handle both spellings
            'barbarian': 1, 'caveman': 2, 'cavewoman': 2,
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
        
        #print(f"DEBUG: Message: '{message[:150]}...'")
        #print(f"DEBUG: Words found: {words}")  # All words for debugging
        
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
                # Set gender based on role if not already set
                if not gender:
                    if 'man' in role_name or role_name == 'priest':
                        gender = self.gender_mapping['male']
                    elif 'woman' in role_name or role_name == 'priestess' or role_name == 'valkyrie':
                        gender = self.gender_mapping['female']
                    else:
                        gender = self.gender_mapping['neuter']  # Default for other roles
                break
        
        # Only return if we found all required components
        #print(f"DEBUG: Found - alignment:{alignment}, gender:{gender}, race:{race}, role:{role}")
        if all(x is not None for x in [alignment, gender, race, role]):
            hero_info = torch.tensor([role, race, gender, alignment], dtype=torch.int32)
            self.game_hero_info[game_id] = hero_info
            #print(f"DEBUG: Successfully parsed hero info: {hero_info}")
            return hero_info

        #print(f"DEBUG: Failed to parse hero info - missing components")
        return None
    
    def collect_data_batch(self, dataset_name: str, adapter: callable, max_batches: int = 1000, 
                          batch_size: int = 32, seq_length: int = 32, game_ids: List[int] | None = None) -> List[Dict]:
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
            shuffle=True,
            gameids=game_ids
        )
        
        collected_batches = []
        batch_count = 0
        
        for batch_idx, minibatch in enumerate(dataset):
            if batch_count >= max_batches:
                break
                
            print(f"Processing batch {batch_idx + 1}...")
            
            this_batch = {}  # Create new dictionary
            
            num_games, num_time = minibatch['tty_chars'].shape[:2]
            
            # Deep copy numpy arrays to torch tensors
            for key, item in minibatch.items():
                if isinstance(item, np.ndarray):
                    # Create a copy of the numpy array first, then convert to tensor
                    this_batch[key] = torch.tensor(item) 
                else:
                    this_batch[key] = item

            message_chars_minibatch = torch.zeros((num_games, num_time, 256), dtype=torch.long)
            game_chars_minibatch = torch.ones((num_games, num_time, 21, 79), dtype=torch.long) * 32  # Fill with spaces
            game_colors_minibatch = torch.zeros((num_games, num_time, 21, 79), dtype=torch.long)
            status_chars_minibatch = torch.zeros((num_games, num_time, 2, 80), dtype=torch.long)
            hero_info_minibatch = torch.ones((num_games, num_time, 4), dtype=torch.int32)
            blstats_minibatch = torch.zeros((num_games, num_time, 27), dtype=torch.float32)
            valid_screen_minibatch = torch.ones((num_games, num_time), dtype=torch.bool)

            # Process each game in the batch
            for game_idx in range(this_batch['tty_chars'].shape[0]):
                # Process each timestep
                for time_idx in range(this_batch['tty_chars'].shape[1]):
                    
                    game_id = int(this_batch['gameids'][game_idx, time_idx])
                    if 'scores' in this_batch:
                        score = float(this_batch['scores'][game_idx, time_idx])
                    else:
                        score = 0.0

                    # Get TTY data
                    tty_chars = this_batch['tty_chars'][game_idx, time_idx]
                    tty_colors = this_batch['tty_colors'][game_idx, time_idx]
                    
                    is_valid_map = detect_valid_map(tty_chars)
                    if not is_valid_map:
                        valid_screen_minibatch[game_idx, time_idx] = False
                        continue
                    
                    game_map = get_game_map(tty_chars, tty_colors)
                    
                    # Extract text information
                    current_message = get_current_message(tty_chars)
                    message_chars = torch.tensor([ord(c) for c in current_message.ljust(256, chr(0))], dtype=torch.long)
                    status_lines = get_status_lines(tty_chars)
                    status_chars = tty_chars[22:, :]
                    
                    hero_info = self.get_hero_info(game_id, current_message)
                    if hero_info is None:
                        print(tty_render(tty_chars, tty_colors))
                        raise Exception(f"⚠️ No hero info found for game {game_id}, skipping sample")

                    # Fill in the minibatch tensors
                    message_chars_minibatch[game_idx, time_idx] = message_chars
                    game_chars_minibatch[game_idx, time_idx] = game_map['chars']
                    game_colors_minibatch[game_idx, time_idx] = game_map['colors']
                    status_chars_minibatch[game_idx, time_idx] = status_chars
                    hero_info_minibatch[game_idx, time_idx] = hero_info
                    blstats_minibatch[game_idx, time_idx] = adapter(game_map['chars'], status_lines, score)

            this_batch['message_chars'] = message_chars_minibatch
            this_batch['game_chars'] = game_chars_minibatch
            this_batch['game_colors'] = game_colors_minibatch
            this_batch['status_chars'] = status_chars_minibatch
            this_batch['hero_info'] = hero_info_minibatch
            this_batch['blstats'] = blstats_minibatch
            this_batch['valid_screen'] = valid_screen_minibatch
            collected_batches.append(this_batch)
            batch_count += 1

        print(f"✅ Successfully collected {len(collected_batches)} batches from {batch_count} processed batches")
        return collected_batches
    
    def save_collected_data(self, collected_batches: List[Dict], save_path: str) -> None:
        """
        Save collected batch data to disk for later reloading.
        
        Args:
            collected_batches: List of processed batch dictionaries from collect_data_batch
            save_path: Path to save the data (should end with .pkl or .pt)
        """
        print(f"💾 Saving {len(collected_batches)} collected batches to {save_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare data to save
        save_data = {
            'collected_batches': collected_batches,
            'game_hero_info': self.game_hero_info,  # Save cached hero info
            'num_batches': len(collected_batches),
            'save_timestamp': time.time(),
            'batch_shapes': {
                'game_chars': collected_batches[0]['game_chars'].shape if collected_batches else None,
                'game_colors': collected_batches[0]['game_colors'].shape if collected_batches else None,
                'hero_info': collected_batches[0]['hero_info'].shape if collected_batches else None,
                'blstats': collected_batches[0]['blstats'].shape if collected_batches else None,
                'message_chars': collected_batches[0]['message_chars'].shape if collected_batches else None
            }
        }
        
        # Save using appropriate method based on file extension
        if save_path.endswith('.pt'):
            torch.save(save_data, save_path)
        elif save_path.endswith('.pkl'):
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            # Default to pickle
            with open(save_path + '.pkl', 'wb') as f:
                pickle.dump(save_data, f)
            save_path = save_path + '.pkl'
        
        # Calculate file size
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        print(f"✅ Data saved successfully!")
        print(f"   📁 File: {save_path}")
        print(f"   📊 Size: {file_size:.1f} MB")
        print(f"   🎯 Batches: {len(collected_batches)}")
    
    def load_collected_data(self, load_path: str) -> List[Dict]:
        """
        Load previously saved collected batch data from disk.
        
        Args:
            load_path: Path to the saved data file
            
        Returns:
            List of processed batch dictionaries
        """
        print(f"📁 Loading collected batches from {load_path}...")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"❌ File not found: {load_path}")
        
        # Load using appropriate method based on file extension
        if load_path.endswith('.pt'):
            save_data = torch.load(load_path, map_location='cpu', weights_only=False)
        elif load_path.endswith('.pkl'):
            with open(load_path, 'rb') as f:
                save_data = pickle.load(f)
        else:
            # Try both formats
            try:
                save_data = torch.load(load_path, map_location='cpu')
            except:
                with open(load_path, 'rb') as f:
                    save_data = pickle.load(f)
        
        # Extract data
        collected_batches = save_data['collected_batches']
        self.game_hero_info = save_data.get('game_hero_info', {}) | self.game_hero_info  # Restore cached hero info
        
        # Calculate file size
        file_size = os.path.getsize(load_path) / (1024 * 1024)  # MB
        print(f"✅ Data loaded successfully!")
        print(f"   📁 File: {load_path}")
        print(f"   📊 Size: {file_size:.1f} MB")
        print(f"   🎯 Batches: {len(collected_batches)}")
        print(f"   🎮 Cached hero info: {len(self.game_hero_info)} games")
        if collected_batches:
            print(f"   📐 Batch shape: {collected_batches[0]['tty_chars'].shape}")
        
        return collected_batches
    
    def collect_or_load_data(self, dataset_name: str, adapter: callable, save_path: str, 
                           max_batches: int = 1000, batch_size: int = 32, seq_length: int = 32,
                           force_recollect: bool = False, game_ids: List[int] | None = None) -> List[Dict]:
        """
        Collect data from dataset or load from saved file if it exists.
        
        Args:
            dataset_name: Name of dataset in database
            adapter: Data adapter for processing
            save_path: Path to save/load the data
            max_batches: Maximum number of batches to collect (if collecting)
            batch_size: Batch size for dataset loader (if collecting)
            seq_length: Sequence length for dataset loader (if collecting)
            force_recollect: If True, always collect fresh data even if saved file exists
            
        Returns:
            List of processed data samples
        """
        if not force_recollect and os.path.exists(save_path):
            print(f"🔄 Found existing data file, loading instead of collecting...")
            return self.load_collected_data(save_path)
        else:
            print(f"🆕 Collecting fresh data...")
            collected_batches = self.collect_data_batch(
                dataset_name=dataset_name,
                adapter=adapter,
                max_batches=max_batches,
                batch_size=batch_size,
                seq_length=seq_length,
                game_ids=game_ids
            )
            
            # Save for future use
            self.save_collected_data(collected_batches, save_path)
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
            chars: Game characters array (21, 79) - already cropped from TTY
            
        Returns:
            Tuple of (x, y) coordinates, or cursor position if '@' not found
        """
        # Search for '@' symbol (ASCII 64) in the game map
        height, width = chars.shape
        for y in range(height):
            for x in range(width):
                if chars[y, x] == ord('@'):
                    return (x, y)
        
        # If '@' not found, return center of map as fallback
        return (width // 2, height // 2)
    
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
    
    def __call__(self, game_chars: np.ndarray, status_lines: List[str], score: float = 0.0) -> np.ndarray:
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
        blstats = torch.zeros(27, dtype=torch.float32)
        
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
        blstats[20] = int(parsed_stats.get('time', 0))
        
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

def ramp_weight(initial_weight: float, final_weight: float, shape: str, progress: float, rate: float = 10.0, centre: float = 0.5, f: Optional[Callable[[float, float, float], float]] = None) -> float:
    """
    Calculate ramped weight based on specified shape and progress
    
    Args:
        initial_weight: Starting weight
        final_weight: Final weight
        shape: Shape of the ramp ('linear', 'cubic', 'sigmoid', 'cosine', 'exponential')
        progress: Progress from 0.0 to 1.0
        rate: Rate of change (used for 'sigmoid' and 'exponential' shapes)
        centre: Centre point (used for 'sigmoid' shape)
        f: Custom function for 'custom' shape, should accept (initial_weight, final_weight, progress)

    Returns:
        Ramped weight value
    """
    if shape == 'linear':
        return initial_weight + (final_weight - initial_weight) * progress
    elif shape == 'cubic':
        return initial_weight + (final_weight - initial_weight) * (progress ** 3)
    elif shape == 'sigmoid':
        return initial_weight + (final_weight - initial_weight) * (1 / (1 + np.exp(-rate * (progress - centre))))
    elif shape == 'cosine':
        return initial_weight + (final_weight - initial_weight) * (0.5 * (1 - np.cos(np.pi * progress)))
    elif shape == 'exponential':
        return initial_weight + (final_weight - initial_weight) * (1 - np.exp(-rate * progress))
    elif shape == 'constant':
        assert initial_weight == final_weight, "For constant shape, initial and final weights must be equal."
        return initial_weight
    elif shape == 'custom':
        assert f is not None, "For custom shape, a function must be provided."
        return f(initial_weight, final_weight, progress)
    else:
        raise ValueError(f"Unknown shape: {shape}. Supported shapes: linear, cubic, sigmoid, cosine, exponential, constant, custom.")


def save_checkpoint(
    model: MultiModalHackVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_losses: List[float],
    test_losses: List[float],
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    scaler = None,  # Remove specific type annotation since GradScaler API changed
    checkpoint_dir: str = "checkpoints",
    keep_last_n: int = 3,
    upload_to_hf: bool = False,
    hf_repo_name: str = None,
    hf_token: str = None
) -> str:
    """
    Save training checkpoint with optional HuggingFace upload
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        train_losses: Training loss history
        test_losses: Test loss history
        scheduler: Learning rate scheduler to save (optional)
        scaler: GradScaler for mixed precision training (optional)
        checkpoint_dir: Directory to save checkpoints
        keep_last_n: Number of recent checkpoints to keep (older ones are deleted)
        upload_to_hf: Whether to upload checkpoint to HuggingFace
        hf_repo_name: HuggingFace repository name
        hf_token: HuggingFace token
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint filename
    checkpoint_filename = f"checkpoint_epoch_{epoch+1:04d}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'model_config': {
            # All MultiModalHackVAE constructor parameters
            'bInclude_glyph_bag': getattr(model, 'include_glyph_bag', True),
            'bInclude_hero': getattr(model, 'include_hero', True),
            'dropout_rate': getattr(model, 'dropout_rate', 0.0),
            'enable_dropout_on_latent': getattr(model, 'enable_dropout_on_latent', False),
            'enable_dropout_on_decoder': getattr(model, 'enable_dropout_on_decoder', False),
        },
        'checkpoint_timestamp': datetime.now().isoformat(),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add scaler state if available
    if scaler is not None:
        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Upload to HuggingFace if requested
    if upload_to_hf and hf_repo_name and HF_AVAILABLE:
        try:
            if hf_token:
                login(token=hf_token)
            
            api = HfApi()
            
            # Upload checkpoint to checkpoints/ folder in repo
            remote_path = f"checkpoints/{checkpoint_filename}"
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=remote_path,
                repo_id=hf_repo_name,
                repo_type="model",
                commit_message=f"Add checkpoint for epoch {epoch+1}"
            )
            print(f"☁️  Checkpoint uploaded to HuggingFace: {remote_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to upload checkpoint to HuggingFace: {e}")
    
    # Clean up old checkpoints (keep only last N)
    if keep_last_n > 0:
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
    
    return checkpoint_path


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    """
    Remove old checkpoint files, keeping only the most recent N checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    try:
        # Get all checkpoint files
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_epoch_") and filename.endswith(".pth"):
                filepath = os.path.join(checkpoint_dir, filename)
                # Extract epoch number for sorting
                try:
                    epoch_str = filename.replace("checkpoint_epoch_", "").replace(".pth", "")
                    epoch_num = int(epoch_str)
                    checkpoint_files.append((epoch_num, filepath))
                except ValueError:
                    continue
        
        # Sort by epoch number (newest first)
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints
        if len(checkpoint_files) > keep_last_n:
            for epoch_num, filepath in checkpoint_files[keep_last_n:]:
                os.remove(filepath)
                print(f"🗑️  Removed old checkpoint: {os.path.basename(filepath)}")
                
    except Exception as e:
        print(f"⚠️  Error cleaning up checkpoints: {e}")

def train_multimodalhack_vae(
    train_file: str, 
    test_file: str,                     
    dbfilename: str = 'ttyrecs.db',
    epochs: int = 10, 
    batch_size: int = 32, 
    sequence_size: int = 32, 
    training_batches: int = 100,
    testing_batches: int = 20,
    max_training_batches: int = 100,
    max_testing_batches: int = 20,
    training_game_ids: List[int] | None = None,
    testing_game_ids: List[int] | None = None,
    max_learning_rate: float = 1e-3,
    device: str = None, 
    logger: logging.Logger = None,
    data_cache_dir: str = "data_cache",
    force_recollect: bool = False,
    shuffle_batches: bool = True,
    shuffle_within_batch: bool = False,
    
    # Mixed precision parameters
    use_bf16: bool = False,
    
    # Adaptive loss weighting parameters
    initial_mi_beta: float = 0.0001,
    final_mi_beta: float = 1.0,
    mi_beta_shape: str = 'cosine',
    initial_tc_beta: float = 0.0001,
    final_tc_beta: float = 1.0,
    tc_beta_shape: str = 'cosine',
    initial_dw_beta: float = 0.0001,
    final_dw_beta: float = 1.0,
    dw_beta_shape: str = 'cosine',
    custom_kl_beta_function: Optional[Callable[[float, float, float], float]] = None,
    free_bits: float = 0.0,
    warmup_epoch_ratio: float = 0.3,
    focal_loss_alpha: float = 0.75,
    focal_loss_gamma: float = 2.0,
    
    # Dropout and regularization parameters
    dropout_rate: float = 0.0,
    enable_dropout_on_latent: bool = True,
    enable_dropout_on_decoder: bool = True,
    
    # Model saving and checkpointing parameters
    save_path: str = "models/nethack-vae.pth",
    save_checkpoints: bool = False,
    checkpoint_dir: str = "checkpoints",
    save_every_n_epochs: int = 2,
    keep_last_n_checkpoints: int = 2,
    
    # HuggingFace integration parameters
    upload_to_hf: bool = False,
    hf_repo_name: str = None,
    hf_token: str = None,
    hf_private: bool = True,
    hf_upload_artifacts: bool = True,
    hf_upload_directly: bool = True,
    hf_upload_checkpoints: bool = False,
    hf_model_card_data: Dict = None,
    
    # Resume training parameters
    resume_checkpoint_path: str = None,
    
    # Weights & Biases monitoring parameters
    use_wandb: bool = True,
    wandb_project: str = "nethack-vae",
    wandb_entity: str = None,
    wandb_run_name: str = None,
    wandb_tags: List[str] = None,
    wandb_notes: str = None,
    log_every_n_steps: int = 10,
    log_model_architecture: bool = True,
    log_gradients: bool = False,
    
    # Early stopping parameters
    early_stopping: bool = True,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 0.01) -> Tuple[MultiModalHackVAE, List[float], List[float]]:
    """
    Train MultiModalHackVAE on NetHack Learning Dataset with adaptive loss weighting

    Args:
        train_file: Path to the training samples
        test_file: Path to the testing samples
        dbfilename: Path to the NetHack Learning Dataset database file
        epochs: Number of training epochs
        batch_size: Training batch size
        max_learning_rate: max learning rate for optimizer
        device: Device to use ('cuda' or 'cpu')
        data_cache_dir: Directory to cache processed data
        force_recollect: Force data recollection even if cache exists
        shuffle_batches: Whether to shuffle training batches at the start of each epoch
        shuffle_within_batch: Whether to shuffle samples within each batch (ignores temporal order)
        use_bf16: Whether to use BF16 mixed precision training for memory efficiency
        initial_mi_beta: Starting mutual information KL divergence weight (very small initially)
        final_mi_beta: Final mutual information KL divergence weight
        mi_beta_shape: Shape of MI beta curve ('linear', 'cubic', 'sigmoid', 'cosine', 'exponential')
        initial_tc_beta: Starting total correlation KL divergence weight (very small initially)
        final_tc_beta: Final total correlation KL divergence weight
        tc_beta_shape: Shape of TC beta curve ('linear', 'cubic', 'sigmoid', 'cosine', 'exponential')
        initial_dw_beta: Starting dimension-wise KL divergence weight (very small initially)
        final_dw_beta: Final dimension-wise KL divergence weight
        dw_beta_shape: Shape of DW beta curve ('linear', 'cubic', 'sigmoid', 'cosine', 'exponential')
        custom_kl_beta_function: Custom function for KL beta ramping (applies to all three betas)
        free_bits: Target free bits for KL loss (0.0 disables)
        warmup_epoch_ratio: Ratio of epochs for warm-up phase
        focal_loss_alpha: Alpha parameter for focal loss (0.0 disables)
        focal_loss_gamma: Gamma parameter for focal loss (0.0 disables)
        dropout_rate: Dropout rate (0.0-1.0) for regularization. 0.0 disables dropout
        enable_dropout_on_latent: Whether to apply dropout to encoder fusion layers
        enable_dropout_on_decoder: Whether to apply dropout to decoder layers
        save_path: Path to save the trained model
        save_checkpoints: Whether to save checkpoints during training
        checkpoint_dir: Directory to save checkpoints
        save_every_n_epochs: Save checkpoint every N epochs
        keep_last_n_checkpoints: Keep only last N checkpoints, delete older ones
        upload_to_hf: Whether to upload model to HuggingFace Hub
        hf_repo_name: HuggingFace repository name for uploading
        hf_token: HuggingFace authentication token
        hf_private: Whether to make the uploaded model private
        hf_upload_artifacts: Whether to upload artifacts (e.g. datasets)
        hf_upload_directly: Whether to upload model directly or via artifacts
        hf_upload_checkpoints: Whether to upload checkpoints to HuggingFace
        hf_model_card_data: Additional metadata for HuggingFace model card
        resume_checkpoint_path: Path to resume training from
        use_wandb: Whether to use Weights & Biases for monitoring
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity (team/user)
        wandb_run_name: Name for the Weights & Biases run
        wandb_tags: Tags for the Weights & Biases run
        wandb_notes: Notes for the Weights & Biases run
        log_every_n_steps: Log metrics every N steps
        log_model_architecture: Whether to log model architecture to Weights & Biases
        log_gradients: Whether to log gradients to Weights & Biases
        early_stopping: Whether to enable early stopping based on test loss
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        early_stopping_min_delta: Minimum relative change in test loss to qualify as an improvement
        

    Returns:
        Tuple of (trained_model, train_losses, test_losses)
    """
    if device is None:
        device = torch.device('cpu')  # Use CPU for debugging
    else:
        # Ensure device is a torch.device object, not a string
        device = torch.device(device)

    # Setup logging
    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('training.log')
            ]
        )
        logger = logging.getLogger(__name__)

    # Initialize Weights & Biases if requested
    if use_wandb and WANDB_AVAILABLE:
        # Prepare configuration for wandb
        wandb_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "sequence_size": sequence_size,
            "max_learning_rate": max_learning_rate,
            "training_batches": training_batches,
            "testing_batches": testing_batches,
            "device": str(device),
            "use_bf16": use_bf16,
            "shuffle_batches": shuffle_batches,
            "shuffle_within_batch": shuffle_within_batch,
            "adaptive_weighting": {
                "initial_mi_beta": initial_mi_beta,
                "final_mi_beta": final_mi_beta,
                "mi_beta_shape": mi_beta_shape,
                "initial_tc_beta": initial_tc_beta,
                "final_tc_beta": final_tc_beta,
                "tc_beta_shape": tc_beta_shape,
                "initial_dw_beta": initial_dw_beta,
                "final_dw_beta": final_dw_beta,
                "dw_beta_shape": dw_beta_shape,
                "warmup_epoch_ratio": warmup_epoch_ratio
            },
            "regularization": {
                "dropout_rate": dropout_rate,
                "enable_dropout_on_latent": enable_dropout_on_latent,
                "enable_dropout_on_decoder": enable_dropout_on_decoder,
                "free_bits": free_bits,
                "focal_loss_alpha": focal_loss_alpha,
                "focal_loss_gamma": focal_loss_gamma
            },
            "checkpointing": {
                "save_checkpoints": save_checkpoints,
                "save_every_n_epochs": save_every_n_epochs,
                "keep_last_n_checkpoints": keep_last_n_checkpoints
            },
            "early_stopping": {
                "enabled": early_stopping,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta
            }
        }
        
        # Initialize wandb run
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
            tags=wandb_tags,
            notes=wandb_notes,
            resume="allow" if resume_checkpoint_path else False
        )
        
        logger.info("Weights & Biases initialized")
        
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️  wandb requested but not available. Install with: pip install wandb")

    logger.info(f"🔥Training MultiModalHackVAE with {training_batches} train batches, {testing_batches} test batches")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Sequence size: {sequence_size}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Mixed Precision: BF16 = {use_bf16}")
    logger.info(f"   Data cache: {data_cache_dir}")
    logger.info(f"   Shuffle batches: {shuffle_batches}")
    logger.info(f"   Shuffle within batch: {shuffle_within_batch}")
    logger.info(f"   Dropout Configuration:")
    logger.info(f"     - Dropout rate: {dropout_rate}")
    logger.info(f"     - Enable dropout on latent: {enable_dropout_on_latent}")
    logger.info(f"     - Enable dropout on decoder: {enable_dropout_on_decoder}")
    logger.info(f"   Adaptive Loss Weighting:")
    logger.info(f"     - MI beta: {initial_mi_beta:.3f} → {final_mi_beta:.3f}")
    logger.info(f"     - TC beta: {initial_tc_beta:.3f} → {final_tc_beta:.3f}")
    logger.info(f"     - DW beta: {initial_dw_beta:.3f} → {final_dw_beta:.3f}")
    logger.info(f"     - Warmup epochs: {int(warmup_epoch_ratio * epochs)} out of {epochs} total epochs")
    logger.info(f"   Free bits: {free_bits}")
    logger.info(f"   Focal loss: alpha={focal_loss_alpha}, gamma={focal_loss_gamma}")
    logger.info(f"   Early Stopping:")
    logger.info(f"     - Enabled: {early_stopping}")
    if early_stopping:
        logger.info(f"     - Patience: {early_stopping_patience} epochs")
        logger.info(f"     - Min delta: {early_stopping_min_delta:.6f}")

    def get_adaptive_weights(global_step: int, total_steps: int, f: Optional[Callable[[float, float, float], float]]) -> Tuple[float, float, float, float, float]:
        """Calculate adaptive weights based on current global training step"""
        # Calculate progress based on global step for smoother transitions
        progress = min(global_step / max(total_steps - 1, 1), 1.0)
        
        # Mutual Information beta: very small initially, then gradually increase
        mi_beta = ramp_weight(initial_weight=initial_mi_beta, 
            final_weight=final_mi_beta, 
            shape=mi_beta_shape, 
            progress=progress,
            f=f
        )
        
        # Total Correlation beta: very small initially, then gradually increase
        tc_beta = ramp_weight(initial_weight=initial_tc_beta, 
            final_weight=final_tc_beta, 
            shape=tc_beta_shape, 
            progress=progress,
            f=f
        )
        
        # Dimension-wise KL beta: very small initially, then gradually increase
        dw_beta = ramp_weight(initial_weight=initial_dw_beta, 
            final_weight=final_dw_beta, 
            shape=dw_beta_shape, 
            progress=progress,
            f=f
        )
        
        # Log the adaptive weights (only occasionally to avoid spam)
        if global_step % 100 == 0:
            logger.debug(f"Step {global_step}/{total_steps} - Adaptive weights: mi_beta={mi_beta:.3f}, tc_beta={tc_beta:.3f}, dw_beta={dw_beta:.3f}")

        return mi_beta, tc_beta, dw_beta
    
    # Create adapter and datasets with caching
    adapter = BLStatsAdapter()
    collector = NetHackDataCollector(dbfilename)
    
    # Create cache directory
    os.makedirs(data_cache_dir, exist_ok=True)
    
    # Cache file names based on dataset parameters
    train_cache_file = os.path.join(data_cache_dir, f"{train_file}_b{batch_size}_s{sequence_size}_m{max_training_batches}.pt")
    test_cache_file = os.path.join(data_cache_dir, f"{test_file}_b{batch_size}_s{sequence_size}_m{max_testing_batches}.pt")

    # Collect or load training data
    logger.info(f"📊 Preparing training data...")
    train_dataset = collector.collect_or_load_data(
        dataset_name=train_file,
        adapter=adapter,
        save_path=train_cache_file,
        max_batches=max_training_batches,
        batch_size=batch_size,
        seq_length=sequence_size,
        force_recollect=force_recollect,
        game_ids=training_game_ids
    )
    train_dataset = train_dataset[:training_batches] if len(train_dataset) > training_batches else train_dataset
    
    # Collect or load testing data
    logger.info(f"📊 Preparing testing data...")
    test_dataset = collector.collect_or_load_data(
        dataset_name=test_file,
        adapter=adapter,
        save_path=test_cache_file,
        max_batches=max_testing_batches,
        batch_size=batch_size,
        seq_length=sequence_size,
        force_recollect=force_recollect,
        game_ids=testing_game_ids
    )
    test_dataset = test_dataset[:testing_batches] if len(test_dataset) > testing_batches else test_dataset

    # Resume from checkpoint if specified
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        logger.info(f"🔄 Resuming from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        logger.info(f"   Resuming from epoch {start_epoch}/{epochs}")
        logger.info(f"   Previous train loss: {checkpoint['final_train_loss']:.4f}")
        logger.info(f"   Previous test loss: {checkpoint['final_test_loss']:.4f}")
        include_glyph_bag = checkpoint['model_config'].get('bInclude_glyph_bag', True)
        include_hero = checkpoint['model_config'].get('bInclude_hero', True)
        dropout_rate = checkpoint['model_config'].get('dropout_rate', dropout_rate)
        enable_dropout_on_latent = checkpoint['model_config'].get('enable_dropout_on_latent', enable_dropout_on_latent)
        enable_dropout_on_decoder = checkpoint['model_config'].get('enable_dropout_on_decoder', enable_dropout_on_decoder)
        model = MultiModalHackVAE(
            bInclude_glyph_bag=include_glyph_bag,
            bInclude_hero=include_hero,
            dropout_rate=dropout_rate,
            enable_dropout_on_latent=enable_dropout_on_latent,
            enable_dropout_on_decoder=enable_dropout_on_decoder,
            logger=logger
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Initialize optimizer and step-based scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_learning_rate, weight_decay=1e-4)
        total_train_steps = len(train_dataset) * epochs
        warmup_steps = int(warmup_epoch_ratio * total_train_steps) if warmup_epoch_ratio > 0 else 0

        scheduler = OneCycleLR(
            optimizer, 
            max_lr=max_learning_rate, 
            total_steps=total_train_steps, 
            pct_start=warmup_epoch_ratio,
            anneal_strategy='cos',
            div_factor=2.0,
            final_div_factor=5.0,
            cycle_momentum=False
        )
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) if 'scheduler_state_dict' in checkpoint else None
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if 'optimizer_state_dict' in checkpoint else None
        
        # Initialize loss tracking from checkpoint
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
    else:
        start_epoch = 0
        # Initialize model with dropout configuration
        # dropout_rate: 0.0 = no dropout, 0.1-0.3 = mild regularization, 0.5+ = strong regularization
        # Dropout is applied to encoder fusion layers and decoder layers when enabled
        model = MultiModalHackVAE(
            dropout_rate=dropout_rate,
            enable_dropout_on_latent=enable_dropout_on_latent,
            enable_dropout_on_decoder=enable_dropout_on_decoder,
            logger=logger
        )
        model = model.to(device)
        
        # Initialize optimizer and step-based scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_learning_rate, weight_decay=1e-4)
        total_train_steps = len(train_dataset) * epochs
        warmup_steps = int(warmup_epoch_ratio * total_train_steps) if warmup_epoch_ratio > 0 else 0
        
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=max_learning_rate, 
            total_steps=total_train_steps, 
            pct_start=warmup_epoch_ratio,
            anneal_strategy='cos',
            div_factor=2.0,
            final_div_factor=5.0,
            cycle_momentum=False
        )
        
        # Initialize loss tracking
        train_losses = []
        test_losses = []

    # Initialize GradScaler for mixed precision training (for both new and resumed training)
    scaler = torch.amp.GradScaler('cuda') if use_bf16 and device.type == 'cuda' else None
    
    # Restore scaler state if resuming from checkpoint and scaler is available
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path) and scaler is not None:
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"   Restored GradScaler state from checkpoint")
    
    # Log BF16 status
    if use_bf16 and device.type == 'cuda':
        logger.info(f"✨ BF16 mixed precision training enabled with GradScaler")
    elif use_bf16 and device.type != 'cuda':
        logger.warning(f"⚠️  BF16 requested but device is {device.type}, using FP32 instead")
    else:
        logger.info(f"🔧 Using FP32 precision training")

    # Log model architecture to wandb if requested
    if use_wandb and WANDB_AVAILABLE and log_model_architecture:
        wandb.watch(model, log_freq=log_every_n_steps, log_graph=True, log="all" if log_gradients else None)

    # Calculate total training steps for step-based adaptive weights and learning rate
    total_train_steps = len(train_dataset) * epochs
    warmup_steps = int(warmup_epoch_ratio * total_train_steps) if warmup_epoch_ratio > 0 else 0
    
    logger.info(f"Model has latent dimension: {model.latent_dim} and low-rank dimension: {model.lowrank_dim}")
    logger.info(f"🎯 Starting training for {epochs} epochs (starting from epoch {start_epoch})...")
    logger.info(f"   Total training steps: {total_train_steps}")
    logger.info(f"   Warmup steps: {warmup_steps}")
    
    # Initialize global step counter
    global_step = start_epoch * len(train_dataset)
    
    # Initialize early stopping variables
    best_test_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0
    best_epoch = -1
    
    # for diagnostics
    bag_char_prob = nn.Sequential(nn.Linear(LATENT_DIM, 256), nn.ReLU(), nn.Linear(256, CHAR_DIM)).to(device)
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"🎯 Epoch {epoch+1}/{epochs} - Starting epoch...")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "progress/overall": global_step / total_train_steps,
                "progress/warmup": min(global_step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0,
                "progress/epoch": epoch / epochs,
                "progress/global_step": global_step
            })
        
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        
        # Shuffle training batches for this epoch (if enabled)
        if shuffle_batches:
            shuffled_train_dataset = train_dataset[:]  # Create a proper copy
            random.shuffle(shuffled_train_dataset)
            logger.debug(f"Shuffled {len(shuffled_train_dataset)} training batches for epoch {epoch+1}")
            shuffled_test_dataset = test_dataset[:]  # Create a proper copy
            random.shuffle(shuffled_test_dataset)
            logger.debug(f"Shuffled {len(shuffled_test_dataset)} testing batches for epoch {epoch+1}")
        else:
            shuffled_train_dataset = train_dataset
            shuffled_test_dataset = test_dataset
        
        with tqdm(shuffled_train_dataset, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        value_device = value.to(device)
                        # Reshape tensors from [B, T, ...] to [B*T, ...]
                        # Multi-dimensional tensors (game_chars, game_colors, etc.)
                        B, T = value_device.shape[:2]
                        remaining_dims = value_device.shape[2:]
                        batch_device[key] = value_device.view(B * T, *remaining_dims)
                        
                    else:
                        batch_device[key] = value
                
                # Optional: shuffle within batch (ignore temporal order for VAE training)
                if shuffle_within_batch:
                    batch_size = batch_device['game_chars'].shape[0]  # B*T
                    shuffle_indices = torch.randperm(batch_size)
                    
                    for key, value in batch_device.items():
                        if value is not None and isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                            batch_device[key] = value[shuffle_indices]
                
                # Forward pass with mixed precision if enabled
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(use_bf16 and device.type == 'cuda')):
                    model.set_bag_detach(global_step < warmup_steps)  # Detach bag during warmup phase
                    model_output = model(
                        glyph_chars=batch_device['game_chars'],
                        glyph_colors=batch_device['game_colors'], 
                        blstats=batch_device['blstats'],
                        msg_tokens=batch_device['message_chars'],
                        hero_info=batch_device['hero_info']
                    )
                    
                    # Calculate adaptive weights for this step
                    mi_beta, tc_beta, dw_beta = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
                    
                    # Calculate loss
                    train_loss_dict = vae_loss(
                        model_output=model_output,
                        glyph_chars=batch_device['game_chars'],
                        glyph_colors=batch_device['game_colors'],
                        blstats=batch_device['blstats'],
                        msg_tokens=batch_device['message_chars'],
                        valid_screen=batch_device['valid_screen'],
                        mi_beta=mi_beta,
                        tc_beta=tc_beta,
                        dw_beta=dw_beta,
                        free_bits=free_bits,
                        focal_loss_alpha=focal_loss_alpha,
                        focal_loss_gamma=focal_loss_gamma
                    )

                    train_loss = train_loss_dict['total_loss']
                mu = model_output['mu'].detach()
                kl_diagnosis = train_loss_dict['kl_diagnosis']
                per_dim_kl = kl_diagnosis['dimension_wise_kl'].detach()
                dim_kl = kl_diagnosis['dimension_wise_kl_sum']
                mutual_info = kl_diagnosis['mutual_info']
                total_correlation = kl_diagnosis['total_correlation']
                eigvals = kl_diagnosis['eigenvalues'].detach()
                eigvals = eigvals.flip(0)  # Sort in descending order
                kl_eig = 0.5 * (eigvals - eigvals.log() - 1)
                var_explained = eigvals.cumsum(dim=0) / eigvals.sum(dim=0)
                median_idx = (var_explained >= 0.5).nonzero(as_tuple=True)[0][0]
                median_ratio = (median_idx + 1) / len(var_explained)
                ninety_percentile_idx = (var_explained >= 0.9).nonzero(as_tuple=True)[0][0]
                ninety_percentile_ratio = (ninety_percentile_idx + 1) / len(var_explained)

                # Backward pass with mixed precision scaling if enabled
                if scaler is not None:
                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    train_loss.backward()
                    optimizer.step()
                scheduler.step()  # Step-based learning rate scheduling
                
                # Update global step counter
                global_step += 1

                epoch_train_loss += train_loss.item()
                batch_count += 1

                # Log to wandb every N steps if enabled
                if use_wandb and WANDB_AVAILABLE and global_step % log_every_n_steps == 0:
                    # Helper function to safely convert tensors for wandb logging
                    def safe_tensor_for_wandb(tensor):
                        """Convert tensor to float32 for wandb compatibility"""
                        if isinstance(tensor, torch.Tensor):
                            return tensor.detach().float().cpu()
                        return tensor
                    
                    wandb_log_dict = {
                        # Training metrics
                        "train/loss": train_loss.item(),
                        "train/batch": batch_count,
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        
                        # Loss components
                        "train/raw_loss/total": train_loss_dict['total_raw_loss'].item(),
                        "train/raw_loss/occupancy": train_loss_dict['raw_losses']['occupy'].item(),
                        "train/raw_loss/rare_occupancy": train_loss_dict['raw_losses']['rare_occupy'].item(),
                        "train/raw_loss/total_variation": train_loss_dict['raw_losses']['tv'].item(),
                        "train/raw_loss/dice_loss": train_loss_dict['raw_losses']['dice'].item(),
                        "train/raw_loss/common_glyph_chars": train_loss_dict['raw_losses']['common_char'].item(),
                        "train/raw_loss/common_glyph_colors": train_loss_dict['raw_losses']['common_color'].item(),
                        "train/raw_loss/rare_glyph_chars": train_loss_dict['raw_losses']['rare_char'].item(),
                        "train/raw_loss/rare_glyph_colors": train_loss_dict['raw_losses']['rare_color'].item(),
                        "train/raw_loss/bag_loss": train_loss_dict['raw_losses']['bag'].item(),
                        "train/raw_loss/hero_loc": train_loss_dict['raw_losses']['hero_loc'].item(),
                        "train/raw_loss/blstats": train_loss_dict['raw_losses']['stats'].item(),
                        "train/raw_loss/message": train_loss_dict['raw_losses']['msg'].item(),

                        "train/kl_loss": train_loss_dict['kl_loss'].item(),
                        "train/kl_loss/dimension_wise": dim_kl,
                        "train/kl_loss/mutual_info": mutual_info,
                        "train/kl_loss/total_correlation": total_correlation,

                        # Adaptive weights
                        "adaptive_weights/mi_beta": mi_beta,
                        "adaptive_weights/tc_beta": tc_beta,
                        "adaptive_weights/dw_beta": dw_beta,
                        
                        # Dropout status
                        "dropout/rate": model.dropout_rate,
                        
                        # Model diagnostics
                        "model/mu_var": safe_tensor_for_wandb(mu.var(dim=0)),
                        "model/mu_var_max": mu.var(dim=0).max().item(),
                        "model/mu_var_min": mu.var(dim=0).min().item(),
                        "model/mu_var_exceed_0.1": mu.var(dim=0).gt(0.1).sum().item() / mu.var(dim=0).numel(),
                        "model/per_dim_kl": safe_tensor_for_wandb(per_dim_kl),
                        "model/per_dim_kl_max": per_dim_kl.max().item(),
                        "model/per_dim_kl_min": per_dim_kl.min().item(),
                        "model/var_explained_median": median_ratio,
                        "model/var_explained_90_percentile": ninety_percentile_ratio,
                        "model/eigenval_max": eigvals[0].item(),
                        "model/eigenval_min": eigvals[-1].item(),
                        "model/eigenval_ratio": (eigvals[0] / eigvals[-1]).item(),
                        "model/eigenval": safe_tensor_for_wandb(eigvals),
                        "model/eigenval_exceed_2": (eigvals > 2).sum().item() / eigvals.numel(),
                        "model/kl_eigenval": safe_tensor_for_wandb(kl_eig),
                        "model/kl_eigenval_max": kl_eig.max().item(),
                        "model/kl_eigenval_min": kl_eig.min().item(),
                        "model/kl_eigenval_exceed_0.2": (kl_eig > 0.2).sum().item() / kl_eig.numel()
                    }
                    wandb.log(wandb_log_dict)

                # Update progress bar with summary metrics only
                pbar.set_postfix({
                    'loss': f"{train_loss.item():.2f}",
                    'total_raw': f"{train_loss_dict['total_raw_loss'].item():.2f}",
                    'kl': f"{train_loss_dict['kl_loss'].item():.2f}",
                })
        
        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Testing phase
        model.eval()
        epoch_test_loss = 0.0
        test_batch_count = 0
        
        with torch.no_grad():
            for batch in shuffled_test_dataset:
                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        value_device = value.to(device)
                        # Reshape tensors from [B, T, ...] to [B*T, ...]
                        # Multi-dimensional tensors (game_chars, game_colors, etc.)
                        B, T = value_device.shape[:2]
                        remaining_dims = value_device.shape[2:]
                        batch_device[key] = value_device.view(B * T, *remaining_dims)
                    else:
                        batch_device[key] = value
                
                # Optional: shuffle within batch (ignore temporal order for VAE training)
                if shuffle_within_batch:
                    batch_size = batch_device['game_chars'].shape[0]  # B*T
                    shuffle_indices = torch.randperm(batch_size)
                    
                    for key, value in batch_device.items():
                        if value is not None and isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                            batch_device[key] = value[shuffle_indices]
                
                # Forward pass with mixed precision if enabled
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(use_bf16 and device.type == 'cuda')):
                    model_output = model(
                        glyph_chars=batch_device['game_chars'],
                        glyph_colors=batch_device['game_colors'], 
                        blstats=batch_device['blstats'],
                        msg_tokens=batch_device['message_chars'],
                        hero_info=batch_device['hero_info']
                    )
                    
                    # Calculate adaptive weights for this step (use current global step for consistency)
                    mi_beta, tc_beta, dw_beta = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
                    
                    # Calculate loss
                    test_loss_dict = vae_loss(
                        model_output=model_output,
                        glyph_chars=batch_device['game_chars'],
                        glyph_colors=batch_device['game_colors'],
                        blstats=batch_device['blstats'],
                        msg_tokens=batch_device['message_chars'],
                        valid_screen=batch_device['valid_screen'],
                        mi_beta=mi_beta,
                        tc_beta=tc_beta,
                        dw_beta=dw_beta,
                        free_bits=free_bits,
                        focal_loss_alpha=focal_loss_alpha,
                        focal_loss_gamma=focal_loss_gamma
                    )

                    test_loss = test_loss_dict['total_loss']
                epoch_test_loss += test_loss.item()
                test_batch_count += 1
                
                if use_wandb and WANDB_AVAILABLE and test_batch_count % log_every_n_steps == 0:
                    wandb_log_dict = {
                        "test/loss": test_loss.item(),
                        
                        # Loss components
                        "test/raw_loss/total": test_loss_dict['total_raw_loss'].item(),
                        "test/raw_loss/occupancy": test_loss_dict['raw_losses']['occupy'].item(),
                        "test/raw_loss/rare_occupancy": test_loss_dict['raw_losses']['rare_occupy'].item(),
                        "test/raw_loss/common_glyph_chars": test_loss_dict['raw_losses']['common_char'].item(),
                        "test/raw_loss/common_glyph_colors": test_loss_dict['raw_losses']['common_color'].item(),
                        "test/raw_loss/rare_glyph_chars": test_loss_dict['raw_losses']['rare_char'].item(),
                        "test/raw_loss/rare_glyph_colors": test_loss_dict['raw_losses']['rare_color'].item(),
                        "test/raw_loss/blstats": test_loss_dict['raw_losses']['stats'].item(),
                        "test/raw_loss/message": test_loss_dict['raw_losses']['msg'].item(),

                        "test/kl_loss": test_loss_dict['kl_loss'].item(),
                    }
                    wandb.log(wandb_log_dict)
        
        avg_test_loss = epoch_test_loss / test_batch_count if test_batch_count > 0 else 0.0
        test_losses.append(avg_test_loss)
        
        # Early stopping logic
        if early_stopping:
            improvement = best_test_loss / avg_test_loss - 1
            if improvement > early_stopping_min_delta:
                # Improvement found
                best_test_loss = avg_test_loss
                best_epoch = epoch
                early_stopping_counter = 0
                # Save the best model state
                best_model_state = {
                    'model_state_dict': model.state_dict().copy(),
                    'optimizer_state_dict': optimizer.state_dict().copy(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'test_loss': avg_test_loss,
                    'train_losses': train_losses.copy(),
                    'test_losses': test_losses.copy()
                }
                logger.info(f"💚 New best test loss: {best_test_loss:.4f} (epoch {epoch+1})")
            else:
                # No improvement
                early_stopping_counter += 1
                logger.info(f"⏰ No improvement in test loss for {early_stopping_counter}/{early_stopping_patience} epochs")
                
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"🛑 Early stopping triggered! Best test loss: {best_test_loss:.4f} at epoch {best_epoch+1}")
                    
                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state['model_state_dict'])
                        optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                        if scheduler and best_model_state['scheduler_state_dict'] is not None:
                            scheduler.load_state_dict(best_model_state['scheduler_state_dict'])
                        
                        # Update loss lists to reflect the best model
                        train_losses = best_model_state['train_losses']
                        test_losses = best_model_state['test_losses']
                        
                        logger.info(f"✅ Restored best model from epoch {best_epoch+1}")
                    
                    # Log early stopping to wandb
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "early_stopping/triggered": True,
                            "early_stopping/best_epoch": best_epoch + 1,
                            "early_stopping/best_test_loss": best_test_loss,
                            "early_stopping/stopped_at_epoch": epoch + 1,
                            "early_stopping/patience_used": early_stopping_counter
                        })
                    
                    break  # Exit training loop
        
        # Log epoch summary
        # Calculate current adaptive weights for display
        current_mi_beta, current_tc_beta, current_dw_beta = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
        
        logger.info(f"\n=== Epoch {epoch+1}/{epochs} Summary ===")
        logger.info(f"Average Train Loss: {avg_train_loss:.3f} | Average Test Loss: {avg_test_loss:.3f}")
        if early_stopping:
            logger.info(f"Early Stopping: Best Test Loss: {best_test_loss:.4f} (epoch {best_epoch+1}) | Counter: {early_stopping_counter}/{early_stopping_patience}")
        logger.info(f"Current KL Betas: mi={current_mi_beta:.3f}, tc={current_tc_beta:.3f}, dw={current_dw_beta:.3f}")
        logger.info(f"Global Step: {global_step}/{total_train_steps} ({100*global_step/total_train_steps:.1f}%)")
        
        # Show detailed modality breakdown for the last batch of training and testing
        logger.info(f"Final Training Batch Details:")
        raw_losses = train_loss_dict['raw_losses']
        raw_loss_str = " | ".join([f"{key}: {value.item():.3f}" for key, value in raw_losses.items()])
        logger.info(f"  Raw Losses: {raw_loss_str}")
        
        logger.info(f"Final Testing Batch Details:")
        raw_losses = test_loss_dict['raw_losses']
        raw_loss_str = " | ".join([f"{key}: {value.item():.3f}" for key, value in raw_losses.items()])
        logger.info(f"  Raw Losses: {raw_loss_str}")

        logger.info(f"Variance of model output (mu): {', '.join(f'{v:.4f}' for v in mu.var(dim=0).tolist())}")
        logger.info(f"Per-dim KL: {', '.join(f'{v:.4f}' for v in per_dim_kl.tolist())}")
        logger.info(f"Eigenvalues of latent space: {', '.join(f'{v:.4f}' for v in eigvals.tolist())}")
        logger.info(f"KL Eigenvalues: {', '.join(f'{v:.4f}' for v in kl_eig.tolist())}")
        logger.info(f"Variance explained by eigenvalues: {', '.join(f'{v:.4f}' for v in var_explained.tolist())}")

        logger.info("=" * 50)
        
        # Log epoch metrics to wandb
        if use_wandb and WANDB_AVAILABLE:
            epoch_log_dict = {
                # Epoch summaries
                "epoch/train_loss": avg_train_loss,
                "epoch/test_loss": avg_test_loss,
                "epoch/number": epoch + 1,
                
                # Final batch details for comparison
                "epoch/final_train_raw_total": train_loss_dict['total_raw_loss'].item(),
                "epoch/final_train_kl": train_loss_dict['kl_loss'].item(),
                
                "epoch/final_test_raw_total": test_loss_dict['total_raw_loss'].item(),
                "epoch/final_test_kl": test_loss_dict['kl_loss'].item()
            }
            
            # Add early stopping metrics
            if early_stopping:
                epoch_log_dict.update({
                    "early_stopping/best_test_loss": best_test_loss,
                    "early_stopping/best_epoch": best_epoch + 1,
                    "early_stopping/counter": early_stopping_counter,
                    "early_stopping/patience": early_stopping_patience,
                    "early_stopping/improvement": best_test_loss - avg_test_loss,
                    "early_stopping/is_best": avg_test_loss == best_test_loss
                })
            
            wandb.log(epoch_log_dict)
        
        
        # Save checkpoint if requested
        if save_checkpoints and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_losses=train_losses,
                test_losses=test_losses,
                scheduler=scheduler,
                scaler=scaler,
                checkpoint_dir=checkpoint_dir,
                keep_last_n=keep_last_n_checkpoints,
                upload_to_hf=hf_upload_checkpoints and upload_to_hf,
                hf_repo_name=hf_repo_name,
                hf_token=hf_token
            )
            
            # Log checkpoint save event to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "checkpoint/saved": True,
                    "checkpoint/epoch": epoch + 1,
                    "checkpoint/path": checkpoint_path,
                    "checkpoint/train_loss": avg_train_loss,
                    "checkpoint/test_loss": avg_test_loss,
                })
    
    logger.info(f"\n✅ MultiModalVAE training completed!")
    
    # Handle early stopping results
    if early_stopping and best_model_state is not None:
        logger.info(f"  - Training stopped early at epoch {epoch+1}")
        logger.info(f"  - Best model from epoch {best_epoch+1}")
        logger.info(f"  - Best train loss: {best_model_state['train_loss']:.4f}")
        logger.info(f"  - Best test loss: {best_model_state['test_loss']:.4f}")
        
        # Ensure we're using the best model for final operations
        model.load_state_dict(best_model_state['model_state_dict'])
        final_train_loss = best_model_state['train_loss']
        final_test_loss = best_model_state['test_loss']
    else:
        logger.info(f"  - Completed all {epochs} epochs")
        logger.info(f"  - Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"  - Final test loss: {test_losses[-1]:.4f}")
        final_train_loss = train_losses[-1]
        final_test_loss = test_losses[-1]
    
    # HuggingFace upload if requested
    if upload_to_hf and hf_repo_name and HF_AVAILABLE:
        logger.info(f"\n🤗 Uploading best model to HuggingFace Hub...")
        try:
            # Prepare training configuration for model card
            training_config = {
                "epochs": epochs,
                "batch_size": batch_size,
                "max_learning_rate": max_learning_rate,
                "sequence_size": sequence_size,
                "shuffle_batches": shuffle_batches,
                "shuffle_within_batch": shuffle_within_batch,
                "adaptive_weighting": {
                    "initial_mi_beta": initial_mi_beta,
                    "final_mi_beta": final_mi_beta,
                    "mi_beta_shape": mi_beta_shape,
                    "initial_tc_beta": initial_tc_beta,
                    "final_tc_beta": final_tc_beta,
                    "tc_beta_shape": tc_beta_shape,
                    "initial_dw_beta": initial_dw_beta,
                    "final_dw_beta": final_dw_beta,
                    "dw_beta_shape": dw_beta_shape,
                    "warmup_epoch_ratio": warmup_epoch_ratio
                },
                "free_bits": free_bits,
                "focal_loss_alpha": focal_loss_alpha,
                "focal_loss_gamma": focal_loss_gamma,
                "dropout_rate": dropout_rate,
                "enable_dropout_on_latent": enable_dropout_on_latent,
                "enable_dropout_on_decoder": enable_dropout_on_decoder,
                "early_stopping": {
                    "enabled": early_stopping,
                    "patience": early_stopping_patience,
                    "min_delta": early_stopping_min_delta,
                    "triggered": early_stopping and best_model_state is not None,
                    "best_epoch": best_epoch + 1 if best_model_state is not None else None,
                }
            }
            
            # Merge with user-provided model card data
            model_card_data = hf_model_card_data or {}
            model_card_data.update({
                "training_config": training_config,
                "final_train_loss": final_train_loss,
                "final_test_loss": final_test_loss,
                "best_train_loss": min(train_losses),
                "best_test_loss": min(test_losses),
                "total_epochs": epochs
            })
            
            # Upload model (save locally first or upload directly)
            commit_msg = f"Upload MultiModalHackVAE"
            if early_stopping and best_model_state is not None:
                commit_msg += f" (early stop at epoch {epoch+1}, best epoch {best_epoch+1}, test_loss={final_test_loss:.4f})"
            else:
                commit_msg += f" (epochs={epochs}, final_loss={final_test_loss:.4f})"
                
            if hf_upload_directly:
                repo_url = save_model_to_huggingface(
                    model=model,
                    repo_name=hf_repo_name,
                    token=hf_token,
                    private=hf_private,
                    commit_message=commit_msg,
                    model_card_data=model_card_data,
                    upload_directly=True
                )
            else:
                # Save model locally first
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'final_train_loss': final_train_loss,
                    'final_test_loss': final_test_loss,
                    'model_config': {
                        'latent_dim': getattr(model, 'latent_dim', 96),
                        'lowrank_dim': getattr(model, 'lowrank_dim', 0),
                    },
                    'training_timestamp': datetime.now().isoformat(),
                }, save_path)
                logger.info(f"💾 Model saved locally: {save_path}")
                
                repo_url = save_model_to_huggingface(
                    model=model,
                    model_save_path=save_path,
                    repo_name=hf_repo_name,
                    token=hf_token,
                    private=hf_private,
                    commit_message=commit_msg,
                    model_card_data=model_card_data,
                    upload_directly=False
                )
            
            # Upload training artifacts if requested
            if hf_upload_artifacts:
                upload_training_artifacts_to_huggingface(
                    repo_name=hf_repo_name,
                    train_losses=train_losses,
                    test_losses=test_losses,
                    training_config=training_config,
                    token=hf_token
                )
                
                # Create and upload demo notebook
                create_model_demo_notebook(hf_repo_name, "demo_notebook.ipynb")
                
                from huggingface_hub import HfApi, login
                api = HfApi()
                if hf_token:
                    login(token=hf_token)
                    
                api.upload_file(
                    path_or_fileobj="demo_notebook.ipynb",
                    path_in_repo="demo_notebook.ipynb",
                    repo_id=hf_repo_name,
                    repo_type="model",
                    commit_message="Add demo notebook"
                )
                os.remove("demo_notebook.ipynb")
            
            logger.info(f"🎉 Model successfully shared at: {repo_url}")
            
            # Log HuggingFace upload success to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "huggingface/upload_success": True,
                    "huggingface/repo_url": repo_url,
                    "huggingface/artifacts_uploaded": hf_upload_artifacts,
                    "huggingface/final_train_loss": train_losses[-1],
                    "huggingface/final_test_loss": test_losses[-1],
                })
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to HuggingFace: {e}")
            logger.info("   Model was still saved locally.")
            
            # Log HuggingFace upload failure to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "huggingface/upload_success": False,
                    "huggingface/error": str(e),
                })
    
    elif upload_to_hf and not HF_AVAILABLE:
        logger.warning("⚠️  HuggingFace Hub not available. Install with: pip install huggingface_hub")
    elif upload_to_hf and not hf_repo_name:
        logger.warning("⚠️  HuggingFace upload requested but no repo_name provided")
    elif not upload_to_hf:
        # Save model locally
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'model_config': {
                'latent_dim': getattr(model, 'latent_dim', 96),
                'lowrank_dim': getattr(model, 'lowrank_dim', 0),
            },
            'training_timestamp': datetime.now().isoformat(),
        }, save_path)
        logger.info(f"💾 Model saved locally: {save_path}")
    
    # Final wandb logging and cleanup
    if use_wandb and WANDB_AVAILABLE:
        # Log final training summary
        final_log_dict = {
            "training/completed": True,
            "training/total_epochs": epochs,
            "training/best_train_loss": min(train_losses),
            "training/best_test_loss": min(test_losses),
            "training/final_train_loss": final_train_loss,
            "training/final_test_loss": final_test_loss,
        }
        
        # Add early stopping metrics to final summary
        if early_stopping:
            final_log_dict.update({
                "training/early_stopping_enabled": True,
                "training/early_stopping_triggered": best_model_state is not None,
                "training/early_stopping_patience": early_stopping_patience,
                "training/epochs_completed": epoch + 1,
            })
            if best_model_state is not None:
                final_log_dict.update({
                    "training/best_model_epoch": best_epoch + 1,
                    "training/stopped_early_at_epoch": epoch + 1,
                })
        else:
            final_log_dict["training/early_stopping_enabled"] = False
        
        wandb.log(final_log_dict)
        
        # Mark run as finished
        wandb.finish()
    
    return model, train_losses, test_losses

def save_model_to_huggingface(
    model: MultiModalHackVAE,
    model_save_path: str = None,
    repo_name: str = None,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: str = "Upload MultiModalHackVAE model",
    model_card_data: Optional[Dict] = None,
    upload_directly: bool = False,
    additional_files: Optional[Dict[str, str]] = None
) -> str:
    """
    Save trained MultiModalHackVAE model to HuggingFace Hub
    
    Args:
        model: Trained MultiModalHackVAE model
        model_save_path: Local path where model is saved (optional if upload_directly=True)
        repo_name: Name for the HuggingFace repository (e.g., "username/nethack-vae")
        token: HuggingFace token (if None, will try to use cached token)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        model_card_data: Additional metadata for the model card
        upload_directly: If True, save model to temp file and upload directly without permanent local save
        additional_files: Dict of {local_path: remote_path} for additional files to upload
        
    Returns:
        Repository URL on HuggingFace Hub
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace Hub is required. Install with: pip install huggingface_hub")
    
    # Login if token is provided
    if token:
        login(token=token)
    
    api = HfApi()
    
    try:
        # Check if repository exists, create if not
        try:
            repo_info = api.repo_info(repo_id=repo_name, repo_type="model")
            print(f"📁 Repository {repo_name} already exists")
        except RepositoryNotFoundError:
            print(f"🆕 Creating new repository: {repo_name}")
            api.create_repo(repo_id=repo_name, private=private, repo_type="model")
    
        # Handle direct upload vs local file upload
        temp_model_file = None
        if upload_directly:
            # Create temporary file for direct upload
            import tempfile
            temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
            model_save_path = temp_model_file.name
            
            # Save model state dict to temporary file
            print(f"💾 Creating temporary model file for direct upload...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'latent_dim': getattr(model, 'latent_dim', 96),
                    'lowrank_dim': getattr(model, 'lowrank_dim', 0),
                },
                'upload_timestamp': datetime.now().isoformat(),
            }, model_save_path)
            temp_model_file.close()
        elif model_save_path is None:
            raise ValueError("Either model_save_path must be provided or upload_directly must be True")
    
        # Prepare model metadata
        model_info = {
            "model_type": "MultiModalHackVAE",
            "framework": "PyTorch",
            "task": "representation-learning",
            "dataset": "NetHack Learning Dataset",
            "latent_dim": getattr(model, 'latent_dim', 'unknown'),
            "lowrank_dim": getattr(model, 'lowrank_dim', 'unknown'),
            'bInclude_glyph_bag': getattr(model, 'include_glyph_bag', True),
            'bInclude_hero': getattr(model, 'include_hero', True),
            'dropout_rate': getattr(model, 'dropout_rate', 0.1),
            'enable_dropout_on_latent': getattr(model, 'enable_dropout_on_latent', True),
            'enable_dropout_on_decoder': getattr(model, 'enable_dropout_on_decoder', True),
            "architecture": "Multi-modal Variational Autoencoder for NetHack game states"
        }
        
        if model_card_data:
            model_info.update(model_card_data)
        
        # Create model card
        model_card_content = f"""---
license: mit
language: en
tags:
- nethack
- reinforcement-learning
- variational-autoencoder
- representation-learning
- multimodal
- world-modeling
pipeline_tag: feature-extraction
---

# MultiModalHackVAE

A multi-modal Variational Autoencoder trained on NetHack game states for representation learning.

## Model Description

This model is a MultiModalHackVAE that learns compact representations of NetHack game states by processing:
- Game character grids (21x79)
- Color information
- Game statistics (blstats)
- Message text
- Bag of glyphs
- Hero information (role, race, gender, alignment)

## Model Details

- **Model Type**: Multi-modal Variational Autoencoder
- **Framework**: PyTorch
- **Dataset**: NetHack Learning Dataset
- **Latent Dimensions**: {model_info.get('latent_dim', 'unknown')}
- **Low-rank Dimensions**: {model_info.get('lowrank_dim', 'unknown')}

## Usage

```python
from train import load_model_from_huggingface
import torch

# Load the model
model = load_model_from_huggingface("{repo_name}")

# Example usage with synthetic data
batch_size = 1
game_chars = torch.randint(32, 127, (batch_size, 21, 79))
game_colors = torch.randint(0, 16, (batch_size, 21, 79))
blstats = torch.randn(batch_size, 27)
msg_tokens = torch.randint(0, 128, (batch_size, 256))
hero_info = torch.randint(0, 10, (batch_size, 4))

with torch.no_grad():
    output = model(
        glyph_chars=game_chars,
        glyph_colors=game_colors,
        blstats=blstats,
        msg_tokens=msg_tokens,
        hero_info=hero_info
    )
    latent_mean = output['mu']
    latent_logvar = output['logvar']
    lowrank_factors = output['lowrank_factors']
```

## Training

This model was trained using adaptive loss weighting with:
- Embedding warm-up for quick convergence
- Gradual raw reconstruction focus
- KL beta annealing for better latent structure

## Citation

If you use this model, please consider citing:

```bibtex
@misc{{nethack-vae,
  title={{MultiModalHackVAE: Multi-modal Variational Autoencoder for NetHack}},
  author={{Xu Chen}},
  year={{2025}},
  url={{https://huggingface.co/{repo_name}}}
}}
```
"""
        
        # Save model card
        model_card_path = "VAE_README.md"
        with open(model_card_path, "w") as f:
            f.write(model_card_content)
        
        # Save model config
        config_path = "VAE_config.json"
        with open(config_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Upload files
        print(f"📤 Uploading model to {repo_name}...")
        
        # Upload model file
        api.upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo="pytorch_model.bin",
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message
        )
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add model card"
        )
        
        # Upload config
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add model config"
        )
        
        # Upload additional files if provided
        if additional_files:
            for local_path, remote_path in additional_files.items():
                if os.path.exists(local_path):
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=remote_path,
                        repo_id=repo_name,
                        repo_type="model",
                        commit_message=f"Add {remote_path}"
                    )
        
        # Clean up temporary files
        os.remove(model_card_path)
        os.remove(config_path)
        
        # Clean up temporary model file if created
        if temp_model_file is not None:
            os.unlink(model_save_path)
            print(f"🗑️  Cleaned up temporary model file")
        
        repo_url = f"https://huggingface.co/{repo_name}"
        print(f"✅ Model successfully uploaded to: {repo_url}")
        return repo_url
        
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace: {e}")
        # Clean up temporary file on error
        if temp_model_file is not None and os.path.exists(model_save_path):
            os.unlink(model_save_path)
        raise


def load_model_from_local(
    checkpoint_path: str,
    device: str = "cpu",
    **model_kwargs
) -> MultiModalHackVAE:
    """
    Load MultiModalHackVAE model from local checkpoint
    
    Args:
        checkpoint_path: Path to local checkpoint file
        device: Device to load the model on
        **model_kwargs: Additional arguments for model initialization (override config)
        
    Returns:
        Loaded MultiModalHackVAE model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        print(f"📥 Loading model from local checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract model configuration from checkpoint
        model_config = {
            "bInclude_glyph_bag": checkpoint.get('model_config', {}).get('bInclude_glyph_bag', True),
            "bInclude_hero": checkpoint.get('model_config', {}).get('bInclude_hero', True),
            "dropout_rate": checkpoint.get('model_config', {}).get('dropout_rate', 0.1),
            "enable_dropout_on_latent": checkpoint.get('model_config', {}).get('enable_dropout_on_latent', True),
            "enable_dropout_on_decoder": checkpoint.get('model_config', {}).get('enable_dropout_on_decoder', True),
        }
        
        # Override with any provided kwargs
        model_config.update(model_kwargs)
        
        print(f"🏗️  Initializing model with config: {model_config}")
        model = MultiModalHackVAE(**model_config)
        
        # Load state dict
        print(f"⚡ Loading model weights...")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict from checkpoint")
            if 'final_train_loss' in checkpoint:
                print(f"📊 Final training loss: {checkpoint['final_train_loss']:.4f}")
            if 'final_test_loss' in checkpoint:
                print(f"📊 Final test loss: {checkpoint['final_test_loss']:.4f}")
        else:
            # Fallback: try to load the checkpoint directly as state dict
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model state dict directly")
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully from local checkpoint")
        print(f"🎯 Model on device: {device}")
        print(f"🎯 Model in evaluation mode")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading from local checkpoint: {e}")
        raise


def load_model_from_huggingface(
    repo_name: str,
    revision_name: Optional[str] = None,
    token: Optional[str] = None,
    device: str = "cpu",
    **model_kwargs
) -> MultiModalHackVAE:
    """
    Load MultiModalHackVAE model from HuggingFace Hub
    
    Args:
        repo_name: HuggingFace repository name (e.g., "username/nethack-vae")
        revision_name: Specific revision to load (default is latest)
        token: HuggingFace token (if needed for private repos)
        device: Device to load the model on
        **model_kwargs: Additional arguments for model initialization (override config)
        
    Returns:
        Loaded MultiModalHackVAE model
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace Hub is required. Install with: pip install huggingface_hub")
    
    # Login if token is provided
    if token:
        login(token=token)
    
    api = HfApi()
    
    try:
        # Download config
        print(f"📥 Downloading model config from {repo_name}...")
        config_path = api.hf_hub_download(
            repo_id=repo_name,
            filename="config.json",
            repo_type="model",
            revision=revision_name
        )
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print(f"📋 Model config loaded: {config}")
        
        # Download model file
        print(f"📥 Downloading model weights from {repo_name}...")
        model_path = api.hf_hub_download(
            repo_id=repo_name,
            filename="pytorch_model.bin",
            repo_type="model",
            revision=revision_name
        )
        
        # Initialize model with config (allow kwargs to override)
        model_config = {
            "bInclude_glyph_bag": config.get("bInclude_glyph_bag", True),
            "bInclude_hero": config.get("bInclude_hero", True),
            "dropout_rate": config.get("dropout_rate", 0.1),
            "enable_dropout_on_latent": config.get("enable_dropout_on_latent", True),
            "enable_dropout_on_decoder": config.get("enable_dropout_on_decoder", True),
        }
        
        # Override with any provided kwargs
        model_config.update(model_kwargs)
        
        print(f"🏗️  Initializing model with config: {model_config}")
        model = MultiModalHackVAE(**model_config)
        
        # Load state dict
        print(f"⚡ Loading model weights...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict from checkpoint")
            if 'final_train_loss' in checkpoint:
                print(f"📊 Final training loss: {checkpoint['final_train_loss']:.4f}")
            if 'final_test_loss' in checkpoint:
                print(f"📊 Final test loss: {checkpoint['final_test_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model state dict directly")
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully from HuggingFace: {repo_name}")
        print(f"🎯 Model on device: {device}")
        print(f"🎯 Model in evaluation mode")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading from HuggingFace: {e}")
        raise


def create_visualization_demo(
    repo_name: str,
    train_dataset: Optional[List[Dict]] = None,
    test_dataset: Optional[List[Dict]] = None,
    revision_name: Optional[str] = None,
    token: Optional[str] = None,
    device: str = "cpu",
    num_samples: int = 4,
    max_latent_samples: int = 100,
    save_dir: str = "vae_analysis",
    random_sampling: bool = True,
    random_seed: Optional[int] = None,
    # VAE sampling parameters
    use_mean: bool = True,
    include_logits: bool = False,
    # Map sampling parameters
    map_temperature: float = 1.0,
    map_occ_thresh: float = 0.5,
    rare_occ_thresh: float = 0.5,
    hero_presence_thresh: float = 0.5,
    map_deterministic: bool = True,
    glyph_top_k: int = 0,
    glyph_top_p: float = 1.0,
    color_top_k: int = 0,
    color_top_p: float = 1.0,
    # Message sampling parameters
    msg_temperature: float = 1.0,
    msg_top_k: int = 0,
    msg_top_p: float = 1.0,
    msg_deterministic: bool = True,
    allow_eos: bool = True,
    forbid_eos_at_start: bool = True,
    allow_pad: bool = False
) -> Dict:
    """
    Complete demo function that loads a model from HuggingFace and creates visualizations
    
    Args:
        repo_name: HuggingFace repository name
        train_dataset: Training dataset from NetHackDataCollector (optional)
        test_dataset: Test dataset from NetHackDataCollector (optional)
        token: HuggingFace token (optional)
        device: Device to run on
        num_samples: Number of reconstruction samples
        max_latent_samples: Maximum samples for latent analysis
        save_dir: Directory to save results
        random_sampling: Whether to use random sampling for reconstruction visualization
        random_seed: Random seed for reproducible sampling
        
        # VAE sampling parameters
        use_mean: If True, use mean of latent distribution; if False, sample from it
        include_logits: Whether to include raw logits in output
        
        # Map sampling parameters (legacy parameters map to new ones)
        map_occ_thresh: Threshold for occupancy prediction
        rare_occ_thresh: Threshold for rare occupancy prediction
        hero_presence_thresh: Threshold for hero presence prediction
        map_temperature: Temperature for map sampling (legacy: temperature)
        glyph_top_k: Top-k filtering for glyph sampling (legacy: top_k)
        glyph_top_p: Top-p filtering for glyph sampling (legacy: top_p)
        map_deterministic: If True, use deterministic sampling for map
        color_top_k: Top-k filtering for color sampling
        color_top_p: Top-p filtering for color sampling
        
        # Message sampling parameters
        msg_temperature: Temperature for message token sampling
        msg_top_k: Top-k filtering for message sampling
        msg_top_p: Top-p filtering for message sampling
        msg_deterministic: If True, use deterministic sampling for messages
        allow_eos: Whether to allow end-of-sequence tokens
        forbid_eos_at_start: Whether to forbid EOS tokens at start
        allow_pad: Whether to allow padding tokens
        
    Returns:
        Dictionary with analysis results
    """
    # Validate inputs
    if train_dataset is None and test_dataset is None:
        raise ValueError("At least one of train_dataset or test_dataset must be provided")
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 Starting VAE Analysis Demo")
    print(f"📦 Repository: {repo_name}")
    print(f"🎯 Device: {device}")
    print(f"📁 Save directory: {save_dir}")
    print(f"🎲 Random sampling: {random_sampling}")
    if random_seed is not None:
        print(f"🌱 Random seed: {random_seed}")
    
    # Load model from HuggingFace with local fallback
    print(f"\n1️⃣ Loading model from HuggingFace...")
    model = None
    
    try:
        model = load_model_from_huggingface(repo_name, token=token, device=device, revision_name=revision_name)
    except Exception as e:
        print(f"⚠️  Failed to load from HuggingFace: {e}")
        print(f"🔄 Attempting to load from local checkpoints...")
        
        # Try to find the latest local checkpoint
        checkpoint_dir = "checkpoints"
        local_checkpoint_path = None
        
        if os.path.exists(checkpoint_dir):
            # Find the latest checkpoint file
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoint_files:
                # Sort by modification time, latest first
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                local_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                print(f"📁 Found latest checkpoint: {local_checkpoint_path}")
            else:
                print(f"❌ No checkpoint files found in {checkpoint_dir}")
        
        # Also check for a saved model file
        if local_checkpoint_path is None:
            potential_paths = [
                "models/nethack-vae.pth",
                "nethack-vae.pth",
                "model.pth"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    local_checkpoint_path = path
                    print(f"📁 Found saved model: {local_checkpoint_path}")
                    break
        
        if local_checkpoint_path is not None:
            try:
                model = load_model_from_local(local_checkpoint_path, device=device)
                print(f"✅ Successfully loaded model from local checkpoint")
            except Exception as local_e:
                print(f"❌ Failed to load from local checkpoint: {local_e}")
                raise RuntimeError(f"Failed to load model from both HuggingFace ({e}) and local checkpoint ({local_e})")
        else:
            print(f"❌ No local checkpoints found")
            raise RuntimeError(f"Failed to load model from HuggingFace ({e}) and no local checkpoints available")

    results = {'model': model, 'save_dir': save_dir}
    
    # Create TTY reconstructions for available datasets
    if train_dataset is not None:
        print(f"\n2️⃣ Creating TTY reconstruction visualizations for TRAINING dataset...")
        train_save_path = "train_recon_comparison.md"
        train_recon_results = visualize_reconstructions(
            model, train_dataset, device, 
            num_samples=num_samples, 
            out_dir=save_dir, 
            save_path=train_save_path,
            img_file_prefix="train_",
            random_sampling=random_sampling,
            dataset_name="Training",
            # VAE sampling parameters
            use_mean=use_mean,
            include_logits=include_logits,
            # Map sampling parameters (map legacy params)
            map_temperature=map_temperature,  # Legacy: temperature -> map_temperature
            map_occ_thresh=map_occ_thresh,
            rare_occ_thresh=rare_occ_thresh,
            hero_presence_thresh=hero_presence_thresh,
            map_deterministic=map_deterministic,
            glyph_top_k=glyph_top_k,  # Legacy: top_k -> glyph_top_k
            glyph_top_p=glyph_top_p,  # Legacy: top_p -> glyph_top_p
            color_top_k=color_top_k,
            color_top_p=color_top_p,
            # Message sampling parameters
            msg_temperature=msg_temperature,
            msg_top_k=msg_top_k,
            msg_top_p=msg_top_p,
            msg_deterministic=msg_deterministic,
            allow_eos=allow_eos,
            forbid_eos_at_start=forbid_eos_at_start,
            allow_pad=allow_pad
        )
        results['train_reconstruction_path'] = os.path.join(save_dir, train_save_path)
        results['train_reconstruction_results'] = train_recon_results
    
    if test_dataset is not None:
        print(f"\n2️⃣ Creating TTY reconstruction visualizations for TESTING dataset...")
        test_save_path = "test_recon_comparison.md"
        test_recon_results = visualize_reconstructions(
            model, test_dataset, device, 
            num_samples=num_samples, 
            out_dir=save_dir, 
            save_path=test_save_path,
            img_file_prefix="test_",
            random_sampling=random_sampling,
            dataset_name="Testing",
            # VAE sampling parameters
            use_mean=use_mean,
            include_logits=include_logits,
            # Map sampling parameters (map legacy params)
            map_temperature=map_temperature,  # Legacy: temperature -> map_temperature
            map_occ_thresh=map_occ_thresh,
            rare_occ_thresh=rare_occ_thresh,
            hero_presence_thresh=hero_presence_thresh,
            map_deterministic=map_deterministic,
            glyph_top_k=glyph_top_k,  # Legacy: top_k -> glyph_top_k
            glyph_top_p=glyph_top_p,  # Legacy: top_p -> glyph_top_p
            color_top_k=color_top_k,
            color_top_p=color_top_p,
            # Message sampling parameters
            msg_temperature=msg_temperature,
            msg_top_k=msg_top_k,
            msg_top_p=msg_top_p,
            msg_deterministic=msg_deterministic,
            allow_eos=allow_eos,
            forbid_eos_at_start=forbid_eos_at_start,
            allow_pad=allow_pad
        )
        results['test_reconstruction_path'] = os.path.join(save_dir, test_save_path)
        results['test_reconstruction_results'] = test_recon_results

    # Analyze latent space (use combined dataset or available one)
    print(f"\n3️⃣ Analyzing latent space...")
    
    # Combine datasets for latent analysis or use what's available
    analysis_datasets = []
    dataset_labels = []
    
    if train_dataset is not None:
        analysis_datasets.extend(train_dataset)
        dataset_labels.extend(['train'] * len(train_dataset))
    
    if test_dataset is not None:
        analysis_datasets.extend(test_dataset)
        dataset_labels.extend(['test'] * len(test_dataset))
    
    latent_path = os.path.join(save_dir, "latent_analysis.png")
    latent_analysis = analyze_latent_space(
        model, analysis_datasets, device, 
        save_path=latent_path, 
        max_samples=max_latent_samples,
        dataset_labels=dataset_labels
    )
    
    results['latent_analysis_path'] = latent_path
    results['latent_analysis'] = latent_analysis
    
    print(f"\n✅ Analysis complete! Results saved to: {save_dir}")
    if train_dataset is not None:
        print(f"📄 Training TTY reconstructions: {results['train_reconstruction_path']}")
    if test_dataset is not None:
        print(f"📄 Testing TTY reconstructions: {results['test_reconstruction_path']}")
    print(f"📊 Latent analysis plot: {latent_path}")
    
    return results

def upload_training_artifacts_to_huggingface(
    repo_name: str,
    train_losses: List[float],
    test_losses: List[float],
    training_config: Dict,
    token: Optional[str] = None,
    plots_dir: str = "training_plots"
) -> None:
    """
    Upload training artifacts (losses, plots, config) to HuggingFace
    
    Args:
        repo_name: HuggingFace repository name
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
        training_config: Dictionary with training configuration
        token: HuggingFace token
        plots_dir: Directory name for plots in the repo
    """
    if not HF_AVAILABLE:
        print("⚠️  HuggingFace Hub not available, skipping artifact upload")
        return
    
    if token:
        login(token=token)
    
    api = HfApi()
    
    try:
        # Create training plots
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss improvement plot
        plt.subplot(1, 2, 2)
        if len(train_losses) > 1:
            train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
            test_improvement = [(test_losses[0] - loss) / test_losses[0] * 100 for loss in test_losses]
            plt.plot(epochs, train_improvement, 'b-', label='Training Improvement (%)', linewidth=2)
            plt.plot(epochs, test_improvement, 'r-', label='Test Improvement (%)', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Improvement (%)')
            plt.title('Loss Improvement Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save training data
        training_data = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "config": training_config,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_test_loss": test_losses[-1] if test_losses else None,
            "total_epochs": len(train_losses),
            "best_train_loss": min(train_losses) if train_losses else None,
            "best_test_loss": min(test_losses) if test_losses else None
        }
        
        with open("training_data.json", "w") as f:
            json.dump(training_data, f, indent=2)
        
        # Upload files
        api.upload_file(
            path_or_fileobj="training_curves.png",
            path_in_repo=f"{plots_dir}/training_curves.png",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add training curves"
        )
        
        api.upload_file(
            path_or_fileobj="training_data.json",
            path_in_repo="training_data.json",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add training data"
        )
        
        # Clean up
        os.remove("training_curves.png")
        os.remove("training_data.json")
        
        print(f"✅ Training artifacts uploaded to {repo_name}")
        
    except Exception as e:
        print(f"❌ Error uploading training artifacts: {e}")


def create_model_demo_notebook(repo_name: str, save_path: str = "demo_notebook.ipynb") -> None:
    """
    Create a Jupyter notebook demonstrating model usage
    
    Args:
        repo_name: HuggingFace repository name
        save_path: Path to save the notebook
    """
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# MultiModalHackVAE Demo\n\n",
                    f"This notebook demonstrates how to use the MultiModalHackVAE model from {repo_name}.\n\n",
                    "## Installation\n\n",
                    "```bash\n",
                    "pip install torch transformers huggingface_hub\n",
                    "# For NetHack environment (optional):\n",
                    "pip install nle\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import torch\n",
                    "import numpy as np\n",
                    "from huggingface_hub import hf_hub_download\n",
                    "import json\n",
                    "\n",
                    "# Load model config\n",
                    f"config_path = hf_hub_download(repo_id='{repo_name}', filename='config.json')\n",
                    "with open(config_path, 'r') as f:\n",
                    "    config = json.load(f)\n",
                    "\n",
                    "print('Model Configuration:')\n",
                    "for key, value in config.items():\n",
                    "    print(f'  {key}: {value}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load the model (you'll need to import your model class)\n",
                    "# from your_package import MultiModalHackVAE\n",
                    "# model = load_model_from_huggingface('{repo_name}')\n",
                    "\n",
                    "# Example synthetic data\n",
                    "batch_size = 1\n",
                    "game_chars = torch.randint(32, 127, (batch_size, 21, 79))\n",
                    "game_colors = torch.randint(0, 16, (batch_size, 21, 79))\n",
                    "blstats = torch.randn(batch_size, 27)\n",
                    "msg_tokens = torch.randint(0, 128, (batch_size, 256))\n",
                    "hero_info = torch.randint(0, 10, (batch_size, 4))\n",
                    "\n",
                    "print('Synthetic data shapes:')\n",
                    "print(f'  game_chars: {game_chars.shape}')\n",
                    "print(f'  game_colors: {game_colors.shape}')\n",
                    "print(f'  blstats: {blstats.shape}')\n",
                    "print(f'  msg_tokens: {msg_tokens.shape}')\n",
                    "print(f'  hero_info: {hero_info.shape}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Encode to latent space\n",
                    "# with torch.no_grad():\n",
                    "#     output = model(\n",
                    "#         glyph_chars=game_chars,\n",
                    "#         glyph_colors=game_colors,\n",
                    "#         blstats=blstats,\n",
                    "#         msg_tokens=msg_tokens,\n",
                    "#         hero_info=hero_info\n",
                    "#     )\n",
                    "#     \n",
                    "#     latent_mean = output['mu']\n",
                    "#     latent_logvar = output['logvar']\n",
                    "#     lowrank_factors = output['lowrank_factors']\n",
                    "#     \n",
                    "#     print(f'Latent representation shape: {latent_mean.shape}')\n",
                    "#     print(f'Latent mean: {latent_mean[0][:5].tolist()}')\n",
                    "\n",
                    "print('Model inference example (uncomment when model is available)')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8+"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(save_path, "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"📓 Demo notebook created: {save_path}")


def analyze_glyph_char_color_pairs(
    dataset: List[Dict],
    top_k: int = 50,
    save_dir: str = "bin_count_analysis",
    save_plot: bool = True,
    show_ascii_chars: bool = True,
    save_complete_data: bool = True
) -> Dict:
    """
    Analyze the distribution of (glyph_char, glyph_color) pairs in the dataset.
    
    Args:
        dataset: List of data batches from NetHackDataCollector
        top_k: Number of top pairs to display
        save_dir: Directory to save analysis results
        save_plot: Whether to save the plot
        show_ascii_chars: Whether to show ASCII character representations
        save_complete_data: Whether to save complete count data to JSON
        
    Returns:
        Dictionary with analysis results
    """
    print(f"🔍 Starting glyph (char, color) pair analysis...")
    print(f"📊 Dataset size: {len(dataset)} batches")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Counter for (char, color) pairs
    pair_counter = Counter()
    total_cells = 0
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(dataset, desc="Processing batches")):
        game_chars = batch['game_chars']  # Shape: (num_games, num_time, 21, 79)
        game_colors = batch['game_colors']  # Shape: (num_games, num_time, 21, 79)
        
        # Flatten the spatial and temporal dimensions
        chars_flat = game_chars.flatten()  # All character codes
        colors_flat = game_colors.flatten()  # All color codes
        
        # Count pairs
        for char, color in zip(chars_flat.tolist(), colors_flat.tolist()):
            pair_counter[(char, color)] += 1
            total_cells += 1
    
    print(f"📈 Total cells analyzed: {total_cells:,}")
    print(f"🎨 Unique (char, color) pairs found: {len(pair_counter):,}")
    
    # Save complete count data to JSON if requested
    if save_complete_data:
        # Create readable format for pairs
        readable_pairs = {}
        for (char, color), count in pair_counter.items():
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_key = f"({char},{color})"
            readable_pairs[pair_key] = {
                'char_code': char,
                'color_code': color,
                'ascii_char': ascii_repr,
                'count': count,
                'percentage': (count / total_cells) * 100
            }
        
        # Create readable format for characters
        char_counter = Counter()
        for (char, color), count in pair_counter.items():
            char_counter[char] += count
        
        readable_chars = {}
        for char, count in char_counter.items():
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            readable_chars[str(char)] = {
                'char_code': char,
                'ascii_char': ascii_repr,
                'total_count': count,
                'percentage': (count / total_cells) * 100
            }
        
        complete_data = {
            'total_cells': total_cells,
            'unique_pairs': len(pair_counter),
            'pair_counts': readable_pairs,
            'char_counts': readable_chars,
            'analysis_metadata': {
                'dataset_size': len(dataset),
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'glyph_char_color_pairs'
            }
        }
        
        # Save complete data
        complete_data_path = os.path.join(save_dir, "complete_bin_counts.json")
        with open(complete_data_path, 'w') as f:
            json.dump(complete_data, f, indent=2)
        print(f"💾 Complete count data saved to: {complete_data_path}")
    
    # Get top k pairs (excluding space character pairs)
    filtered_pairs = [(key, count) for key, count in pair_counter.items() if key[0] != 32]
    top_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Convert to readable format
    pair_data = []
    for (char, color), count in top_pairs:
        ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
        percentage = (count / total_cells) * 100
        pair_data.append({
            'char_code': char,
            'color_code': color,
            'ascii_char': ascii_repr,
            'count': count,
            'percentage': percentage
        })
    
    # Print top pairs
    print(f"\n🏆 Top {top_k} (char, color) pairs (excluding spaces):")
    print("-" * 80)
    if show_ascii_chars:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'ASCII':<8} {'Count':<12} {'Percentage':<10}")
    else:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'Count':<12} {'Percentage':<10}")
    print("-" * 80)
    
    for i, data in enumerate(pair_data):
        if show_ascii_chars:
            ascii_str = f"'{data['ascii_char']}'"
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{ascii_str:<8} {data['count']:<12,} {data['percentage']:<10.2f}%")
        else:
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{data['count']:<12,} {data['percentage']:<10.2f}%")
    
    # Create visualization
    if save_plot and len(top_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Bar chart of top pairs
        pair_labels = []
        counts = []
        colors_for_plot = []
        
        # Color mapping for NetHack colors (0-15)
        nethack_colors = [
            '#000000',  # 0: black
            '#800000',  # 1: red  
            '#008000',  # 2: green
            '#808000',  # 3: yellow
            '#000080',  # 4: blue
            '#800080',  # 5: magenta
            '#008080',  # 6: cyan
            '#C0C0C0',  # 7: white
            '#808080',  # 8: gray
            '#ff0000',  # 9: bright red
            '#00ff00',  # 10: bright green
            '#ffff00',  # 11: bright yellow
            '#0000ff',  # 12: bright blue
            '#ff00ff',  # 13: bright magenta
            '#00ffff',  # 14: bright cyan
            '#ffffff'   # 15: bright white
        ]
        
        for (char, color), count in top_pairs:
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_labels.append(f"'{ascii_repr}' ({char}, {color})")
            counts.append(count)
            # Use actual NetHack color if available, otherwise use default
            if 0 <= color < len(nethack_colors):
                colors_for_plot.append(nethack_colors[color])
            else:
                colors_for_plot.append('#808080')  # Default gray
        
        bars = ax1.bar(range(len(counts)), counts, color=colors_for_plot, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('(Character, Color) Pairs')
        ax1.set_ylabel('Count (log scale)')
        ax1.set_yscale('log')
        ax1.set_title(f'Top {top_k} Most Frequent (Glyph Char, Glyph Color) Pairs (Excluding Spaces)')
        ax1.set_xticks(range(len(pair_labels)))
        ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Character distribution (top chars only, excluding space)
        char_counter = Counter()
        for (char, color), count in pair_counter.items():
            if char != 32:  # Exclude space character
                char_counter[char] += count
        
        top_chars = char_counter.most_common(20)  # Top 20 characters
        char_codes = [char for char, _ in top_chars]
        char_counts = [count for _, count in top_chars]
        char_labels = [f"'{chr(char)}' ({char})" if 32 <= char <= 126 else f"\\x{char:02x} ({char})" 
                      for char in char_codes]
        
        bars2 = ax2.bar(range(len(char_counts)), char_counts, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Character Codes')
        ax2.set_ylabel('Total Count (log scale)')
        ax2.set_yscale('log')
        ax2.set_title('Top 20 Most Frequent Characters (All Colors Combined, Excluding Spaces)')
        ax2.set_xticks(range(len(char_labels)))
        ax2.set_xticklabels(char_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars2, char_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(save_dir, f"glyph_char_color_analysis_top_{top_k}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    # Save detailed results to JSON
    results = {
        'total_cells': total_cells,
        'unique_pairs': len(pair_counter),
        'top_pairs': pair_data,
        'analysis_params': {
            'top_k': top_k,
            'dataset_size': len(dataset),
            'show_ascii_chars': show_ascii_chars
        }
    }
    
    if save_plot:
        results_path = os.path.join(save_dir, "glyph_analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_path}")
    
    return results


def plot_glyph_char_color_pairs_from_saved(
    data_path: str,
    top_k: int = 50,
    save_dir: str = None,
    save_plot: bool = True,
    show_ascii_chars: bool = True,
    exclude_space: bool = True
) -> Dict:
    """
    Load saved bin count data and create visualizations.
    
    Args:
        data_path: Path to the saved complete_bin_counts.json file
        top_k: Number of top pairs to display
        save_dir: Directory to save plots (if None, uses directory of data_path)
        save_plot: Whether to save the plot
        show_ascii_chars: Whether to show ASCII character representations
        exclude_space: Whether to exclude space character (ASCII 32) from analysis
        
    Returns:
        Dictionary with analysis results
    """
    print(f"📥 Loading saved bin count data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Saved data file not found: {data_path}")
    
    # Load the complete data
    with open(data_path, 'r') as f:
        complete_data = json.load(f)
    
    total_cells = complete_data['total_cells']
    unique_pairs = complete_data['unique_pairs']
    
    print(f"📈 Loaded data: {total_cells:,} total cells, {unique_pairs:,} unique pairs")
    
    # Convert pair_counts back to Counter format
    pair_counter = Counter()
    for pair_key, pair_data in complete_data['pair_counts'].items():
        char = pair_data['char_code']
        color = pair_data['color_code']
        count = pair_data['count']
        pair_counter[(char, color)] = count
    
    # Set save directory
    if save_dir is None:
        save_dir = os.path.dirname(data_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter pairs if excluding space
    if exclude_space:
        filtered_pairs = [(key, count) for key, count in pair_counter.items() if key[0] != 32]
        print(f"🚫 Excluding space character pairs")
    else:
        filtered_pairs = list(pair_counter.items())
    
    # Get top k pairs
    top_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Convert to readable format
    pair_data = []
    for (char, color), count in top_pairs:
        ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
        percentage = (count / total_cells) * 100
        pair_data.append({
            'char_code': char,
            'color_code': color,
            'ascii_char': ascii_repr,
            'count': count,
            'percentage': percentage
        })
    
    # Print top pairs
    exclude_text = " (excluding spaces)" if exclude_space else ""
    print(f"\n🏆 Top {top_k} (char, color) pairs{exclude_text}:")
    print("-" * 80)
    if show_ascii_chars:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'ASCII':<8} {'Count':<12} {'Percentage':<10}")
    else:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'Count':<12} {'Percentage':<10}")
    print("-" * 80)
    
    for i, data in enumerate(pair_data):
        if show_ascii_chars:
            ascii_str = f"'{data['ascii_char']}'"
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{ascii_str:<8} {data['count']:<12,} {data['percentage']:<10.2f}%")
        else:
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{data['count']:<12,} {data['percentage']:<10.2f}%")
    
    # Create visualization
    if save_plot and len(top_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Bar chart of top pairs
        pair_labels = []
        counts = []
        colors_for_plot = []
        
        # Color mapping for NetHack colors (0-15)
        nethack_colors = [
            '#000000',  # 0: black
            '#800000',  # 1: red  
            '#008000',  # 2: green
            '#808000',  # 3: yellow
            '#000080',  # 4: blue
            '#800080',  # 5: magenta
            '#008080',  # 6: cyan
            '#C0C0C0',  # 7: white
            '#808080',  # 8: gray
            '#ff0000',  # 9: bright red
            '#00ff00',  # 10: bright green
            '#ffff00',  # 11: bright yellow
            '#0000ff',  # 12: bright blue
            '#ff00ff',  # 13: bright magenta
            '#00ffff',  # 14: bright cyan
            '#ffffff'   # 15: bright white
        ]
        
        for (char, color), count in top_pairs:
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_labels.append(f"'{ascii_repr}' ({char}, {color})")
            counts.append(count)
            # Use actual NetHack color if available, otherwise use default
            if 0 <= color < len(nethack_colors):
                colors_for_plot.append(nethack_colors[color])
            else:
                colors_for_plot.append('#808080')  # Default gray
        
        bars = ax1.bar(range(len(counts)), counts, color=colors_for_plot, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('(Character, Color) Pairs')
        ax1.set_ylabel('Count (log scale)')
        ax1.set_yscale('log')
        title_suffix = " (Excluding Spaces)" if exclude_space else ""
        ax1.set_title(f'Top {top_k} Most Frequent (Glyph Char, Glyph Color) Pairs{title_suffix}')
        ax1.set_xticks(range(len(pair_labels)))
        ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Character distribution (top chars only, optionally excluding space)
        char_counter = Counter()
        for char_str, char_data in complete_data['char_counts'].items():
            char = char_data['char_code']
            count = char_data['total_count']
            if not exclude_space or char != 32:
                char_counter[char] = count
        
        top_chars = char_counter.most_common(20)  # Top 20 characters
        char_codes = [char for char, _ in top_chars]
        char_counts = [count for _, count in top_chars]
        char_labels = [f"'{chr(char)}' ({char})" if 32 <= char <= 126 else f"\\x{char:02x} ({char})" 
                      for char in char_codes]
        
        bars2 = ax2.bar(range(len(char_counts)), char_counts, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Character Codes')
        ax2.set_ylabel('Total Count (log scale)')
        ax2.set_yscale('log')
        char_title_suffix = " (Excluding Spaces)" if exclude_space else ""
        ax2.set_title(f'Top 20 Most Frequent Characters (All Colors Combined{char_title_suffix})')
        ax2.set_xticks(range(len(char_labels)))
        ax2.set_xticklabels(char_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars2, char_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_suffix = "_no_space" if exclude_space else ""
            plot_path = os.path.join(save_dir, f"glyph_char_color_analysis_top_{top_k}{plot_suffix}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    # Return results
    results = {
        'total_cells': total_cells,
        'unique_pairs': unique_pairs,
        'top_pairs': pair_data,
        'analysis_params': {
            'top_k': top_k,
            'exclude_space': exclude_space,
            'show_ascii_chars': show_ascii_chars,
            'data_source': data_path
        },
        'metadata': complete_data.get('analysis_metadata', {})
    }
    
    return results


if __name__ == "__main__":
    
    train_file = "nld-aa-training"
    test_file = "nld-aa-testing"
    data_cache_dir = "data_cache"
    batch_size = 32
    sequence_size = 32
    max_training_batches = 100
    max_testing_batches = 20
    train_cache_file = os.path.join(data_cache_dir, f"{train_file}_b{batch_size}_s{sequence_size}_m{max_training_batches}.pt")
    test_cache_file = os.path.join(data_cache_dir, f"{test_file}_b{batch_size}_s{sequence_size}_m{max_testing_batches}.pt")
    
    
    if len(sys.argv) > 1 and sys.argv[1] == "vae_analysis":
        # Demo mode: python train.py vae_analysis <repo_name> [revision_name]
        repo_name = sys.argv[2] if len(sys.argv) > 2 else "CatkinChen/nethack-vae"
        revision_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        print(f"🚀 Running VAE Analysis Demo")
        print(f"📦 Repository: {repo_name}")
        
        # Create both training and test data
        print(f"📊 Preparing training and test data...")
        
        collector = NetHackDataCollector('ttyrecs.db')
        adapter = BLStatsAdapter()
        
        # Load training dataset
        print(f"📊 Loading training dataset...")
        train_dataset = collector.collect_or_load_data(
            dataset_name=train_file,
            adapter=adapter,
            save_path=train_cache_file,
            max_batches=max_training_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=False
        )
        
        # Load test dataset  
        print(f"📊 Loading test dataset...")
        test_dataset = collector.collect_or_load_data(
            dataset_name=test_file,
            adapter=adapter,
            save_path=test_cache_file,
            max_batches=max_testing_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=False
        )
        
        # Run the complete analysis on both datasets
        try:
            results = create_visualization_demo(
                repo_name=repo_name,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                revision_name=revision_name,
                device="cpu",  # Use CPU for demo
                num_samples=10,
                max_latent_samples=1000,  # More samples since we have both datasets
                save_dir="vae_analysis",
                random_sampling=True,  # Enable random sampling
                random_seed=50,  # For reproducible results
                use_mean=True,  # Use mean for latent space
                map_occ_thresh=0.5,
                rare_occ_thresh=0.5,
                hero_presence_thresh=0.2,
                map_deterministic=True  # Use deterministic sampling for maps
            )
            print(f"✅ Demo completed successfully!")
            print(f"📁 Results saved to: {results['save_dir']}")
            print(f"📊 Training dataset: {len(train_dataset)} batches")
            print(f"📊 Test dataset: {len(test_dataset)} batches")
            
            # Print detailed results
            if 'train_reconstruction_results' in results:
                print(f"🎨 Training reconstructions: {results['train_reconstruction_results']['num_samples']} samples")
            if 'test_reconstruction_results' in results:
                print(f"🎨 Test reconstructions: {results['test_reconstruction_results']['num_samples']} samples")
            if 'latent_analysis' in results:
                print(f"🧠 Latent analysis: {len(results['latent_analysis']['latent_vectors'])} total samples analyzed")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"💡 Make sure the repository exists and is accessible")
            print(f"💡 You can create synthetic data for testing by setting repo_name to a local path")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "collect_data":
        # test collecting and saving data
        collector = NetHackDataCollector('ttyrecs.db')
        adapter = BLStatsAdapter()
        train_dataset = collector.collect_or_load_data(
            dataset_name=train_file,
            adapter=adapter,
            save_path=train_cache_file,
            max_batches=max_training_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=True
        )
        test_dataset = collector.collect_or_load_data(
            dataset_name=test_file,
            adapter=adapter,
            save_path=test_cache_file,
            max_batches=max_testing_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=True
        )
        
        print(f"✅ Data collection completed!")
        print(f"   📊 Train batches: {len(train_dataset)}")
        print(f"   📊 Test batches: {len(test_dataset)}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "bin_count_analysis":
        # Bin count analysis mode: python train.py bin_count_analysis [top_k] [dataset_type]
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        dataset_type = sys.argv[3] if len(sys.argv) > 3 else "both"  # "train", "test", or "both"
        
        print(f"🔍 Running Glyph (Char, Color) Bin Count Analysis")
        print(f"📊 Top K pairs to analyze: {top_k}")
        print(f"📁 Dataset type: {dataset_type}")
        
        # Prepare data collector
        collector = NetHackDataCollector('ttyrecs.db')
        adapter = BLStatsAdapter()
        
        datasets_to_analyze = []
        dataset_names = []
        
        if dataset_type in ["train", "both"]:
            print(f"📊 Loading training dataset...")
            train_dataset = collector.collect_or_load_data(
                dataset_name=train_file,
                adapter=adapter,
                save_path=train_cache_file,
                max_batches=max_training_batches,
                batch_size=batch_size,
                seq_length=sequence_size,
                force_recollect=False
            )
            datasets_to_analyze.append(train_dataset)
            dataset_names.append("train")
        
        if dataset_type in ["test", "both"]:
            print(f"📊 Loading test dataset...")
            test_dataset = collector.collect_or_load_data(
                dataset_name=test_file,
                adapter=adapter,
                save_path=test_cache_file,
                max_batches=max_testing_batches,
                batch_size=batch_size,
                seq_length=sequence_size,
                force_recollect=False
            )
            datasets_to_analyze.append(test_dataset)
            dataset_names.append("test")
        
        # Run analysis on each dataset
        for dataset, dataset_name in zip(datasets_to_analyze, dataset_names):
            print(f"\n🔬 Analyzing {dataset_name} dataset...")
            save_dir = f"bin_count_analysis/{dataset_name}"
            
            try:
                results = analyze_glyph_char_color_pairs(
                    dataset=dataset,
                    top_k=top_k,
                    save_dir=save_dir,
                    save_plot=True,
                    show_ascii_chars=True,
                    save_complete_data=True
                )
                
                print(f"✅ {dataset_name.capitalize()} analysis completed!")
                print(f"📁 Results saved to: {save_dir}")
                print(f"📊 Total cells: {results['total_cells']:,}")
                print(f"🎨 Unique pairs: {results['unique_pairs']:,}")
                
            except Exception as e:
                print(f"❌ {dataset_name.capitalize()} analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        # If analyzing both datasets, create a combined analysis
        if dataset_type == "both" and len(datasets_to_analyze) == 2:
            print(f"\n🔗 Creating combined analysis...")
            combined_dataset = datasets_to_analyze[0] + datasets_to_analyze[1]
            save_dir = "bin_count_analysis/combined"
            
            try:
                results = analyze_glyph_char_color_pairs(
                    dataset=combined_dataset,
                    top_k=top_k,
                    save_dir=save_dir,
                    save_plot=True,
                    show_ascii_chars=True,
                    save_complete_data=True
                )
                
                print(f"✅ Combined analysis completed!")
                print(f"📁 Results saved to: {save_dir}")
                print(f"📊 Total cells: {results['total_cells']:,}")
                print(f"🎨 Unique pairs: {results['unique_pairs']:,}")
                
            except Exception as e:
                print(f"❌ Combined analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 Bin count analysis completed!")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "plot_bin_count":
        # Plot from saved data mode: python train.py plot_bin_count <data_path> [top_k] [exclude_space]
        if len(sys.argv) < 3:
            print("❌ Usage: python train.py plot_bin_count <data_path> [top_k] [exclude_space]")
            print("   Example: python train.py plot_bin_count bin_count_analysis/train/complete_bin_counts.json 30 true")
            sys.exit(1)
        
        data_path = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        exclude_space = sys.argv[4].lower() in ['true', '1', 'yes'] if len(sys.argv) > 4 else True
        
        print(f"📊 Plotting bin count analysis from saved data")
        print(f"📁 Data path: {data_path}")
        print(f"📊 Top K pairs: {top_k}")
        print(f"🚫 Exclude spaces: {exclude_space}")
        
        try:
            results = plot_glyph_char_color_pairs_from_saved(
                data_path=data_path,
                top_k=top_k,
                save_plot=True,
                show_ascii_chars=True,
                exclude_space=exclude_space
            )
            
            print(f"✅ Plot generation completed!")
            print(f"📊 Total cells: {results['total_cells']:,}")
            print(f"🎨 Unique pairs: {results['unique_pairs']:,}")
            print(f"📈 Showing top {len(results['top_pairs'])} pairs")
            
        except Exception as e:
            print(f"❌ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        hf_model_card_data = {
            "author": "Xu Chen",
            "description": "Advanced NetHack VAE",
            "tags": ["nethack", "reinforcement-learning", "multimodal", "world-modeling", "vae"],
            "use_cases": [
                "Game state representation learning",
                "RL agent state abstraction",
                "NetHack gameplay analysis"
            ],
        }
        
        print(f"\n🧪 Starting train_multimodalhack_vae...")
        model, train_losses, test_losses = train_multimodalhack_vae(
            train_file=train_file,
            test_file=test_file,
            epochs=15,          
            batch_size=batch_size,
            sequence_size=sequence_size,    
            max_learning_rate=1e-3,
            training_batches=max_training_batches,
            testing_batches=max_testing_batches,
            max_training_batches=max_training_batches,
            max_testing_batches=max_testing_batches,
            save_path="models/nethack-vae.pth",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_bf16=False,  # Enable BF16 mixed precision training
            data_cache_dir="data_cache",
            force_recollect=False,  # Use the data we just collected
            shuffle_batches=True,  # Shuffle training batches each epoch for better training
            shuffle_within_batch=True,  # Shuffle within each batch for more variety
            initial_mi_beta=0.0,
            final_mi_beta=0.0,
            mi_beta_shape='constant',
            initial_tc_beta=5.0,
            final_tc_beta=5.0,
            tc_beta_shape='constant',
            initial_dw_beta=0.02,
            final_dw_beta=4.0,
            dw_beta_shape='linear',
            custom_kl_beta_function = lambda init, end, progress: init + (end - init) * min(progress, 0.2) * 5.0, 
            warmup_epoch_ratio = 0.2,
            free_bits=0.15,
            focal_loss_alpha=0.75,
            focal_loss_gamma=2.0,

            # Dropout and regularization settings
            dropout_rate=0.1,  # Set to 0.1 for mild regularization
            enable_dropout_on_latent=True,
            enable_dropout_on_decoder=True,
            
            # Early stopping settings
            early_stopping = True,
            early_stopping_patience = 3,
            early_stopping_min_delta = 0.01,

            # Enable checkpointing
            save_checkpoints=True,
            checkpoint_dir="checkpoints",
            save_every_n_epochs=1,
            keep_last_n_checkpoints=2,
            
            # Wandb integration example
            use_wandb=True,
            wandb_project="nethack-vae",
            wandb_entity="xchen-catkin-ucl",  # Replace with your wandb username
            wandb_run_name=f"vae-test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            wandb_tags=["nethack", "vae"],
            wandb_notes="Full VAE training run",
            log_every_n_steps=5,  # Log every 5 steps
            log_model_architecture=True,
            log_gradients=True,
            
            # HuggingFace integration example
            upload_to_hf=True, 
            hf_repo_name="CatkinChen/nethack-vae",
            hf_upload_directly=True,  # Upload directly without extra local save
            hf_upload_checkpoints=True,  # Also upload checkpoints
            hf_model_card_data=hf_model_card_data
        )

        print(f"\n🎉 Full VAE training run completed successfully!")
        print(f"   📈 Train losses: {train_losses}")
        print(f"   📈 Test losses: {test_losses}")

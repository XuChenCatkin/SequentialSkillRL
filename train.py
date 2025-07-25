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

# Add src to path for importing the existing MiniHackVAE
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import MiniHackVAE, vae_loss
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
        
    def detect_inventory_screen(self, tty_chars: np.ndarray) -> Dict:
        """
        Detect if current screen shows inventory and extract items
        
        Args:
            tty_chars: TTY characters array (24, 80)
            
        Returns:
            Dict with inventory info
        """
        # Convert to text
        screen_lines = []
        for row in range(tty_chars.shape[0]):
            line = ''.join([chr(c) if 32 <= c <= 126 else ' ' for c in tty_chars[row]])
            screen_lines.append(line.rstrip())
        
        screen_text = ' '.join(screen_lines).lower()
        
        # Detect inventory indicators
        inventory_indicators = [
            'you are carrying:',
            'you are not carrying anything',
            'inventory:',
            'your possessions:',
            'items carried:',
        ]
        
        is_inventory = any(indicator in screen_text for indicator in inventory_indicators)
        
        # Extract items in a) b) c) format
        items = []
        item_pattern_count = 0
        
        for i, line in enumerate(screen_lines):
            line = line.strip()
            if len(line) > 2 and line[1] == ')' and line[0].isalpha():
                items.append({
                    'slot': line[0],
                    'description': line[2:].strip(),
                    'line_number': i
                })
                item_pattern_count += 1
        
        # If we have multiple item patterns, likely inventory
        if item_pattern_count >= 2:
            is_inventory = True
        
        return {
            'is_inventory': is_inventory,
            'is_empty': 'not carrying anything' in screen_text,
            'items': items,
            'item_count': len(items),
            'inventory_features': self._extract_inventory_features(items)
        }
    
    def _extract_inventory_features(self, items: List[Dict]) -> Dict:
        """Extract numerical features from inventory items"""
        features = {
            'total_items': len(items),
            'equipped_items': 0,
            'food_items': 0,
            'weapon_items': 0,
            'armor_items': 0,
            'potion_items': 0,
            'scroll_items': 0,
            'tool_items': 0,
            'ring_items': 0,
            'wand_items': 0,
        }
        
        for item in items:
            desc_lower = item['description'].lower()
            
            # Count equipped items
            if 'being worn' in desc_lower or 'wielded' in desc_lower:
                features['equipped_items'] += 1
            
            # Categorize items
            if any(food in desc_lower for food in ['food', 'ration', 'apple', 'orange', 'banana']):
                features['food_items'] += 1
            elif any(weapon in desc_lower for weapon in ['sword', 'dagger', 'bow', 'arrow', 'spear', 'mace']):
                features['weapon_items'] += 1
            elif any(armor in desc_lower for armor in ['armor', 'helm', 'shield', 'boots', 'gloves', 'cloak']):
                features['armor_items'] += 1
            elif 'potion' in desc_lower:
                features['potion_items'] += 1
            elif 'scroll' in desc_lower:
                features['scroll_items'] += 1
            elif any(tool in desc_lower for tool in ['tool', 'key', 'lamp', 'pick-axe']):
                features['tool_items'] += 1
            elif 'ring' in desc_lower:
                features['ring_items'] += 1
            elif 'wand' in desc_lower:
                features['wand_items'] += 1
        
        return features
    
    def collect_data_batch(self, dataset_name: str, max_samples: int = 1000, 
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
        print(f"Collecting {max_samples} samples from {dataset_name} for VAE training...")
        
        # Create dataset loader
        dataset = nld.TtyrecDataset(
            dataset_name,
            batch_size=batch_size,
            seq_length=seq_length,
            dbfilename=self.dbfilename,
            shuffle=True
        )
        
        collected_samples = []
        sample_count = 0
        processed_count = 0
        
        for batch_idx, minibatch in enumerate(dataset):
            if sample_count >= max_samples:
                break
                
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}, collected {sample_count}/{max_samples} samples...")
            
            # Process each game in the batch
            for game_idx in range(minibatch['tty_chars'].shape[0]):
                # Process each timestep
                for time_idx in range(minibatch['tty_chars'].shape[1]):
                    if sample_count >= max_samples:
                        break
                    
                    # Skip padding (gameid = 0)
                    if minibatch['gameids'][game_idx, time_idx] == 0:
                        continue
                    
                    processed_count += 1
                    
                    # Get TTY data
                    tty_chars = minibatch['tty_chars'][game_idx, time_idx]
                    tty_colors = minibatch['tty_colors'][game_idx, time_idx]
                    tty_cursor = minibatch['tty_cursor'][game_idx, time_idx]
                    
                    # Separate TTY components using our utility function
                    tty_components = separate_tty_components(tty_chars, tty_colors)
                    
                    # Extract text information
                    current_message = get_current_message(tty_chars)
                    status_lines = get_status_lines(tty_chars)
                    game_map = get_game_map(tty_chars, tty_colors)
                    
                    # Optional: detect inventory for potential future analysis
                    inventory_info = self.detect_inventory_screen(tty_chars)
                    
                    # Create sample
                    sample = {
                        # Metadata
                        'gameid': int(minibatch['gameids'][game_idx, time_idx]),
                        'timestep': time_idx,
                        'timestamp': minibatch['timestamps'][game_idx, time_idx],
                        'keypress': int(minibatch['keypresses'][game_idx, time_idx]),
                        'score': int(minibatch['scores'][game_idx, time_idx]),
                        'done': bool(minibatch['done'][game_idx, time_idx]),
                        
                        # TTY components (for VAE training)
                        'tty_chars': tty_chars.astype(np.uint8),
                        'tty_colors': tty_colors.astype(np.int8),
                        'tty_cursor': tty_cursor.astype(np.uint8),
                        
                        # Separated components
                        'message_chars': tty_components['message_chars'].astype(np.uint8),
                        'message_colors': tty_components['message_colors'].astype(np.int8),
                        'game_chars': tty_components['game_chars'].astype(np.uint8),
                        'game_colors': tty_components['game_colors'].astype(np.int8),
                        'status_chars': tty_components['status_chars'].astype(np.uint8),
                        'status_colors': tty_components['status_colors'].astype(np.int8),
                        
                        # Text information
                        'current_message': current_message,
                        'status_lines': status_lines,
                        
                        # Keep inventory info for potential future use (even if rarely found)
                        'is_inventory_screen': inventory_info['is_inventory'],
                        'inventory_items': inventory_info['items'],
                        'inventory_features': inventory_info['inventory_features'],
                    }
                    
                    collected_samples.append(sample)
                    sample_count += 1
            
            # Break from batch loop if we have enough samples
            if sample_count >= max_samples:
                break
        
        print(f"✅ Successfully collected {len(collected_samples)} samples from {processed_count} processed screens")
        return collected_samples


class NetHackDataset(Dataset):
    """PyTorch Dataset for NetHack data"""
    
    def __init__(self, samples: List[Dict], target_type: str = 'reconstruction'):
        """
        Args:
            samples: List of processed samples
            target_type: Type of target ('reconstruction', 'inventory', 'action')
        """
        self.samples = samples
        self.target_type = target_type
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Input: TTY characters and colors
        tty_chars = torch.tensor(sample['tty_chars'], dtype=torch.float32)
        tty_colors = torch.tensor(sample['tty_colors'], dtype=torch.float32)
        
        # Combine chars and colors as input
        input_tensor = torch.stack([tty_chars, tty_colors], dim=0)  # Shape: (2, 24, 80)
        
        if self.target_type == 'reconstruction':
            # For VAE: target is same as input
            target = input_tensor
        elif self.target_type == 'inventory':
            # For inventory prediction: target is inventory features
            inv_features = sample['inventory_features']
            target = torch.tensor([
                inv_features['total_items'],
                inv_features['equipped_items'],
                inv_features['food_items'],
                inv_features['weapon_items'],
                inv_features['armor_items'],
                inv_features['potion_items'],
                inv_features['scroll_items'],
                inv_features['tool_items'],
                inv_features['ring_items'],
                inv_features['wand_items'],
            ], dtype=torch.float32)
        elif self.target_type == 'action':
            # For action prediction: target is next action
            target = torch.tensor(sample['keypress'], dtype=torch.long)
        
        return {
            'input': input_tensor,
            'target': target,
            'metadata': {
                'gameid': sample['gameid'],
                'timestep': sample['timestep'],
                'is_inventory_screen': sample['is_inventory_screen'],
                'current_message': sample['current_message'],
            }
        }


class TTYToMiniHackAdapter:
    """
    Converts TTY data to MiniHackVAE expected format
    
    TTY data format (from NetHack Learning Dataset):
    - chars: (24, 80) array of ASCII characters
    - colors: (24, 80) array of color indices
    - cursor: (y, x) position
    
    MiniHackVAE expected format:
    - glyph_chars: LongTensor[B,H,W] - ASCII character codes (32-127) per map cell
    - glyph_colors: LongTensor[B,H,W] - color indices (0-15) per map cell  
    - blstats: FloatTensor[B,27] - game statistics (hp, gold, position, etc.)
    - msg_tokens: LongTensor[B,256] - tokenized message text (0-127 + SOS/EOS)
    - hero_info: optional dict with role, race, gender, alignment
    - inv_oclasses: optional LongTensor[B,55] - inventory object classes
    - inv_strs: optional LongTensor[B,55,80] - inventory string descriptions
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
    
    def extract_status_line(self, chars: np.ndarray) -> List[str]:
        """Extract NetHack status lines from TTY bottom rows"""
        status_lines = []
        
        # NetHack specifically uses rows 22 and 23 for status information
        status_row_indices = [22, 23]
        
        for row_idx in status_row_indices:
            if row_idx < chars.shape[0]:
                line = ''.join([chr(c) if 32 <= c <= 126 else ' ' for c in chars[row_idx]])
                line = line.strip()
                if line:
                    status_lines.append(line)
        
        return status_lines
    
    def find_player_position(self, chars: np.ndarray) -> Tuple[int, int]:
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
    
    def parse_status_comprehensive(self, status_lines: List[str]) -> Dict:
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
    
    def create_accurate_blstats(self, chars: np.ndarray, cursor: Tuple[int, int], 
                                score: float = 0.0, timestamp: float = 0.0) -> np.ndarray:
        """
        Create accurate 27-dimensional blstats from TTY data
        
        1. Player position found by locating '@' symbol on map (not from status line)
        2. When hunger not shown -> assume NOT_HUNGRY state (hunger_state=1)
        3. When no conditions shown -> assume no conditions (condition_mask=0)
        4. Use reasonable defaults for stats not available in status line
        """
        # Parse status line first
        status_lines = self.extract_status_line(chars)
        parsed_stats = self.parse_status_comprehensive(status_lines)
        
        # Create 27-dimensional blstats array
        blstats = np.zeros(27, dtype=np.int64)
        
        # Position (0, 1) - Find player '@' on map, not from status line
        player_x, player_y = self.find_player_position(chars)
        blstats[0] = player_x  # NLE_BL_X - x coordinate
        blstats[1] = player_y  # NLE_BL_Y - y coordinate
        
        # CORRECTED: Use exact NLE blstats mapping from nle/include/nletypes.h
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
        
    def extract_map_from_tty(self, chars: np.ndarray, colors: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract game map from TTY data"""
        # TTY is (24, 80), map is typically rows 1-21, cols 0-78
        map_chars = chars[self.map_start_row:self.map_end_row, :self.map_width]  # (21, 79)
        map_colors = colors[self.map_start_row:self.map_end_row, :self.map_width]  # (21, 79)
        
        return torch.tensor(map_chars, dtype=torch.long), torch.tensor(map_colors, dtype=torch.long)
    
    def extract_message_from_tty(self, chars: np.ndarray) -> torch.Tensor:
        """Extract message from top row of TTY and convert to tokens"""
        # Message is typically in the top row
        message_chars = chars[0, :]  # (80,)
        
        # Convert to valid message tokens (0-127, pad with 0s)
        msg_tokens = []
        for char in message_chars:
            if 32 <= char <= 127:  # Valid printable ASCII
                msg_tokens.append(char)
            elif char == 0:  # Already padding
                break
            else:
                msg_tokens.append(32)  # Replace invalid with space
        
        # Pad or truncate to 256 tokens
        msg_tensor = torch.zeros(256, dtype=torch.long)
        if msg_tokens:
            msg_len = min(len(msg_tokens), 256)
            msg_tensor[:msg_len] = torch.tensor(msg_tokens[:msg_len], dtype=torch.long)
        
        return msg_tensor
    
    def create_dummy_inventory(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create dummy inventory data - can be None for optional parameters"""
        # Return None to skip inventory for now
        return None, None
    
    def convert_tty_batch(self, tty_batch: Dict[str, torch.Tensor], 
                         game_manager: Optional['GameSequenceManager'] = None,
                         samples_metadata: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """
        Convert a batch of TTY data to MiniHackVAE format with improved blstats.
        Uses GameSequenceManager for hero info extraction and caching.
        
        Args:
            tty_batch: Dict with keys 'chars', 'colors', 'cursor', 'score', 'timestamp'
                - chars: (B, 24, 80)
                - colors: (B, 24, 80)  
                - cursor: (B, 2)
                - score: (B,)
                - timestamp: (B,)
            game_manager: GameSequenceManager for hero info handling
            samples_metadata: Optional list of sample metadata (for game_id, etc.)
        
        Returns:
            Dict with MiniHackVAE expected format including accurate blstats
        """
        batch_size = tty_batch['chars'].shape[0]
        device = tty_batch['chars'].device
        
        # Convert each sample in the batch
        glyph_chars_list = []
        glyph_colors_list = []
        blstats_list = []
        msg_tokens_list = []
        
        for i in range(batch_size):
            chars_i = tty_batch['chars'][i].cpu().numpy()  # (24, 80)
            colors_i = tty_batch['colors'][i].cpu().numpy()  # (24, 80)
            cursor_i = (tty_batch['cursor'][i, 0].item(), tty_batch['cursor'][i, 1].item())
            score_i = tty_batch['score'][i].item() if 'score' in tty_batch else 0.0
            timestamp_i = tty_batch['timestamp'][i].item() if 'timestamp' in tty_batch else 0.0
            
            # Extract map
            map_chars, map_colors = self.extract_map_from_tty(chars_i, colors_i)
            glyph_chars_list.append(map_chars)
            glyph_colors_list.append(map_colors)
            
            # Extract message
            msg_tokens = self.extract_message_from_tty(chars_i)
            msg_tokens_list.append(msg_tokens)
            
            # Extract blstats using improved method
            blstats = self.create_accurate_blstats(chars_i, cursor_i, score_i, timestamp_i)
            blstats_list.append(torch.tensor(blstats, dtype=torch.float32))
        
        # Get hero info using GameSequenceManager
        hero_info_batch = None
        if game_manager is not None and samples_metadata is not None:
            hero_info_batch = game_manager.get_hero_info_batch(samples_metadata, device)
        
        # Stack into batches
        result = {
            'glyph_chars': torch.stack(glyph_chars_list).to(device),      # (B, 21, 79)
            'glyph_colors': torch.stack(glyph_colors_list).to(device),    # (B, 21, 79)
            'blstats': torch.stack(blstats_list).to(device),              # (B, 27)
            'msg_tokens': torch.stack(msg_tokens_list).to(device),        # (B, 256)
            'hero_info': hero_info_batch,                                 # Hero info if available
            'inv_oclasses': None,                                         # None for now
            'inv_strs': None                                              # None for now
        }
        
        return result
    
    def extract_hero_info_for_game_sequence(self, samples: List[Dict], game_id: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Extract hero info for a specific game sequence from the first welcome message found.
        This ensures hero info is consistent throughout the entire game.
        
        Args:
            samples: List of samples from the same game
            game_id: Optional game ID to filter samples
            
        Returns:
            Hero info tensor (4,) with [role, race, gender, alignment] or None if not found
        """
        # Filter by game_id if provided
        if game_id is not None:
            samples = [s for s in samples if s.get('gameid') == game_id]
        
        # Look for welcome message in chronological order (earliest first)
        sorted_samples = sorted(samples, key=lambda x: x.get('timestep', 0))
        
        for sample in sorted_samples:
            if 'message' in sample and sample['message']:
                message = sample['message'].strip()
                if message.startswith('Hello') and 'welcome to NetHack' in message:
                    hero_info_dict = self.parse_hero_info_from_message(message)
                    if hero_info_dict:
                        return self.create_hero_info_from_dict(hero_info_dict)
        return None
    
    def ensure_hero_info_format(self, hero_info: Optional[Union[torch.Tensor, Dict]], batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Ensure hero_info is in the correct [B, 4] tensor format.
        
        Args:
            hero_info: Hero info in various formats (tensor, dict, or None)
            batch_size: Target batch size
            device: Target device
            
        Returns:
            Hero info tensor of shape [B, 4] or None if no hero info available
        """
        if hero_info is None:
            return None
        
        # Convert to single tensor if it's a dict
        if isinstance(hero_info, dict):
            if all(key in hero_info for key in ['role', 'race', 'gender', 'alignment']):
                hero_tensor = torch.tensor([
                    hero_info['role'],
                    hero_info['race'], 
                    hero_info['gender'],
                    hero_info['alignment']
                ], dtype=torch.float32)
            else:
                return None
        elif isinstance(hero_info, torch.Tensor):
            hero_tensor = hero_info.clone()
        else:
            return None
        
        # Ensure it's the right shape and type
        if hero_tensor.dim() == 1 and hero_tensor.size(0) == 4:
            # Single hero info - replicate across batch
            batch_hero_info = hero_tensor.unsqueeze(0).expand(batch_size, -1).to(device)
            return batch_hero_info
        elif hero_tensor.dim() == 2 and hero_tensor.size() == (batch_size, 4):
            # Already correct batch format
            return hero_tensor.to(device)
        else:
            # Invalid format
            return None
    

class GameSequenceManager:
    """
    Centralized manager for hero information persistence throughout NetHack game sequences.
    Handles hero info extraction, caching, and batch processing for mixed-game scenarios.
    """
    
    def __init__(self):
        self.game_hero_info = {}  # game_id -> hero_info tensor [4,]
        
        # NetHack mappings for hero info parsing
        self.role_mapping = {
            'archaeologist': 0, 'barbarian': 1, 'caveman': 2, 'cavewoman': 2,
            'healer': 3, 'knight': 4, 'monk': 5, 'priest': 6, 'priestess': 6,
            'ranger': 7, 'rogue': 8, 'samurai': 9, 'tourist': 10, 'valkyrie': 11,
            'wizard': 12
        }
        
        self.race_mapping = {
            'human': 0, 'elf': 1, 'dwarf': 2, 'gnome': 3, 'orc': 4
        }
        
        self.gender_mapping = {
            'male': 1, 'female': 0  # Note: 0=female, 1=male in NetHack
        }
        
        self.alignment_mapping = {
            'lawful': 1, 'neutral': 0, 'chaotic': -1
        }
    
    def parse_hero_info_from_message(self, message: str) -> Optional[torch.Tensor]:
        """
        Parse hero information from NetHack welcome message and return tensor.
        
        Args:
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
                break
        
        # Only return if we found all required components
        if all(x is not None for x in [alignment, gender, race, role]):
            return torch.tensor([role, race, gender, alignment], dtype=torch.float32)
        
        return None
    
    def get_hero_info_for_game(self, game_id: int, samples: List[Dict]) -> Optional[torch.Tensor]:
        """
        Get hero info for a specific game, extracting from welcome message if not cached.
        
        Args:
            game_id: Game ID to get hero info for
            samples: List of samples from this game (to find welcome message)
            
        Returns:
            Hero info tensor [4,] or None if not found
        """
        # Check cache first
        if game_id in self.game_hero_info:
            return self.game_hero_info[game_id]
        
        # Find welcome message in samples (should be in earliest timestep)
        game_samples = [s for s in samples if s.get('gameid') == game_id]
        sorted_samples = sorted(game_samples, key=lambda x: x.get('timestep', 0))
        
        for sample in sorted_samples:
            if 'message' in sample and sample['message']:
                message = sample['message'].strip()
                if message.startswith('Hello') and 'welcome to NetHack' in message:
                    hero_info = self.parse_hero_info_from_message(message)
                    if hero_info is not None:
                        # Cache for future use
                        self.game_hero_info[game_id] = hero_info
                        return hero_info
        
        return None
    
    def get_hero_info_batch(self, samples_metadata: List[Dict], device: torch.device) -> Optional[torch.Tensor]:
        """
        Get hero info batch for mixed-game samples, ensuring [B, 4] format.
        
        Args:
            samples_metadata: List of sample metadata with 'gameid' and 'message' fields
            device: Target device for tensors
            
        Returns:
            Hero info tensor [B, 4] or None if no hero info available
        """
        batch_size = len(samples_metadata)
        hero_info_list = []
        
        for metadata in samples_metadata:
            game_id = metadata.get('gameid', 0)
            sample_hero_info = None
            
            # Check cache first
            if game_id in self.game_hero_info:
                sample_hero_info = self.game_hero_info[game_id]
            else:
                # Try to extract from current sample message
                message = metadata.get('message', '')
                if message:
                    message = message.strip()
                    if message.startswith('Hello') and 'welcome to NetHack' in message:
                        hero_info = self.parse_hero_info_from_message(message)
                        if hero_info is not None:
                            # Cache for future use
                            self.game_hero_info[game_id] = hero_info
                            sample_hero_info = hero_info
            
            hero_info_list.append(sample_hero_info)
        
        # Check if we have any valid hero info
        valid_hero_info = [h for h in hero_info_list if h is not None]
        if not valid_hero_info:
            return None
        
        # Create batch tensor
        batch_hero_info = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        
        for i, hero_info in enumerate(hero_info_list):
            if hero_info is not None:
                batch_hero_info[i] = hero_info.to(device)
            else:
                # Use zeros for missing hero info (model should handle this)
                batch_hero_info[i] = torch.zeros(4, dtype=torch.float32, device=device)
        
        return batch_hero_info
    
    def preload_hero_info(self, all_samples: List[Dict]):
        """
        Preload hero info for all games in the dataset.
        
        Args:
            all_samples: All samples in the dataset
        """
        # Group samples by game_id
        games = {}
        for sample in all_samples:
            game_id = sample.get('gameid', 0)
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(sample)
        
        # Extract hero info for each game
        for game_id, game_samples in games.items():
            if game_id not in self.game_hero_info:
                self.get_hero_info_for_game(game_id, game_samples)
    
    def clear_cache(self):
        """Clear cached hero info (useful when switching datasets)"""
        self.game_hero_info.clear()


class MiniHackDataset(Dataset):
    """
    Dataset wrapper for NetHack data that converts TTY format to MiniHackVAE expected format.
    Supports hero information persistence throughout game sequences.
    """
    def __init__(self, data_entries: List[Dict], adapter: TTYToMiniHackAdapter, 
                 use_game_manager: bool = True):
        self.data_entries = data_entries
        self.adapter = adapter
        self.use_game_manager = use_game_manager
        
        # Initialize game sequence manager for hero info persistence
        if use_game_manager:
            self.game_manager = GameSequenceManager()
            # Preload hero info for all games in the dataset
            self.game_manager.preload_hero_info(data_entries)
        else:
            self.game_manager = None
        
    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        
        # Convert TTY data to tensors
        chars = torch.tensor(entry['chars'], dtype=torch.long)  # (24, 80)
        colors = torch.tensor(entry['colors'], dtype=torch.long)  # (24, 80)
        cursor = torch.tensor([entry['cursor'][0], entry['cursor'][1]], dtype=torch.long)  # (2,)
        score = torch.tensor(entry.get('score', 0.0), dtype=torch.float32)  # scalar
        timestamp = torch.tensor(entry.get('timestamp', 0.0), dtype=torch.float32)  # scalar
        
        # Create batch format (add batch dimension)
        tty_batch = {
            'chars': chars.unsqueeze(0),    # (1, 24, 80)
            'colors': colors.unsqueeze(0),  # (1, 24, 80)
            'cursor': cursor.unsqueeze(0),  # (1, 2)
            'score': score.unsqueeze(0),    # (1,)
            'timestamp': timestamp.unsqueeze(0)  # (1,)
        }
        
        # Create metadata for GameSequenceManager
        metadata = [{
            'gameid': entry.get('gameid', 0),
            'message': entry.get('message', ''),
            'timestep': entry.get('timestep', 0)
        }]
        
        # Convert to MiniHackVAE format and remove batch dimension
        minihack_data = self.adapter.convert_tty_batch(
            tty_batch, 
            game_manager=self.game_manager,
            samples_metadata=metadata
        )
        
        result = {}
        for key, value in minihack_data.items():
            if value is not None:
                result[key] = value.squeeze(0)  # Remove batch dimension
            else:
                result[key] = None
        
        # Add original metadata for collate function
        result['_metadata'] = metadata[0]
                
        return result


def create_mixed_game_collate_fn(adapter: TTYToMiniHackAdapter, game_manager: Optional[GameSequenceManager] = None):
    """
    Create a custom collate function that handles mixed-game batches properly.
    Each sample in the batch gets its own game-specific hero info.
    
    Args:
        adapter: TTYToMiniHackAdapter instance
        game_manager: Optional GameSequenceManager for hero info caching
        
    Returns:
        Collate function for DataLoader
    """
    def custom_collate_fn(batch):
        """
        Custom collate function for mixed-game batches.
        
        Args:
            batch: List of samples from MiniHackDataset.__getitem__()
            
        Returns:
            Batched data with proper hero info handling
        """
        # Handle None values in batch
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        
        # Extract metadata for hero info processing
        samples_metadata = [item.get('_metadata', {}) for item in batch]
        
        # Stack the main data
        batch_size = len(batch)
        device = batch[0]['glyph_chars'].device if batch else torch.device('cpu')
        
        try:
            glyph_chars = torch.stack([item['glyph_chars'] for item in batch])  # (B, H, W)
            glyph_colors = torch.stack([item['glyph_colors'] for item in batch])  # (B, H, W)
            blstats = torch.stack([item['blstats'] for item in batch])  # (B, 27)
            msg_tokens = torch.stack([item['msg_tokens'] for item in batch])  # (B, 256)
        except Exception as e:
            print(f"Error stacking batch tensors: {e}")
            return None
        
        # Get per-sample hero info using game manager
        hero_info_batch = None
        if game_manager is not None:
            hero_info_batch = game_manager.get_hero_info_batch(samples_metadata, device)
        else:
            # Fallback: collect hero info from individual samples
            hero_info_list = [item.get('hero_info') for item in batch]
            valid_hero_info = [h for h in hero_info_list if h is not None]
            if valid_hero_info:
                hero_info_batch = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
                for i, hero_info in enumerate(hero_info_list):
                    if hero_info is not None:
                        if hero_info.dim() == 1:
                            hero_info_batch[i] = hero_info.to(device)
                        else:
                            hero_info_batch[i] = hero_info.squeeze().to(device)
                    else:
                        hero_info_batch[i] = torch.zeros(4, dtype=torch.float32, device=device)
        
        # Build final batch
        result = {
            'glyph_chars': glyph_chars,
            'glyph_colors': glyph_colors,
            'blstats': blstats,
            'msg_tokens': msg_tokens,
            'hero_info': hero_info_batch,
            'inv_oclasses': None,
            'inv_strs': None
        }
        
        return result
    
    return custom_collate_fn


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


def train_minihack_vae(train_samples: List[Dict], test_samples: List[Dict], 
                      epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-3,
                      latent_dim: int = 64, save_path: str = "models/minihack_vae.pth",
                      device: str = None, include_inventory: bool = False) -> Tuple[MiniHackVAE, List[float], List[float]]:
    """
    Train MiniHackVAE on NetHack Learning Dataset
    
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
    adapter = TTYToMiniHackAdapter()
    train_dataset = MiniHackDataset(train_samples, adapter)
    test_dataset = MiniHackDataset(test_samples, adapter)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = MiniHackVAE(latent_dim=latent_dim, include_inventory=include_inventory)
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


def create_train_test_split(samples: List[Dict], test_ratio: float = 0.2, 
                           random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into train and test sets
    
    Args:
        samples: List of data samples
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_samples, test_samples)
    """
    print(f"Splitting {len(samples)} samples into train/test (test_ratio={test_ratio})")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle samples
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    # Split
    test_size = int(len(shuffled_samples) * test_ratio)
    test_samples = shuffled_samples[:test_size]
    train_samples = shuffled_samples[test_size:]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    return train_samples, test_samples


def save_dataset(samples: List[Dict], filepath: str):
    """Save dataset to disk"""
    print(f"Saving dataset to {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Saved {len(samples)} samples")


def load_dataset(filepath: str) -> List[Dict]:
    """Load dataset from disk"""
    print(f"Loading dataset from {filepath}")
    with open(filepath, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")
    return samples


def test_data_collection(max_samples: int = 100):
    """Test data collection with small samples"""
    print("=" * 60)
    print("TESTING DATA COLLECTION PIPELINE")
    print("=" * 60)
    
    # Initialize collector
    collector = NetHackDataCollector()
    
    try:
        samples = collector.collect_data_batch(
            dataset_name="taster-dataset",
            max_samples=max_samples,
            batch_size=16,
            seq_length=32
        )
        
        print(f"\n✅ Successfully collected {len(samples)} samples")
        
        # Analyze collected data
        print(f"📊 Data Analysis:")
        print(f"  - Total samples: {len(samples)}")
        print(f"  - Unique games: {len(set(s['gameid'] for s in samples))}")
        print(f"  - Score range: {min(s['score'] for s in samples)} - {max(s['score'] for s in samples)}")
        
        # Show sample data
        sample = samples[0]
        print(f"\n📋 Sample Data Structure:")
        print(f"  - TTY chars shape: {sample['tty_chars'].shape}")
        print(f"  - TTY colors shape: {sample['tty_colors'].shape}")
        print(f"  - Game chars shape: {sample['game_chars'].shape}")
        print(f"  - Message: '{sample['current_message']}'")
        print(f"  - Status: {sample['status_lines']}")
        
        # Test train/test split
        print(f"\n🔀 Testing Train/Test Split:")
        train_samples, test_samples = create_train_test_split(samples, test_ratio=0.2)
        
        # Test PyTorch Dataset
        print(f"\n🔥 Testing PyTorch Dataset:")
        dataset = NetHackDataset(train_samples, target_type='reconstruction')
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        batch = next(iter(dataloader))
        print(f"  - Input shape: {batch['input'].shape}")
        print(f"  - Target shape: {batch['target'].shape}")
        print(f"  - Batch size: {len(batch['metadata']['gameid'])}")
        
        # Test action prediction dataset
        print(f"\n⚡ Testing Action Prediction Dataset:")
        action_dataset = NetHackDataset(train_samples, target_type='action')
        action_dataloader = DataLoader(action_dataset, batch_size=8, shuffle=True)
        
        action_batch = next(iter(action_dataloader))
        print(f"  - Input shape: {action_batch['input'].shape}")
        print(f"  - Target shape: {action_batch['target'].shape}")
        print(f"  - Sample actions: {action_batch['target'][:5]}")
        
        # Test saving/loading
        print(f"\n💾 Testing Save/Load:")
        save_dataset(samples[:20], "test_dataset.pkl")
        loaded_samples = load_dataset("test_dataset.pkl")
        print(f"  - Original: {len(samples[:20])} samples")
        print(f"  - Loaded: {len(loaded_samples)} samples")
        
        # Clean up test file
        Path("test_dataset.pkl").unlink()
        
        print(f"\n✅ ALL TESTS PASSED!")
        return samples
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_training_data(max_samples: int = 5000, save_path: str = "nethack_dataset.pkl"):
    """Collect dataset for VAE training"""
    print("=" * 60)
    print("COLLECTING TRAINING DATA")
    print("=" * 60)
    
    collector = NetHackDataCollector()
    
    try:
        # Collect training data
        print(f"Collecting {max_samples} samples for VAE training...")
        samples = collector.collect_data_batch(
            dataset_name="taster-dataset",
            max_samples=max_samples,
            batch_size=64,
            seq_length=128
        )
        
        # Analyze the collected data
        print(f"\n📊 Dataset Analysis:")
        print(f"  - Total samples: {len(samples)}")
        print(f"  - Unique games: {len(set(s['gameid'] for s in samples))}")
        print(f"  - Data size: ~{len(samples) * 24 * 80 * 2 / 1024 / 1024:.1f} MB")
        
        # Show data distribution
        scores = [s['score'] for s in samples]
        keypresses = [s['keypress'] for s in samples]
        print(f"  - Score range: {min(scores)} - {max(scores)}")
        print(f"  - Unique actions: {len(set(keypresses))}")
        print(f"  - Most common actions: {sorted(set(keypresses), key=keypresses.count, reverse=True)[:10]}")
        
        # Create train/test split
        train_samples, test_samples = create_train_test_split(samples, test_ratio=0.2)
        
        # Save datasets
        save_dataset(train_samples, f"train_{save_path}")
        save_dataset(test_samples, f"test_{save_path}")
        save_dataset(samples, save_path)  # Full dataset
        
        print(f"\n✅ Dataset saved!")
        print(f"  - Full dataset: {len(samples)} samples -> {save_path}")
        print(f"  - Training: {len(train_samples)} samples -> train_{save_path}")
        print(f"  - Testing: {len(test_samples)} samples -> test_{save_path}")
        
        return train_samples, test_samples, samples
        
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Test with small samples first
    print("1️⃣ Testing basic data collection...")
    test_samples = test_data_collection(max_samples=100)
    
    if test_samples:
        print("\n2️⃣ Ready for full data collection!")
        
        # Ask user if they want to proceed with full collection
        print("\n🚀 To complete the full VAE training pipeline:")
        print("\n📋 Step 1: Collect training data")
        print("  train_samples, test_samples, full_samples = collect_training_data(max_samples=2000)")
        
        print("\n🧠 Step 2: Train VAE")
        print("  model, train_losses, test_losses = train_vae(train_samples, test_samples, num_epochs=25)")
        
        print("\n� Step 3: Analyze results")
        print("  visualize_reconstructions(model, test_samples, device)")
        print("  latent_vectors, game_ids, messages = analyze_latent_space(model, test_samples, device)")
        
        print("\n💡 Complete example:")
        print("```python")
        print("# Collect data")
        print("train_data, test_data, full_data = collect_training_data(max_samples=2000)")
        print("")
        print("# Train VAE")
        print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        print("model, train_losses, test_losses = train_vae(")
        print("    train_data, test_data, ")
        print("    latent_dim=128, batch_size=32, num_epochs=25")
        print(")")
        print("")
        print("# Analyze results")
        print("visualize_reconstructions(model, test_data, device)")
        print("latent_vectors, _, _ = analyze_latent_space(model, test_data, device)")
        print("")
        print("# Use latent representations for downstream tasks")
        print("print(f'Latent representation shape: {latent_vectors.shape}')")
        print("```")
        
        print("\n🎯 Your VAE will learn to:")
        print("  - Encode NetHack screens into a compact latent space")
        print("  - Reconstruct screens from latent representations")
        print("  - Capture meaningful game state information")
        print("  - Enable analysis of game progression patterns")
        
        print("\n🆕 NEW: MiniHackVAE Integration!")
        print("  You can now use the sophisticated MiniHackVAE from src/model.py")
        print("  Example:")
        print("  ```python")
        print("  # Use the advanced MiniHackVAE instead of simple NetHackVAE")
        print("  train_data, test_data, _ = collect_training_data(max_samples=1000)")
        print("  model, train_losses, test_losses = train_minihack_vae(")
        print("      train_data, test_data, ")
        print("      epochs=15, batch_size=16, latent_dim=64")
        print("  )")
        print("  ```")
        print("  ✨ Features: Multi-modal encoding, sophisticated decoders, flexible covariance")
        
    else:
        print("❌ Basic testing failed")

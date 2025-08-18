from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import torch
import nle.dataset as nld
from nle.nethack import tty_render
from utils import detect_valid_map, get_game_map, get_current_message, get_status_lines
import time
import numpy as np
import os
import pickle
import re

# ---- Direction / actions -----------------------------------------------------
DIRS_8 = [(-1,0),(+1,0),(0,+1),(0,-1),(-1,+1),(-1,-1),(+1,+1),(+1,-1)]  # N,S,E,W,NE,NW,SE,SW
ACTION_NAMES = ["N","S","E","W","NE","NW","SE","SW","WAIT"]
ACTION_DIM = 9  # 8 moves + wait

VI_KEY_TO_DIR = {
    ord('k'):(-1,0), ord('j'):(+1,0), ord('l'):(0,+1), ord('h'):(0,-1),
    ord('u'):(-1,+1), ord('y'):(-1,-1), ord('n'):(+1,+1), ord('b'):(+1,-1),
}
WAIT_KEYS = {ord('.'), ord(' '), 13}  # '.', space, Enter

def map_keypress_to_action_index(code: int) -> int:
    """Map tty key code to action index in 0..8 (8 = WAIT). Unknown → WAIT."""
    if code in VI_KEY_TO_DIR:
        dy, dx = VI_KEY_TO_DIR[code]
    else:
        # try arrows (optional): you can extend this table if your dataset encodes arrows specially
        dy, dx = (0,0)
        if code in WAIT_KEYS:
            dy, dx = (0,0)
    # map to index
    if   (dy,dx)==(-1,0): return 0
    elif (dy,dx)==(+1,0): return 1
    elif (dy,dx)==(0,+1): return 2
    elif (dy,dx)==(0,-1): return 3
    elif (dy,dx)==(-1,+1): return 4
    elif (dy,dx)==(-1,-1):return 5
    elif (dy,dx)==(+1,+1):return 6
    elif (dy,dx)==(+1,-1):return 7
    else:                  return 8  # WAIT

def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(*indices.shape, num_classes, dtype=torch.float32)
    out.scatter_(-1, indices.long().unsqueeze(-1), 1.0)
    return out

# ---- Passability / safety rules (ASCII) -------------------------------------
WALLS       = {ord('|'), ord('-')}
GROUND      = {ord('.'), ord('#')}
CLOSED_DOOR = {ord('+')}
OPEN_DOOR   = {ord('/')}          # if absent in your tty, treat as ground
STAIRS      = {ord('<'), ord('>')}
TRAPS       = {ord('^')}
BOULDER     = {ord('0')}
WATER_LAVA  = {ord('~')}
SPECIAL     = {ord('_'), ord('}'), ord('{'), ord('\\')}
ITEMS       = set(map(ord, list(')$!?"[=*%/,:;')))

def is_monster(ch: int) -> bool:
    return (65 <= ch <= 90) or (97 <= ch <= 122)  # A..Z or a..z

PASSABLE = GROUND | OPEN_DOOR | STAIRS | SPECIAL | ITEMS  # monsters treated separately

def compute_passability_and_safety(chars: torch.Tensor,
                                   colors: torch.Tensor,
                                   hero_y: int, hero_x: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pass8, safe8) as float tensors of shape [8]."""
    H, W = chars.shape
    pass8 = torch.zeros(8, dtype=torch.float32)
    safe8 = torch.zeros(8, dtype=torch.float32)
    for i,(dy,dx) in enumerate(DIRS_8):
        y, x = hero_y + dy, hero_x + dx
        if not (0 <= y < H and 0 <= x < W):
            pass8[i] = 0.0; safe8[i] = 0.0; continue
        ch = int(chars[y,x])
        # legal to step?
        legal = (ch in PASSABLE) or is_monster(ch)
        # safety: legal AND not a hazard/unknown/monster
        unsafe = (ch in TRAPS) or (ch in WATER_LAVA) or (ch == ord(' ')) or is_monster(ch)
        pass8[i] = 1.0 if legal else 0.0
        safe8[i] = 1.0 if (legal and not unsafe) else 0.0
    return pass8, safe8

# ---- Goal vector (nearest stairs) -------------------------------------------
def nearest_stairs_vector(chars: torch.Tensor,
                          hero_y: int, hero_x: int,
                          prefer: str = '>') -> tuple[float, float] | None:
    """Return (dx,dy) in [-1,1] from hero to nearest visible stairs (prefer '>' or '<')."""
    H, W = chars.shape
    goal_ch = ord(prefer)
    ys, xs = (chars == goal_ch).nonzero(as_tuple=True)
    if ys.numel() == 0:
        # fall back to the other stairs if present
        alt = ord('<') if prefer == '>' else ord('>')
        ys, xs = (chars == alt).nonzero(as_tuple=True)
        if ys.numel() == 0:
            return None
    # choose Euclidean nearest
    dy = ys - hero_y
    dx = xs - hero_x
    i = (dy*dy + dx*dx).argmin().item()
    gx, gy = int(xs[i].item()), int(ys[i].item())
    # normalize to [-1,1]
    dxn = max(-1.0, min(1.0, (gx - hero_x) / (W/2)))
    dyn = max(-1.0, min(1.0, (gy - hero_y) / (H/2)))
    return (dxn, dyn)

# ---- Ego crop and class mapping ---------------------------------------------
def crop_ego(chars: torch.Tensor, hero_y: int, hero_x: int, k: int) -> torch.Tensor:
    """Return ego crop [k,k] of ASCII codes, padded with spaces 32."""
    H, W = chars.shape
    r = k // 2
    out = torch.full((k,k), fill_value=32, dtype=chars.dtype)  # spaces
    y0, y1 = max(0, hero_y - r), min(H, hero_y + r + 1)
    x0, x1 = max(0, hero_x - r), min(W, hero_x + r + 1)
    oy0, oy1 = r - (hero_y - y0), r + (y1 - hero_y)
    ox0, ox1 = r - (hero_x - x0), r + (x1 - hero_x)
    out[oy0:oy1, ox0:ox1] = chars[y0:y1, x0:x1]
    return out

def map_ego_classes(ego_chars: torch.Tensor, mapper: callable | None) -> torch.Tensor:
    """
    Map ego ASCII grid [k,k] -> class ids [k,k].
    If mapper is None, use a compact affordance palette as fallback.
    """
    k = ego_chars.shape[0]
    out = torch.zeros((k,k), dtype=torch.long)
    if mapper is not None:
        # expect mapper(char_int:int) -> class_id:int
        for y in range(k):
            for x in range(k):
                out[y,x] = int(mapper(int(ego_chars[y,x])))
        return out

    # Fallback coarse palette ~12 classes
    for y in range(k):
        for x in range(k):
            ch = int(ego_chars[y,x])
            if ch == 32:             cid = 0  # unknown
            elif ch in WALLS:        cid = 2
            elif ch in GROUND:       cid = 1
            elif ch in CLOSED_DOOR:  cid = 3
            elif ch == ord('<'):     cid = 4
            elif ch == ord('>'):     cid = 5
            elif ch in TRAPS:        cid = 8
            elif ch in BOULDER:      cid = 9
            elif ch in WATER_LAVA:   cid = 10
            elif ch in SPECIAL:      cid = 11
            elif is_monster(ch):     cid = 7
            elif ch in ITEMS:        cid = 6
            else:                    cid = 1
            out[y,x] = cid
    return out

# ---- Discounted k-step returns ----------------------------------------------
def discounted_k_step(rewards: np.ndarray, done: np.ndarray, k: int, gamma: float) -> np.ndarray:
    """
    rewards: [T] float, done: [T] bool for the *result* at timestep t (i.e., episode ends at t if done[t])
    Returns G^{(k)}_t = sum_{i=0}^{k-1} gamma^i r_{t+i}, stopping at done.
    """
    T = rewards.shape[0]
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        g, w = 0.0, 1.0
        for i in range(k):
            ti = t + i
            if ti >= T or done[ti]:
                break
            g += w * rewards[ti]
            w *= gamma
        out[t] = g
    return out


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
                       batch_size: int = 32, seq_length: int = 32, game_ids: List[int] | None = None,
                       *,
                       ego_window: int = 11,
                       value_horizons: List[int] = [5, 10],
                       gamma: float = 0.99,
                       goal_prefer: str = '>',
                       ego_class_mapper: callable | None = None,
                       allow_diagonal: bool = True) -> List[Dict]:
        """
        Collect data from the dataset for VAE training and build decision-sufficient targets.

        - Adds: action_index, action_onehot, reward_target, done_target, value_k_target,
                passability_target, safety_target, goal_target (+goal_mask), ego_sem_target, has_next.
        - Shapes are [num_games, num_time, ...], consistent with your current pipeline.
        - z_next_detached is produced later during training by encoding obs_{t+1}; here we prepare has_next masks.

        ego_class_mapper: optional callable(int ascii_code) -> class_id.
                        If None, a compact fallback mapping is used.
        """
        print(f"Collecting {max_batches} batches from {dataset_name} for VAE training...")
        print(f"  - Batch size: {batch_size}, Sequence length: {seq_length}")
        
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
            
            this_batch: Dict[str, torch.Tensor] = {}
            num_games, num_time = minibatch['tty_chars'].shape[:2]
            
            # Copy raw arrays -> torch
            for key, item in minibatch.items():
                if isinstance(item, np.ndarray):
                    this_batch[key] = torch.tensor(item)
                else:
                    this_batch[key] = item

            # Preallocate tensors
            message_chars_minibatch = torch.zeros((num_games, num_time, 256), dtype=torch.long)
            game_chars_minibatch    = torch.ones((num_games, num_time, 21, 79), dtype=torch.long) * 32
            game_colors_minibatch   = torch.zeros((num_games, num_time, 21, 79), dtype=torch.long)
            status_chars_minibatch  = torch.zeros((num_games, num_time, 2, 80), dtype=torch.long)
            hero_info_minibatch     = torch.ones((num_games, num_time, 4), dtype=torch.int32)
            blstats_minibatch       = torch.zeros((num_games, num_time, 27), dtype=torch.float32)
            valid_screen_minibatch  = torch.ones((num_games, num_time), dtype=torch.bool)

            # New targets
            action_index_mb    = torch.full((num_games, num_time), fill_value=8, dtype=torch.long)          # default WAIT
            action_onehot_mb   = torch.zeros((num_games, num_time, ACTION_DIM), dtype=torch.float32)
            reward_target_mb   = torch.zeros((num_games, num_time), dtype=torch.float32)
            done_target_mb     = torch.zeros((num_games, num_time), dtype=torch.float32)
            passability_mb     = torch.zeros((num_games, num_time, 8), dtype=torch.float32)
            safety_mb          = torch.zeros((num_games, num_time, 8), dtype=torch.float32)
            goal_target_mb     = torch.zeros((num_games, num_time, 2), dtype=torch.float32)
            goal_mask_mb       = torch.zeros((num_games, num_time), dtype=torch.float32)
            has_next_mb        = torch.zeros((num_games, num_time), dtype=torch.bool)
            ego_sem_mb         = torch.zeros((num_games, num_time, ego_window, ego_window), dtype=torch.long)

            # Convenience aliases to raw
            tty_chars = this_batch['tty_chars']    # [G,T, H,W]
            tty_colors= this_batch['tty_colors']   # [G,T, H,W]
            done_raw  = this_batch['done'].astype(bool) if isinstance(this_batch['done'], np.ndarray) else np.array(this_batch['done'], dtype=bool)
            done_raw  = torch.tensor(done_raw)     # [G,T] bool
            scores    = torch.tensor(this_batch['scores'], dtype=torch.float32) if 'scores' in this_batch else torch.zeros(num_games, num_time)
            keycodes  = torch.tensor(this_batch['keypresses'], dtype=torch.int64)  # [G,T]
            
            # Process each game in the batch
            for g in range(num_games):
                last_hero = None
                # Per-game rewards: r[t] = score[t+1]-score[t]
                r = torch.zeros(num_time, dtype=torch.float32)
                if 'scores' in this_batch:
                    r[:-1] = scores[g,1:] - scores[g,:-1]
                # Fill scalar targets and actions first (vectorized per game)
                for t in range(num_time):
                    kcode = int(keycodes[g,t].item())
                    ai = map_keypress_to_action_index(kcode)
                    action_index_mb[g,t] = ai
                    # done at this timestep (result of previous action)
                    done_target_mb[g,t]  = float(done_raw[g,t].item())
                    # reward at this timestep
                    reward_target_mb[g,t] = float(r[t].item())
                    has_next_mb[g,t] = (t+1 < num_time) and (not bool(done_raw[g,t].item()))
                # One-hot actions
                action_onehot_mb[g] = one_hot(action_index_mb[g], ACTION_DIM)

                # Build k-step returns per game (stop at done)
                done_np = done_raw[g].cpu().numpy()
                r_np    = r.cpu().numpy()
                vals = []
                for k in value_horizons:
                    vals.append(discounted_k_step(r_np, done_np, k=k, gamma=gamma))
                # [M,T] -> [T,M]
                vals = np.stack(vals, axis=0).transpose(1,0)
                vals = torch.tensor(vals, dtype=torch.float32)
                this_val = vals  # [T,M]

                # Per-timestep spatial targets (passability/safety/ego/goal)
                for t in range(num_time):
                    chars = tty_chars[g,t]     # [H,W] int
                    colors= tty_colors[g,t]
                    is_valid_map = detect_valid_map(chars.numpy() if isinstance(chars, torch.Tensor) else chars)
                    if not is_valid_map:
                        valid_screen_minibatch[g,t] = False
                        continue

                    game_map = get_game_map(chars, colors)
                    chars_map = game_map['chars']    # torch.[H,W] long
                    colors_map= game_map['colors']   # torch.[H,W] long

                    # hero pos: prefer '@', else tty_cursor if within bounds, else last seen
                    ys, xs = (chars_map == ord('@')).nonzero(as_tuple=True)
                    if ys.numel() > 0:
                        hy, hx = int(ys[0].item()), int(xs[0].item())
                        last_hero = (hy, hx)
                    elif 'tty_cursor' in this_batch:
                        cy, cx = int(this_batch['tty_cursor'][g,t,0]), int(this_batch['tty_cursor'][g,t,1])
                        if 0 <= cy < chars_map.shape[0] and 0 <= cx < chars_map.shape[1]:
                            hy, hx = cy, cx
                            last_hero = (hy, hx)
                        elif last_hero is not None:
                            hy, hx = last_hero
                        else:
                            valid_screen_minibatch[g,t] = False
                            continue
                    elif last_hero is not None:
                        hy, hx = last_hero
                    else:
                        valid_screen_minibatch[g,t] = False
                        continue

                    # passability & safety
                    p8, s8 = compute_passability_and_safety(chars_map, colors_map, hy, hx)
                    passability_mb[g,t] = p8
                    safety_mb[g,t]      = s8

                    # ego semantic classes
                    ego_chars = crop_ego(chars_map, hy, hx, ego_window)
                    ego_sem   = map_ego_classes(ego_chars, ego_class_mapper)
                    ego_sem_mb[g,t] = ego_sem

                    # goal vector
                    goal_vec = nearest_stairs_vector(chars_map, hy, hx, prefer=goal_prefer)
                    if goal_vec is not None:
                        goal_target_mb[g,t,0] = goal_vec[0]
                        goal_target_mb[g,t,1] = goal_vec[1]
                        goal_mask_mb[g,t]     = 1.0
                    else:
                        goal_target_mb[g,t].zero_()
                        goal_mask_mb[g,t] = 0.0

                # attach value targets (same T for the game)
                # allocate on first assignment, then fill
                if 'value_k_target' not in this_batch:
                    this_batch['value_k_target'] = torch.zeros((num_games, num_time, len(value_horizons)), dtype=torch.float32)
                this_batch['value_k_target'][g] = this_val

            # Pack up everything
            this_batch['message_chars']   = message_chars_minibatch
            this_batch['game_chars']      = game_chars_minibatch
            this_batch['game_colors']     = game_colors_minibatch
            this_batch['status_chars']    = status_chars_minibatch
            this_batch['hero_info']       = hero_info_minibatch
            this_batch['blstats']         = blstats_minibatch
            this_batch['valid_screen']    = valid_screen_minibatch

            # New keys
            this_batch['action_index']      = action_index_mb
            this_batch['action_onehot']     = action_onehot_mb
            this_batch['reward_target']     = reward_target_mb
            this_batch['done_target']       = done_target_mb
            # value_k_target already set per-game above
            this_batch['passability_target']= passability_mb
            this_batch['safety_target']     = safety_mb
            this_batch['goal_target']       = goal_target_mb
            this_batch['goal_mask']         = goal_mask_mb
            this_batch['ego_sem_target']    = ego_sem_mb
            this_batch['has_next']          = has_next_mb

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

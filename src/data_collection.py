from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from enum import IntEnum
import torch
import nle.dataset as nld
from nle.nethack import tty_render
from utils.env_utils import detect_valid_map, get_game_map, get_current_message, get_status_lines
import time
import numpy as np
import os
import pickle
import re
from utils.action_utils import ACTION_DIM, batch_keypress_static_map

# Constants (avoid circular import)
PADDING_CHAR = 32      # padding character (blank space)
PADDING_COLOR = 0       # padding color (black)  
HERO_CHAR = ord('@')  # hero character code (ASCII 64)

DIRS_8 = [(-1,0),(0,+1),(+1,0),(0,-1),(-1,+1),(+1,+1),(+1,-1),(-1,-1)]  # N,E,S,W,NE,SE,SW,NW
ACTION_NAMES = ["N","E","S","W","NE","SE","SW","NW"]

def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(*indices.shape, num_classes, dtype=torch.float32)
    out.scatter_(-1, indices.long().unsqueeze(-1), 1.0)
    return out

# Color constants
CLR_BLACK, CLR_RED, CLR_GREEN, CLR_BROWN = 0, 1, 2, 3
CLR_BLUE, CLR_MAGENTA, CLR_CYAN, CLR_GRAY = 4, 5, 6, 7
NO_COLOR, CLR_ORANGE, CLR_BRIGHT_GREEN = 8, 9, 10
CLR_YELLOW, CLR_BRIGHT_BLUE, CLR_BRIGHT_MAGENTA = 11, 12, 13
CLR_BRIGHT_CYAN, CLR_WHITE = 14, 15


class NetHackCategory(IntEnum):
    """
    Enum for categorizing NetHack game elements into meaningful groups.
    
    Usage:
        category = categorize_glyph(char_code, color_code)
        if category == NetHackCategory.WALLS:
            # handle wall logic
        
    Values are designed to be used as indices for:
    - Neural network outputs
    - Data analysis and visualization
    - Game logic categorization
    """
    UNKNOWN = 0      # Default/unrecognized elements
    WALLS = 1        # Walls and barriers
    DOORS = 2        # Doors
    FLOORS = 3       # Walkable floor tiles
    UP_STAIRS = 4    # Stairs and ladders
    DOWN_STAIRS = 5  # Stairs and ladders
    WATER_LAVA = 6   # Water and lava
    TRAPS = 7        # All types of traps
    ITEMS = 8        # Pickupable items (weapons, armor, etc.)
    FURNITURE = 9    # Non-movable environment objects
    PLAYER_OR_HUMAN = 10  # Player character or human-like entities
    HARMLESS_MONSTERS = 11    # Living creatures that do not pose a threat
    LOW_THREAT_MONSTERS = 12  # Low-threat monsters (e.g. rats, kobolds)
    MEDIUM_THREAT_MONSTERS = 13  # Medium-threat monsters (e.g. goblins, orcs)
    HIGH_THREAT_MONSTERS = 14  # High-threat monsters (e.g. dragons, demons)
    EXTREME_THREAT_MONSTERS = 15  # Extreme-threat monsters (e.g. demons, liches)
    UNKNOWN_MONSTERS = 16  # Monsters with unknown threat level

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of all category names."""
        return [category.name for category in cls]
    
    @classmethod
    def get_category_count(cls) -> int:
        """Get total number of categories."""
        return len(cls)

    def get_category_name(self) -> str:
        """Get the name of the category."""
        return self.name


# Complete categorization sets (existing structure preserved)
WALLS = {
    (ord('|'), CLR_GRAY), 
    (ord('-'), CLR_GRAY),
    (ord('#'), CLR_CYAN),    # iron bars
}

DOORS = {
    (ord('+'), CLR_BROWN),   # closed door
    (ord('#'), CLR_GREEN),   # tree
}

FLOORS = {
    (ord('.'), CLR_GRAY),    # room floor
    (ord('.'), CLR_BLACK),   # dark room
    (ord('#'), CLR_GRAY),    # corridor
    (ord('.'), CLR_CYAN),    # ice
    (ord('|'), CLR_BROWN),   # open door horizontal
    (ord('-'), CLR_BROWN),   # open door vertical
}

UP_STAIRS = {
    (ord('<'), CLR_GRAY),    # stairs up
    (ord('<'), CLR_BROWN)    # ladder up
}

DOWN_STAIRS = {
    (ord('>'), CLR_GRAY),    # stairs down
    (ord('>'), CLR_BROWN),   # ladder down
}

WATER_LAVA = {
    (ord('}'), CLR_BLUE),    # water
    (ord('}'), CLR_RED),     # lava
}

TRAPS = {
    (ord('^'), CLR_CYAN),              # metal traps (arrow, dart, bear)
    (ord('^'), CLR_GRAY),              # stone traps (falling rock, boulder, statue)
    (ord('^'), CLR_BROWN),             # wooden traps (squeaky board, hole, trap door)
    (ord('^'), CLR_RED),               # fire traps (land mine, fire)
    (ord('^'), CLR_BLUE),              # rust traps
    (ord('^'), CLR_BLACK),             # pit traps
    (ord('^'), CLR_BRIGHT_BLUE),       # magic traps (sleeping gas, magic, anti-magic)
    (ord('^'), CLR_MAGENTA),           # teleport traps
    (ord('^'), CLR_BRIGHT_MAGENTA),    # magic portal
    (ord('^'), CLR_BRIGHT_GREEN),      # polymorph traps
    (ord('"'), CLR_GRAY),              # web
    (ord('~'), CLR_MAGENTA),           # vibrating square
}

ITEMS = {
    (ord('$'), CLR_YELLOW),            # gold
    (ord('"'), CLR_BRIGHT_MAGENTA),    # amulet
    (ord(')'), CLR_CYAN),              # weapon
    (ord('['), CLR_BROWN),             # armor
    (ord('='), CLR_CYAN),              # ring
    (ord('('), CLR_BROWN),             # tool
    (ord('%'), CLR_RED),               # food
    (ord('!'), CLR_BLUE),              # potion
    (ord('?'), CLR_WHITE),             # scroll
    (ord('+'), CLR_BROWN),             # spellbook
    (ord('/'), CLR_BROWN),             # wand
    (ord('*'), CLR_GRAY),              # gem
    (ord('`'), CLR_GRAY),              # rock
}

FURNITURE = {
    (ord('_'), CLR_GRAY),              # altar
    (ord('|'), CLR_WHITE),             # grave
    (ord('\\'), CLR_YELLOW),           # throne
    (ord('#'), CLR_GRAY),              # sink (when context is furniture)
    (ord('{'), CLR_BRIGHT_BLUE),       # fountain
}

def categorize_monster_by_threat_level(char, color):
    """Categorize monsters by threat level using both character and color."""

    # CLR_BRIGHT_MAGENTA color always indicates extreme threat regardless of character
    if color == CLR_BRIGHT_MAGENTA:
        return NetHackCategory.EXTREME_THREAT_MONSTERS
    
    # Character-based base threat assessment with color modifiers
    
    # Harmless/weak - but color can upgrade threat
    if char in ['r', 'n', 'y', ':', 'l']:  # rodents, nymphs, lights, lizards, leprechauns
        if color in [CLR_RED, CLR_BLACK]:  # Aggressive/dark variants
            return NetHackCategory.LOW_THREAT_MONSTERS  # Upgrade from harmless
        return NetHackCategory.HARMLESS_MONSTERS

    # Low threat - color can modify
    if char in ['a', 'b', 'j', 'k', 'f', 'B', 'F', 'e', ';']:  # ants, blobs, jellies, kobolds, cats, bats, fungi, eyes, eels
        if color == CLR_RED:  # Fire variants (fire ants, hell hounds)
            return NetHackCategory.MEDIUM_THREAT_MONSTERS  # Upgrade threat
        elif color == CLR_BLACK:  # Dark/evil variants
            return NetHackCategory.MEDIUM_THREAT_MONSTERS  # Upgrade threat
        return NetHackCategory.LOW_THREAT_MONSTERS

    # Medium threat - color can modify significantly
    if char in ['d', 'o', 'g', 'h', 's', 'G', 'O', 'c', 'p', 'q', 'u', 'C', 'S', 'K', 'Y']:
        if color == CLR_RED:  # Fire/hell variants (hell hounds, fire giants)
            return NetHackCategory.HIGH_THREAT_MONSTERS  # Upgrade threat
        elif color == CLR_BLACK:  # Dark/powerful variants
            return NetHackCategory.HIGH_THREAT_MONSTERS  # Upgrade threat
        return NetHackCategory.MEDIUM_THREAT_MONSTERS

    # High threat - color can push to extreme
    if char in ['D', 'L', 'V', 'H', 'T', 'v', 'w', 'E', 'N', 'P', 'R', 'U', 'X', 'J']:
        # Special case: Dragons have color-specific threat levels
        if char == 'D':
            if color == CLR_RED:    # Red dragon - extremely dangerous
                return NetHackCategory.EXTREME_THREAT_MONSTERS
            elif color == CLR_BLACK: # Black dragon - very dangerous
                return NetHackCategory.EXTREME_THREAT_MONSTERS
            elif color in [1, 4, 2, 11, 0, 15, 6]:  # All adult dragons are high threat
                return NetHackCategory.HIGH_THREAT_MONSTERS
        
        # Liches with special colors
        if char == 'L' and color in [CLR_RED, CLR_BLACK]:
            return NetHackCategory.EXTREME_THREAT_MONSTERS  # Master lich, arch-lich

        # Giants with fire/dark variants
        if char == 'H' and color in [CLR_RED, CLR_BLACK]:
            return NetHackCategory.EXTREME_THREAT_MONSTERS  # Fire giants, storm giants

        return NetHackCategory.HIGH_THREAT_MONSTERS
    
    # Extreme threat - color confirms or maintains
    if char in ['&', 'A', 'W', 'Q']:
        # Demons with CLR_BRIGHT_MAGENTA color are the worst
        if char == '&' and color == CLR_BRIGHT_MAGENTA:
            return NetHackCategory.EXTREME_THREAT_MONSTERS  # Demon lords, princes
        return NetHackCategory.EXTREME_THREAT_MONSTERS

    # Special/variable threat
    if char == '@':
        return NetHackCategory.PLAYER_OR_HUMAN  # Could be player OR NPC human

    # Unique/special cases with color consideration
    if char == ' ':  # ghosts
        if color == CLR_BLACK:
            return NetHackCategory.HIGH_THREAT_MONSTERS  # Dark/powerful ghosts
        return NetHackCategory.MEDIUM_THREAT_MONSTERS  # Regular ghosts
    
    if char == "'":  # golems
        if color in [CLR_RED, CLR_BLACK, NO_COLOR]:  # Iron golems, special golems
            return NetHackCategory.EXTREME_THREAT_MONSTERS
        return NetHackCategory.HIGH_THREAT_MONSTERS  # Regular golems
    
    if char in ['m', 't', ']']:  # mimics, trappers, mimic_def
        if color in [CLR_RED, CLR_BLACK]:  # Aggressive/dangerous variants
            return NetHackCategory.HIGH_THREAT_MONSTERS
        return NetHackCategory.MEDIUM_THREAT_MONSTERS  # Surprise/ambush monsters
    
    if char in ['i', 'x', 'z', 'M', 'Z']:  # imps, xan, zruty, mummies, zombies
        if color in [CLR_RED, CLR_BLACK]:  # Powerful undead/demons
            return NetHackCategory.HIGH_THREAT_MONSTERS
        return NetHackCategory.MEDIUM_THREAT_MONSTERS
    
    if char == '~':  # worm tails
        return NetHackCategory.LOW_THREAT_MONSTERS  # Just tail segments

    return NetHackCategory.UNKNOWN

def is_monster(char_code: int) -> bool:
    return chr(char_code).isalpha() or chr(char_code) in ['@', ' ', "'", '&', ';', ':', '~', ']']

def categorize_glyph(char_code: int, color_code: int) -> NetHackCategory:
    """
    Categorize a (character, color) pair into a NetHackCategory.
    
    Args:
        char_code: ASCII character code (32-127)
        color_code: Color index (0-15)
        
    Returns:
        NetHackCategory enum value
        
    Examples:
        categorize_glyph(ord('|'), CLR_GRAY) -> NetHackCategory.WALLS
        categorize_glyph(ord('.'), CLR_GRAY) -> NetHackCategory.FLOORS
    """
    pair = (char_code, color_code)
    
    # Space/padding
    if char_code == PADDING_CHAR and color_code == PADDING_COLOR:
        return NetHackCategory.UNKNOWN
    
    # Check predefined categories
    if pair in WALLS:
        return NetHackCategory.WALLS
    elif pair in DOORS:
        return NetHackCategory.DOORS
    elif pair in FLOORS:
        return NetHackCategory.FLOORS
    elif pair in UP_STAIRS:
        return NetHackCategory.UP_STAIRS
    elif pair in DOWN_STAIRS:
        return NetHackCategory.DOWN_STAIRS
    elif pair in WATER_LAVA:
        return NetHackCategory.WATER_LAVA
    elif pair in TRAPS:
        return NetHackCategory.TRAPS
    elif pair in ITEMS:
        return NetHackCategory.ITEMS
    elif pair in FURNITURE:
        return NetHackCategory.FURNITURE
    elif is_monster(char_code):
        return categorize_monster_by_threat_level(chr(char_code), color_code)
    else:
        return NetHackCategory.UNKNOWN


def categorize_glyph_tensor(char_tensor: torch.Tensor, color_tensor: torch.Tensor) -> torch.Tensor:
    """
    Vectorized categorization for tensors of glyph data.
    
    Args:
        char_tensor: Tensor of character codes, any shape
        color_tensor: Tensor of color codes, same shape as char_tensor
        
    Returns:
        Tensor of category indices (NetHackCategory values), same shape as input
        
    Example:
        chars = torch.tensor([[ord('.'), ord('|')], [ord('@'), ord(' ')]])
        colors = torch.tensor([[CLR_GRAY, CLR_GRAY], [CLR_WHITE, CLR_BLACK]])
        categories = categorize_glyph_tensor(chars, colors)
        # Returns: [[FLOORS, WALLS], [HERO, SPACE]]
    """
    original_shape = char_tensor.shape
    char_flat = char_tensor.flatten()
    color_flat = color_tensor.flatten()
    
    # Initialize result tensor
    result = torch.full_like(char_flat, NetHackCategory.UNKNOWN.value, dtype=torch.long)
    
    # Vectorized categorization using masks
    hero_mask = char_flat == HERO_CHAR
    result[hero_mask] = NetHackCategory.PLAYER_OR_HUMAN.value

    space_mask = (char_flat == PADDING_CHAR) & (color_flat == PADDING_COLOR)
    
    # For other categories, we need to check pairs
    # This is less efficient but more readable - could be optimized if needed
    for i in range(len(char_flat)):
        if not (hero_mask[i] or space_mask[i]):
            category = categorize_glyph(char_flat[i].item(), color_flat[i].item())
            result[i] = category.value
    
    return result.reshape(original_shape)


def get_category_distribution(char_tensor: torch.Tensor, color_tensor: torch.Tensor) -> torch.Tensor:
    """
    Get the distribution of categories in the given tensors.
    
    Args:
        char_tensor: Tensor of character codes
        color_tensor: Tensor of color codes
        
    Returns:
        Tensor of shape [num_categories] with counts for each category
    """
    categories = categorize_glyph_tensor(char_tensor, color_tensor)
    num_categories = NetHackCategory.get_category_count()
    
    # Count occurrences of each category
    counts = torch.zeros(num_categories, dtype=torch.long)
    for i in range(num_categories):
        counts[i] = (categories == i).sum()
    
    return counts


def print_category_distribution(char_tensor: torch.Tensor, color_tensor: torch.Tensor):
    """
    Print a human-readable distribution of categories.
    
    Args:
        char_tensor: Tensor of character codes
        color_tensor: Tensor of color codes
    """
    counts = get_category_distribution(char_tensor, color_tensor)
    total = counts.sum().item()
    
    print("Category Distribution:")
    print("-" * 40)
    for category in NetHackCategory:
        count = counts[category.value].item()
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{category.name:<12}: {count:>6} ({percentage:>5.1f}%)")
    print("-" * 40)
    print(f"{'TOTAL':<12}: {total:>6} (100.0%)")


PASSABLE = FLOORS | UP_STAIRS | DOWN_STAIRS | TRAPS | (ITEMS - {(ord('`'), CLR_GRAY)}) | FURNITURE  # monsters treated separately

def compute_passability_and_safety(
    chars: torch.Tensor,   # [H,W] long
    colors: torch.Tensor,  # [H,W] long
    hero_y: int, hero_x: int,
    unknown_weight: float = 0.5,   # how much to trust unknown in-map tiles
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      pass8:      [8] float targets in {0,1} or 0.5 for unknown
      safe8:      [8] float targets in {0,1} or 0.5 for unknown
      hard_mask8: [8] {0,1}  -> 0 for off-map neighbors (don’t train/eval)
      weight8:    [8] >=0    -> per-dir weight (unknown gets 'unknown_weight')
    """
    H, W = chars.shape
    device = chars.device
    pass8      = torch.zeros(8, dtype=torch.float32, device=device)
    safe8      = torch.zeros(8, dtype=torch.float32, device=device)
    hard_mask8 = torch.zeros(8, dtype=torch.float32, device=device)
    weight8    = torch.zeros(8, dtype=torch.float32, device=device)
    for i,(dy,dx) in enumerate(DIRS_8):
        y, x = hero_y + dy, hero_x + dx
        if not (0 <= y < H and 0 <= x < W):
            continue
        ch = int(chars[y,x])
        cl = int(colors[y,x])
        hard_mask8[i] = 1.0
        
        # --- unknown in-map (screen shows padding pair inside the map)
        if ch == PADDING_CHAR and cl == PADDING_COLOR:
            pass8[i] = 0.5     # uncertain target
            safe8[i] = 0.5
            weight8[i] = unknown_weight
            continue
        # legal to step?
        legal = ((ch, cl) in PASSABLE) or is_monster(ch)
        if ch == ord('`') and cl == CLR_GRAY:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                ch2 = int(chars[yy,xx])
                cl2 = int(colors[yy,xx])
                legal = (ch2, cl2) not in (WALLS | DOORS)
            else:
                legal = False
        # safety: legal AND not a hazard/unknown/monster
        unsafe = ((ch, cl) in TRAPS) or ((ch, cl) in WATER_LAVA) or is_monster(ch)
        pass8[i] = 1.0 if legal else 0.0
        safe8[i] = 1.0 if (legal and not unsafe) else 0.0
        weight8[i] = 1.0
    return pass8, safe8, hard_mask8, weight8

# ---- Goal vector (nearest stairs) -------------------------------------------
def nearest_stairs_vector(chars: torch.Tensor,
                          hero_y: int, hero_x: int,
                          prefer: str = '>') -> tuple[float, float] | None:
    """Return (dx,dy) in [-1,1] from hero to nearest visible stairs (prefer '>' or '<')."""
    H, W = chars.shape
    goal_ch = ord(prefer)
    ys, xs = (chars == goal_ch).nonzero(as_tuple=True)
    if ys.numel() == 0:
        return None
    # choose Euclidean nearest
    dy = ys - hero_y
    dx = xs - hero_x
    i = (dy*dy + dx*dx).argmin().item()
    gx, gy = int(xs[i].item()), int(ys[i].item())
    # normalize to [-1,1]
    dxn = max(-1.0, min(1.0, (gx - hero_x) / (W - 1)))
    dyn = max(-1.0, min(1.0, (gy - hero_y) / (H - 1)))
    return (dyn, dxn)

# ---- Ego crop and class mapping ---------------------------------------------
def crop_ego(chars: torch.Tensor, colors: torch.Tensor, hero_y: int, hero_x: int, k: int) -> torch.Tensor:
    """Return ego crop [k,k] of ASCII codes, padded with spaces 32."""
    H, W = chars.shape
    r = k // 2
    out_chars = torch.full((k,k), fill_value=PADDING_CHAR, dtype=chars.dtype)  # spaces
    out_colors = torch.full((k,k), fill_value=PADDING_COLOR, dtype=colors.dtype)  # padding color
    y0, y1 = max(0, hero_y - r), min(H, hero_y + r + 1)
    x0, x1 = max(0, hero_x - r), min(W, hero_x + r + 1)
    oy0, oy1 = r - (hero_y - y0), r + (y1 - hero_y)
    ox0, ox1 = r - (hero_x - x0), r + (x1 - hero_x)
    out_chars[oy0:oy1, ox0:ox1] = chars[y0:y1, x0:x1]
    out_colors[oy0:oy1, ox0:ox1] = colors[y0:y1, x0:x1]
    return out_chars, out_colors
    
# ---- Discounted k-step returns ----------------------------------------------
def discounted_k_step_multi_with_mask(
    rewards: np.ndarray,      # [T] float
    done: np.ndarray,         # [T] bool  (True if episode terminates AT timestep t)
    horizons: list[int],      # e.g., [5, 10]
    gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      vals: [T, M] where vals[t,m] = sum_{i=0}^{k-1} gamma^i r_{t+i}
            computed ONLY if all r_{t..t+k-1} exist and done[u]==False for u in [t..t+k-1]
      mask: [T, M] where mask[t,m] = 1.0 if above condition holds, else 0.0
    Notes:
      - This avoids cross-batch leakage; steps whose k-window crosses the segment end are masked 0.
      - If an episode ends at t (done[t]==True), then no horizon starting at t is valid.
    """
    T = int(rewards.shape[0])
    M = len(horizons)
    vals = np.zeros((T, M), dtype=np.float32)
    mask = np.zeros((T, M), dtype=np.float32)

    for m, k in enumerate(horizons):
        for t in range(T):
            g, w = 0.0, 1.0
            ok = True
            for i in range(k):
                ti = t + i
                if ti >= T or done[ti]:
                    ok = False
                    break
                g += w * rewards[ti]
                w *= gamma
            vals[t, m] = g
            mask[t, m] = 1.0 if ok else 0.0
    return vals, mask



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
        Collect data from the dataset for VAE training and build decision-sufficient targets.

        - Adds: action_index, action_onehot, reward_target, done_target,
                passability_target, safety_target, has_next.
        - Shapes are [num_games, num_time, ...], consistent with your current pipeline.
        - z_next_detached is produced later during training by encoding obs_{t+1}; here we prepare has_next masks.
        - Ego view data (ego_char, ego_color, ego_class) is computed on-the-fly in vae_loss function for flexibility.
        - Goal and value targets are computed on-the-fly in vae_loss using VAEConfig parameters for flexibility.
        """
        print(f"Collecting {max_batches} batches from {dataset_name} for VAE training...")
        print(f"  - Batch size: {batch_size}, Sequence length: {seq_length}")
        print(f"  - Game IDs: {len(game_ids) if game_ids else 'All'}")
        
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
        total_samples = 0
        valid_samples = 0
        
        for batch_idx, minibatch in enumerate(dataset):
            if batch_count >= max_batches:
                break
                
            print(f"Processing batch {batch_idx + 1}/{max_batches}...")
            
            this_batch: Dict[str, torch.Tensor] = {}
            num_games, num_time = minibatch['tty_chars'].shape[:2]
            total_samples += num_games * num_time
            
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
            action_onehot_mb   = torch.zeros((num_games, num_time, ACTION_DIM), dtype=torch.float32)
            done_target_mb     = torch.zeros((num_games, num_time), dtype=torch.float32)
            passability_mb     = torch.zeros((num_games, num_time, 8), dtype=torch.float32)
            safety_mb          = torch.zeros((num_games, num_time, 8), dtype=torch.float32)
            hard_mask_mb       = torch.zeros((num_games, num_time, 8), dtype=torch.float32)
            weight_mb          = torch.zeros((num_games, num_time, 8), dtype=torch.float32)
            has_next_mb        = torch.zeros((num_games, num_time), dtype=torch.bool)

            # Convenience aliases to raw
            tty_chars = this_batch['tty_chars']    # [G,T, H,W]
            tty_colors= this_batch['tty_colors']   # [G,T, H,W]
            done_raw  = this_batch['done'].astype(bool) if isinstance(this_batch['done'], np.ndarray) else np.array(this_batch['done'], dtype=bool)
            done_raw  = torch.tensor(done_raw)     # [G,T] bool
            keycodes  = torch.tensor(this_batch['keypresses'], dtype=torch.int64)  # [G,T]
            
            reward_target_mb    = torch.tensor(this_batch['scores'], dtype=torch.float32) if 'scores' in this_batch else torch.zeros(num_games, num_time)
            action_index_mb = batch_keypress_static_map(keycodes) # [G,T] -> [G,T] with static mapping
            # Process each game in the batch
            for g in range(num_games):
                # Fill scalar targets and actions first (vectorized per game)  
                # One-hot actions
                action_onehot_mb[g] = one_hot(action_index_mb[g], ACTION_DIM)

                # Per-timestep spatial targets (passability/safety)
                for t in range(num_time):
                    done_target_mb[g,t]  = float(done_raw[g,t].item())
                    has_next_mb[g,t] = (t+1 < num_time) and (not bool(done_raw[g,t].item()))
                    game_id = int(this_batch['gameids'][g,t])
                    chars = tty_chars[g,t]     # [H,W] int
                    colors= tty_colors[g,t]
                    is_valid_map = detect_valid_map(chars)
                    if not is_valid_map:
                        valid_screen_minibatch[g,t] = False
                        continue

                    game_map = get_game_map(chars, colors)
                    chars_map = game_map['chars']    # torch.[H,W] long
                    colors_map= game_map['colors']   # torch.[H,W] long
                    
                    # Extract text information
                    current_message = get_current_message(chars)
                    message_chars = torch.tensor([ord(c) for c in current_message.ljust(256, chr(0))], dtype=torch.long)
                    status_lines = get_status_lines(chars)
                    status_chars = chars[22:, :]
                    
                    hero_info = self.get_hero_info(game_id, current_message)
                    if hero_info is None:
                        print(f"⚠️ Warning: No hero info found for game {game_id}, skipping sample")
                        # print(tty_render(chars, colors))  # Only print TTY for debugging when needed
                        valid_screen_minibatch[g,t] = False
                        continue
                    
                    score = reward_target_mb[g,t].item()
                    
                    # Fill in the minibatch tensors
                    message_chars_minibatch[g, t] = message_chars
                    game_chars_minibatch[g, t] = game_map['chars']
                    game_colors_minibatch[g, t] = game_map['colors']
                    status_chars_minibatch[g, t] = status_chars
                    hero_info_minibatch[g, t] = hero_info
                    blstats_minibatch[g, t] = adapter(game_map['chars'], status_lines, score)

                    # hero pos: prefer '@', else tty_cursor if within bounds, else last seen
                    ys, xs = (chars_map == ord('@')).nonzero(as_tuple=True)
                    if ys.numel() > 0:
                        hy, hx = int(ys[0].item()), int(xs[0].item())
                    else:
                        valid_screen_minibatch[g,t] = False
                        continue
                    
                    valid_samples += 1

                    # passability & safety
                    p8, s8, hm8, w8 = compute_passability_and_safety(chars_map, colors_map, hy, hx)
                    passability_mb[g,t] = p8
                    safety_mb[g,t]      = s8
                    hard_mask_mb[g,t]   = hm8
                    weight_mb[g,t]      = w8

                # Note: goal_target, goal_mask, and value_k_target are now computed on-the-fly in vae_loss

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
            # value_k_target, goal_target, and goal_mask are now computed on-the-fly in vae_loss
            this_batch['passability_target']= passability_mb
            this_batch['safety_target']     = safety_mb
            this_batch['hard_mask']         = hard_mask_mb
            this_batch['weight']            = weight_mb
            this_batch['has_next']          = has_next_mb

            collected_batches.append(this_batch)
            batch_count += 1
            
            # Optional: Clean up intermediate tensors to save memory
            if batch_count % 10 == 0:
                print(f"   💾 Processed {batch_count}/{max_batches} batches, {valid_samples}/{total_samples} valid samples ({100*valid_samples/total_samples:.1f}%)")

        print(f"✅ Successfully collected {len(collected_batches)} batches from {batch_count} processed batches")
        print(f"   📊 Total samples: {total_samples}, Valid samples: {valid_samples} ({100*valid_samples/total_samples:.1f}%)")
        print(f"   🎯 Average batch size: {total_samples/len(collected_batches):.1f} samples")
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

# Utility functions for handling actions in the Nethack environment.
# In Nethack, actions have integer values representing unicode code of a specified character
# but we use indices of the actions to perform in each step.
# so we have one-to-one mapping between (action, index, description).
from nle.nethack.actions import action_id_to_type
from nle import nethack
from enum import IntEnum
import gymnasium as gym
from typing import Union, List, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import torch

# Global hash table for efficient keypress to action index mapping
# This is a dictionary (hash table) that provides O(1) average lookup time
# Key: ASCII keypress value (int), Value: action index (int)
KEYPRESS_INDEX_MAPPING: Dict[int, int] = {}

def _initialize_keypress_mapping() -> None:
    """
    Initialize the global keypress mapping hash table.
    This function is called automatically when the module is imported.
    """
    global KEYPRESS_INDEX_MAPPING
    if not KEYPRESS_INDEX_MAPPING:  # Only initialize if empty
        KEYPRESS_INDEX_MAPPING = {
            action.value: index 
            for index, action in enumerate(nethack.ACTIONS)
        }

# Initialize the mapping when module is imported
_initialize_keypress_mapping()

def get_keypress_mapping() -> Dict[int, int]:
    """
    Get the global keypress to action index mapping hash table.
    This function provides access to the mapping from other modules.
    
    Returns:
        Dict[int, int]: Hash table mapping keypress values to action indices
    
    Example:
        >>> from utils.action_utils import get_keypress_mapping
        >>> mapping = get_keypress_mapping()
        >>> north_action_index = mapping[107]  # 'k' for North
    """
    if not KEYPRESS_INDEX_MAPPING:
        _initialize_keypress_mapping()
    return KEYPRESS_INDEX_MAPPING.copy()  # Return a copy to prevent external modification

def is_valid_keypress(keypress: int) -> bool:
    """
    Check if a keypress is a valid NetHack action.
    Uses O(1) hash table lookup for efficiency.
    
    Args:
        keypress: ASCII value of the keypress
    
    Returns:
        bool: True if keypress is valid, False otherwise
    
    Example:
        >>> is_valid_keypress(107)  # 'k' for North
        True
        >>> is_valid_keypress(999)  # Invalid keypress
        False
    """
    return keypress in KEYPRESS_INDEX_MAPPING

def get_action_index_fast(keypress: int, default: int = 0) -> int:
    """
    Get action index for a keypress with a default fallback.
    Uses O(1) hash table lookup with no exception handling for maximum speed.
    
    Args:
        keypress: ASCII value of the keypress
        default: Default value to return if keypress is invalid
    
    Returns:
        int: Action index or default value
    
    Example:
        >>> get_action_index_fast(107)  # 'k' for North
        0
        >>> get_action_index_fast(999, default=-1)  # Invalid keypress
        -1
    """
    return KEYPRESS_INDEX_MAPPING.get(keypress, default)

def get_valid_keypresses() -> List[int]:
    """
    Get all valid keypress values (ASCII codes) that map to NetHack actions.
    
    Returns:
        List[int]: Sorted list of valid keypress ASCII values
    
    Example:
        >>> valid_keys = get_valid_keypresses()
        >>> print(f"Total valid keypresses: {len(valid_keys)}")
        >>> print(f"First few: {valid_keys[:5]}")
    """
    return sorted(KEYPRESS_INDEX_MAPPING.keys())

def get_mapping_stats() -> Dict[str, Union[int, List[int]]]:
    """
    Get statistics about the keypress mapping hash table.
    
    Returns:
        Dict containing mapping statistics
    
    Example:
        >>> stats = get_mapping_stats()
        >>> print(f"Total mappings: {stats['total_mappings']}")
    """
    valid_keys = list(KEYPRESS_INDEX_MAPPING.keys())
    return {
        'total_mappings': len(KEYPRESS_INDEX_MAPPING),
        'min_keypress': min(valid_keys) if valid_keys else 0,
        'max_keypress': max(valid_keys) if valid_keys else 0,
        'printable_chars': [k for k in valid_keys if 32 <= k <= 126],
        'control_chars': [k for k in valid_keys if k < 32 or k > 126]
    }

def keypress_static_map(keypress: int) -> int:
    """
    Convert a keypress (ASCII value) to its corresponding action index in the NetHack environment.
    Uses O(1) hash table lookup for maximum efficiency.
    
    Args:
        keypress: ASCII value of the keypress (e.g., 107 for 'k' which means North)
    
    Returns:
        int: Action index that can be used with env.step()
    
    Raises:
        KeyError: If keypress is not a valid NetHack action
    
    Example:
        >>> index = keypress_static_map(107)  # 'k' for North
        >>> print(index)  # Should print the action index for North
    """
    try:
        return KEYPRESS_INDEX_MAPPING[keypress]
    except KeyError:
        raise KeyError(f"Keypress {keypress} is not a valid action in NetHack. "
                      f"Use is_valid_keypress({keypress}) to check validity first.")

def batch_keypress_static_map(keypresses: Union[List[int], 'torch.Tensor']) -> 'torch.Tensor':
    """ Convert a batch of keypresses to action indices using a static mapping.
    Supports n-dimensional tensors and preserves the original shape.
    Uses optimized hash table lookup for maximum efficiency.
 
    Args:
        keypresses: List or tensor of keypress values (ASCII codes). 
                   Can be any shape (1D, 2D, 3D, etc.)
    
    Returns:
        torch.Tensor: Tensor of action indices with same shape as input
    
    Example:
        >>> import torch
        >>> # 1D tensor
        >>> keypresses_1d = [107, 106, 108, 104]  # k, j, l, h (North, South, East, West)
        >>> action_indices = batch_keypress_static_map(keypresses_1d)
        >>> print(action_indices)  # Shape: [4]
        >>> 
        >>> # 2D tensor (batch_size=2, seq_length=3)
        >>> keypresses_2d = torch.tensor([[107, 106, 108], [104, 46, 121]])
        >>> action_indices = batch_keypress_static_map(keypresses_2d)
        >>> print(action_indices.shape)  # Shape: [2, 3]
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for batch processing. Please install torch.")
    
    if isinstance(keypresses, list):
        keypresses = torch.tensor(keypresses, dtype=torch.long)
    elif not isinstance(keypresses, torch.Tensor):
        # Handle numpy arrays and other array-like objects
        keypresses = torch.tensor(keypresses, dtype=torch.long)
    else:
        # Ensure keypresses is long tensor for proper indexing
        keypresses = keypresses.long()
    
    # Create a lookup tensor for efficient mapping using the hash table
    # This is more efficient than recreating the mapping each time
    max_keypress = max(KEYPRESS_INDEX_MAPPING.keys()) if KEYPRESS_INDEX_MAPPING else 255
    lookup_size = max(256, max_keypress + 1)  # Ensure we cover all possible ASCII values
    
    # Initialize with zeros (fallback for invalid keypresses)
    lookup_table = torch.zeros(lookup_size, dtype=torch.long)
    
    # Fill the lookup table using the hash table - O(n) where n is number of valid actions
    for keypress_value, action_index in KEYPRESS_INDEX_MAPPING.items():
        if 0 <= keypress_value < lookup_size:
            lookup_table[keypress_value] = action_index
    
    # Clamp keypresses to valid range to prevent indexing errors
    valid_keypresses = torch.clamp(keypresses, 0, lookup_size - 1)
    
    # Use advanced indexing for efficient batch conversion
    # This preserves the original tensor shape automatically
    return lookup_table[valid_keypresses]

def keypress_to_action_index(env: gym.Env, keypress: int) -> int:
    """
    Convert a keypress (ASCII value) to its corresponding action index in the NetHack environment.
    
    Args:
        env: NetHack gym environment
        keypress: ASCII value of the keypress (e.g., 107 for 'k' which means North)
    
    Returns:
        int: Action index that can be used with env.step()
    
    Example:
        >>> env = gym.make("NetHackChallenge-v0")
        >>> index = keypress_to_action_index(env, 107)  # 'k' for North
        >>> obs, reward, done, info = env.step(index)
    """
    if not isinstance(env, gym.Env):
        raise TypeError("The environment must be an instance of gym.Env")
    if not hasattr(env.unwrapped, 'actions'):
        raise ValueError("The environment does not have an 'actions' attribute")
    if not isinstance(keypress, int):
        raise TypeError("Keypress must be an integer (ASCII value)")
    if keypress < 0 or keypress > 255:
        raise ValueError("Keypress must be a valid ASCII value (0-255)")
    
    # Find the action that corresponds to this keypress
    for i, action in enumerate(env.unwrapped.actions):
        if action.value == keypress:
            return i
    
    # If no exact match found, return 0 (usually corresponds to a no-op or default action)
    # This handles cases where the keypress doesn't correspond to a valid action
    return 0

def batch_keypress_to_action_index(env: gym.Env, keypresses) -> 'torch.Tensor':
    """
    Convert a batch of keypresses to action indices using a precomputed embedding.
    This is more efficient for batch processing than calling keypress_to_action_index repeatedly.
    
    Args:
        env: NetHack gym environment  
        keypresses: Tensor or array of keypress values (ASCII codes)
    
    Returns:
        torch.Tensor: Tensor of action indices
    
    Example:
        >>> import torch
        >>> env = gym.make("NetHackChallenge-v0")
        >>> keypresses = torch.tensor([107, 106, 108, 104])  # k, j, l, h (North, South, East, West)
        >>> action_indices = batch_keypress_to_action_index(env, keypresses)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for batch processing. Please install torch.")
    
    if not isinstance(env, gym.Env):
        raise TypeError("The environment must be an instance of gym.Env")
    if not hasattr(env.unwrapped, 'actions'):
        raise ValueError("The environment does not have an 'actions' attribute")
    
    # Create embedding table: ASCII value -> action index
    embed_actions = torch.zeros((256, 1))
    for i, action in enumerate(env.unwrapped.actions):
        embed_actions[action.value][0] = i
    
    # Convert to embedding layer
    embed_layer = torch.nn.Embedding.from_pretrained(embed_actions)
    
    # Convert keypresses to tensor if needed
    if not isinstance(keypresses, torch.Tensor):
        keypresses = torch.tensor(keypresses, dtype=torch.long)
    
    # Apply embedding and squeeze
    action_indices = embed_layer(keypresses).squeeze(-1).long()
    
    return action_indices
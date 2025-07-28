import numpy as np
import torch

def separate_tty_components(tty_chars, tty_colors):
    """
    Separate TTY terminal into its components.
    
    Args:
        tty_chars: np.array shape (24, 80)
        tty_colors: np.array shape (24, 80) 
    
    Returns:
        dict with separated components
    """
    messages = get_current_message(tty_chars)
    # convert messages to ASCI characters and pad to 256 characters
    messages_chars = np.array([ord(c) for c in messages.ljust(256, chr(0))])
    game_map = get_game_map(tty_chars, tty_colors)
    return {
        # Message line (top of screen)
        'message_chars': messages_chars,      # Shape: (256,)

        # Game map (matches obs["chars"] and obs["colors"])
        'game_chars': game_map['chars'],   # Shape: (21, 79)
        'game_colors': game_map['colors'], # Shape: (21, 79)

        # Status lines (bottom of screen)
        'status_chars': tty_chars[22:24, :],   # Shape: (2, 80)
    }

# Detect where message ends and map begins
def detect_message_map_boundary(tty_chars):
    """
    Detect the boundary between message lines and the game map.
    
    NetHack display layout:
    - Row 0: Always message
    - Rows 1-2: Can be message (if multi-line) or start of map
    - Rows 1-21: Usually the map area (21x79)
    - Rows 22-23: Status lines
    
    Returns:
        int: The row index where the map starts (typically 1, 2, or 3)
    """
    height, width = tty_chars.shape
    
    # Row 0 is always message, so map starts at least at row 1
    map_start_row = 3
    
    # Check rows 1 and 2 to see if they contain map-like content
    for check_row in [1, 2]:
        row_chars = tty_chars[check_row, :]
        row_text = ''.join(chr(c) if 32 <= c <= 126 else ' ' for c in row_chars)
        
        # Heuristics to detect if this row is part of the map:
        # 1. Contains typical map characters: walls (#), floors (.), doors (+), etc.
        # 2. Has a pattern typical of dungeon layout
        # 3. Is not predominantly text (which would indicate a message)
        
        map_chars = set('#.-+|<>^')  # Common NetHack map symbols
        map_char_count = sum(1 for c in row_text if c in map_chars)
        
        # Count non-space characters
        non_space_count = sum(1 for c in row_text if c != ' ')
        
        # If this row has a high ratio of map characters to total characters,
        # or if it looks like a typical dungeon row, it's likely the map start
        if non_space_count > 0:
            map_char_ratio = map_char_count / non_space_count

            # If >50% of non-space chars are map symbols, likely map
            if map_char_ratio > 0.5:
                map_start_row = check_row
                break
                
            # Also check for typical dungeon patterns (walls, rooms, etc.)
            # Look for sequences of # (walls) or . (floors)
            if ('###' in row_text or '...' in row_text or 
                '---' in row_text or '|||' in row_text):
                map_start_row = check_row
                break
        elif non_space_count == 0:
            map_start_row = check_row
            break
    
    return map_start_row

# Extract full message (potentially multi-line)
def get_current_message(tty_chars):
    """
    Extract the complete message, which may span multiple lines.
    Uses boundary detection to determine where message ends and map begins.
    """
    map_start_row = detect_message_map_boundary(tty_chars)
    
    # Collect all message lines (from row 0 up to map start)
    message_lines = []
    for row_idx in range(map_start_row):
        if row_idx >= tty_chars.shape[0]:
            break
        message_row = tty_chars[row_idx, :]
        line_text = ''.join(chr(c) if 32 <= c <= 126 else ' ' for c in message_row).strip()
        if line_text:  # Only add non-empty lines
            message_lines.append(line_text)
    
    # Join all message lines with space
    return ' '.join(message_lines)

# Extract status info  
def get_status_lines(tty_chars):
    status = tty_chars[22:24, :]
    lines = []
    for row in status:
        line = ''.join(chr(c) if 32 <= c <= 126 else ' ' for c in row).strip()
        lines.append(line)
    return lines

# The game map for AI processing
def get_game_map(tty_chars, tty_colors):
    """
    Extract the game map portion, dynamically detecting where it starts.
    """
    map_start_row = detect_message_map_boundary(tty_chars)
    
    # Calculate map dimensions (typically 21 rows, 79 columns)
    # Map ends 2 rows before the bottom (leaving space for status)
    map_end_row = min(tty_chars.shape[0] - 2, map_start_row + 21)
    map_width = min(tty_chars.shape[1], 79)
    
    # We pad on top of the map to make it 21 times 79. We pad it with spaces
    if map_start_row > 1:
        padding = torch.full((map_start_row - 1, map_width), ord(' '), dtype=torch.int32, device=tty_chars.device)
        tty_chars_new = torch.cat([padding, tty_chars[map_start_row:map_end_row, :map_width]], dim=0)
        color_padding = torch.zeros_like(padding, dtype=tty_colors.dtype)
        tty_colors_new = torch.cat([color_padding, tty_colors[map_start_row:map_end_row, :map_width]], dim=0)
    else:
        tty_chars_new = tty_chars[map_start_row:map_end_row, :map_width]
        tty_colors_new = tty_colors[map_start_row:map_end_row, :map_width]
    return {
        'chars': tty_chars_new,
        'colors': tty_colors_new,
        'map_start_row': map_start_row,  # Include boundary info for debugging
        'map_end_row': map_end_row,
    }


def detect_valid_map(tty_chars):
    """
    Detect if the current tty_chars represents a valid map.
    
    Args:
        tty_chars: np.array shape (24, 80)
    
    Returns:
        bool: True if valid map, False otherwise
    """
    map_start_row = detect_message_map_boundary(tty_chars)
    
    game_map = tty_chars[map_start_row:22, :-1]  # Extract the map area 
    
    # collect a set of characters in the map
    map_chars = set()
    for row in game_map:
        for c in row:
            if 32 <= c <= 126:
                map_chars.add(chr(c))

    # Check for required characters in the map
    required_chars = {'.', '@', '-', '|'}
    if not required_chars.issubset(map_chars):
        return False

    return True

if __name__ == "__main__":
    import gymnasium as gym
    import minihack
    from action_utils import available_actions
    
    env = gym.make("MiniHack-Room-5x5-v0")
    env.reset()
    available_actions_list = available_actions(env)
    print("Available actions:", available_actions_list)
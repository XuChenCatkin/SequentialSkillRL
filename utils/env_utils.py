import numpy as np

def separate_tty_components(tty_chars, tty_colors):
    """
    Separate TTY terminal into its components.
    
    Args:
        tty_chars: np.array shape (24, 80)
        tty_colors: np.array shape (24, 80) 
    
    Returns:
        dict with separated components
    """
    return {
        # Message line (top of screen)
        'message_chars': tty_chars[0, :],      # Shape: (80,)
        'message_colors': tty_colors[0, :],    # Shape: (80,)
        
        # Game map (matches obs["chars"] and obs["colors"])
        'game_chars': tty_chars[1:22, 0:79],   # Shape: (21, 79)
        'game_colors': tty_colors[1:22, 0:79], # Shape: (21, 79)
        
        # Status lines (bottom of screen)
        'status_chars': tty_chars[22:24, :],   # Shape: (2, 80)
        'status_colors': tty_colors[22:24, :], # Shape: (2, 80)
    }

# Extract message
def get_current_message(tty_chars):
    message_row = tty_chars[0, :]
    return ''.join(chr(c) if 32 <= c <= 126 else ' ' for c in message_row).strip()

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
    return {
        'chars': tty_chars[1:22, 0:79],    # Same as obs["chars"]
        'colors': tty_colors[1:22, 0:79],  # Same as obs["colors"]
    }

if __name__ == "__main__":
    import gymnasium as gym
    import minihack
    from action_utils import available_actions
    
    env = gym.make("MiniHack-Room-5x5-v0")
    env.reset()
    available_actions_list = available_actions(env)
    print("Available actions:", available_actions_list)
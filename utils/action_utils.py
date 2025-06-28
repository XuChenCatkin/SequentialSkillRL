# Utility functions for handling actions in the Nethack environment.
# In Nethack, actions have integer values representing unicode code of a specified character
# but we use indices of the actions to perform in each step.
# so we have one-to-one mapping between (action, index, description).
from nle.nethack.actions import action_id_to_type
from nle import nethack
from enum import IntEnum

def name_to_action(action_name: str) -> IntEnum:
    """
    Convert an action name to its corresponding action ID
    """
    for action in nethack.ACTIONS:
        if action.name == action_name:
            return action
    raise ValueError(f"Action name '{action_name}' not found in Nethack actions")

def action_to_index(action):
    """
    Convert an action to its corresponding index used for the Nethack environment step
    """
    if isinstance(action, IntEnum):
        return nethack.ACTIONS.index(action)
    elif isinstance(action, str):
        return nethack.ACTIONS.index(name_to_action(action))
    else:
        raise ValueError(f"Unsupported action type: {type(action)}")
    
def index_to_action(index: int) -> IntEnum:
    """
    Convert an index to its corresponding action
    """
    if index < 0 or index >= len(nethack.ACTIONS):
        raise ValueError(f"Index {index} is out of bounds for actions")
    return nethack.ACTIONS[index]

def action_full_name(action: IntEnum) -> str:
    """
    Get the full name of an action
    """
    if action not in nethack.ACTIONS:
        raise ValueError(f"Action {action} is not a valid Nethack action")
    return action_id_to_type(action)

def action_abbr_name(action: IntEnum) -> str:
    """
    Get the abbreviated name of an action
    """
    if action not in nethack.ACTIONS:
        raise ValueError(f"Action {action} is not a valid Nethack action")
    return action.name
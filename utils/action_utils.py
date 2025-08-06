# Utility functions for handling actions in the Nethack environment.
# In Nethack, actions have integer values representing unicode code of a specified character
# but we use indices of the actions to perform in each step.
# so we have one-to-one mapping between (action, index, description).
from nle.nethack.actions import action_id_to_type
from nle import nethack
from enum import IntEnum
import gymnasium as gym
from typing import Union, List

def name_to_action(action_name: str) -> IntEnum:
    """
    Convert an action name to its corresponding action ID
    """
    if not isinstance(action_name, str):
        raise ValueError("Action name must be a string")
    for action in nethack.ACTIONS:
        if action.name.lower() == action_name.lower():
            return action
    raise ValueError(f"Action name '{action_name}' not found in Nethack actions")

def available_actions(env: gym.Env) -> List[IntEnum]:
    """
    Get a list of available actions in the Nethack environment
    """
    if not isinstance(env, gym.Env):
        raise TypeError("The environment must be an instance of gym.Env")
    if not hasattr(env.unwrapped, 'actions'):
        raise ValueError("The environment does not have an 'actions' attribute")
    return list(env.unwrapped.actions)

def action_to_index(env: gym.Env, action: Union[IntEnum, str]) -> int:
    """
    Convert an action to its corresponding index used for the Nethack environment step
    """
    if not isinstance(env, gym.Env):
        raise TypeError("The environment must be an instance of gym.Env")
    if not hasattr(env.unwrapped, 'actions'):
        raise ValueError("The environment does not have an 'actions' attribute")
    if isinstance(action, str):
        if action.isdigit():
            action_value = int(action)
        else:
            action_value = name_to_action(action)
    elif isinstance(action, IntEnum):
        action_value = action
    else:
        raise TypeError("Action must be an IntEnum or a string representing the action name")
    available_actions = env.unwrapped.actions
    if action_value not in available_actions:
        raise ValueError(f"Action {action_value} is not available in the environment")
    else:
        return available_actions.index(action_value)
    
def index_to_action(env: gym.Env, index: int) -> IntEnum:
    """
    Convert an index to its corresponding action
    """
    if not isinstance(env, gym.Env):
        raise TypeError("The environment must be an instance of gym.Env")
    if not hasattr(env.unwrapped, 'actions'):
        raise ValueError("The environment does not have an 'actions' attribute")
    if not isinstance(index, int):
        raise TypeError("Index must be an integer")
    available_actions = env.unwrapped.actions
    if index < 0 or index >= len(available_actions):
        raise ValueError(f"Index {index} is out of bounds for actions")
    return available_actions[index]

def action_full_name(action: IntEnum) -> str:
    """
    Get the full name of an action
    """
    if not isinstance(action, IntEnum):
        raise TypeError("Action must be an IntEnum")
    if action not in nethack.ACTIONS:
        raise ValueError(f"Action {action} is not a valid Nethack action")
    return action_id_to_type(action)

def action_abbr_name(action: IntEnum) -> str:
    """
    Get the abbreviated name of an action
    """
    if not isinstance(action, IntEnum):
        raise TypeError("Action must be an IntEnum")
    if action not in nethack.ACTIONS:
        raise ValueError(f"Action {action} is not a valid Nethack action")
    return action.name
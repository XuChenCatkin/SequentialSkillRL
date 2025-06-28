import gymnasium as gym
import minihack
from action_utils import action_to_index, index_to_action, available_actions

if __name__ == "__main__":
    env = gym.make("MiniHack-Room-5x5-v0")
    env.reset()
    available_actions_list = available_actions(env)
    print("Available actions:", available_actions_list)
import gymnasium as gym
import minihack

if __name__ == "__main__":
    env = gym.make("MiniHack-Room-5x5-v0")
    env.reset()
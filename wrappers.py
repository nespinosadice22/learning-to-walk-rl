import gymnasium as gym
import numpy as np


class TorqueCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, total_episodes: int = 200, min_scale: float = 5e-1):
        super().__init__(env)
        self.total_episodes = total_episodes
        self.min_scale = min_scale
        self.current_episode = 0
        self.torque_scale = min_scale

    def reset(self, **kwargs):
        self.current_episode += 1
        progress = min(1.0, self.current_episode / self.total_episodes)
        self.torque_scale = self.min_scale + progress * (1.0 - self.min_scale)
        return self.env.reset(**kwargs)

    def step(self, action):
        scaled_action = action * self.torque_scale
        return self.env.step(scaled_action)

import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed

from wrappers import TorqueCurriculumWrapper

def make_env(env_id: str, rank: int, seed: int = 0, starting_pos: str = "standing"):
    def _init():
        if starting_pos == "knees":
            env = gym.make(env_id, starting_pos=starting_pos, xml_file="./environments/assets/humanoid_w_box.xml", healthy_z_range=(0.75, 2.0))
        else:
            env = gym.make(env_id, starting_pos=starting_pos)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


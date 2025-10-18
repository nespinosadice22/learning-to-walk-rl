#can't record on della (i think) so if you want to see use this 
import os
import typer
import numpy as np
import optuna
import time
import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


import wandb
from wandb.integration.sb3 import WandbCallback


from model import NGNActorCriticPolicy

register(
    id="HumanoidGetUp-v0",
    entry_point="environments.sitting:HumanoidGetUp",
    max_episode_steps=2000,
)


def record_videos(model_path, norm_save_path, starting_pos, boxes, video_folder, final_arch, seed: int):

    def make_test_env(seed_value):
        if boxes: 
            env = gym.make(
                "HumanoidGetUp-v0",
                starting_pos=starting_pos, 
                render_mode="rgb_array",
                terminate_when_unhealthy=False,
                xml_file="./environments/assets/humanoid_w_2_boxes.xml" 
            )
        else: 
             env = gym.make(
                "HumanoidGetUp-v0",
                starting_pos=starting_pos, 
                render_mode="rgb_array",
                terminate_when_unhealthy=False,
            )
        env.reset(seed=seed_value)
        return env

    for ep in range(5):
        new_seed = seed + ep
        test_env = DummyVecEnv([lambda new_seed=new_seed: make_test_env(new_seed)])
        test_env = VecNormalize.load(norm_save_path, test_env)
        test_env.training = False
        test_env.norm_reward = False

        name_prefix = f"video_seed_{new_seed}"
        video_env = VecVideoRecorder(
            test_env,
            video_folder=video_folder,
            record_video_trigger=lambda episode_id: True,
            video_length=10000,
            name_prefix=name_prefix
        )
        #model = PPO.load(model_path, env=test_env)
        FINAL_ARCH = final_arch
        CUSTOM_OBJECTS = {
            "policy_kwargs": dict(n_units=FINAL_ARCH, optimizer_kwargs={"weight_decay": 0.0}),
        }
        model = PPO.load(model_path, env=test_env, custom_objects=CUSTOM_OBJECTS)

        obs = video_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = video_env.step(action)
            if done:
                break
        video_env.close()

import os
import re

# ---------------------------------------------------------------------------
#  Parse directory names of the form
#  “…/<start>to<hidden>_lastlayer<last>_layers<N>_seed<S>_lrdc_<…>”
#  The “lastlayer<last>” part is optional.
#  Returns: hidden_size, lastlayer, num_layers, seed, archs[]
# ---------------------------------------------------------------------------
def parse_width_dir(base_dir: str):
    leaf = os.path.basename(base_dir.rstrip("/"))

    # hidden size (the number after “to”)
    m = re.search(r"to(\d+)", leaf)
    if not m:
        raise ValueError(f"Cannot find 'to<hidden>' segment in {leaf}")
    hidden_size = int(m.group(1))

    # optional last layer width
    m = re.search(r"lastlayer(\d+)", leaf)
    if m:
        lastlayer = int(m.group(1))
    else:
        # fall-back heuristic: quarter of hidden_size (e.g. 128 → 32, 64 → 16)
        lastlayer = hidden_size // 4

    # number of layers
    m = re.search(r"layers(\d+)", leaf)
    if not m:
        raise ValueError(f"Cannot find 'layers<N>' segment in {leaf}")
    num_layers = int(m.group(1))

    # seed
    m = re.search(r"seed(\d+)", leaf)
    seed = int(m.group(1)) if m else None

    # build full architecture: [hidden_size, …, hidden_size, lastlayer]
    archs = [hidden_size] * (num_layers - 1) + [lastlayer]

    return hidden_size, lastlayer, num_layers, seed, archs

if __name__ == "__main__":

    base_dir = "./new_standing_expanding_ablations/width/4to64_lastlayer64_layers4_seed92_lrdc_0.7"
    archs = [64, 64, 64, 64]
    seed = 92
    starting_pos = "standing"
    boxes = False 
    model_path = f"{base_dir}/final_model"
    norm_save_path = f"{base_dir}/vecnormalize_train.pkl"
    video_save_path = f"{base_dir}/videos_{starting_pos}"
    os.makedirs(video_save_path, exist_ok=True)
    record_videos(model_path, norm_save_path, starting_pos, boxes, video_save_path, archs, seed)   


    base_dir = "./new_standing_expanding_ablations/width/4to64_lastlayer64_layers4_seed90_lrdc_0.7"
    archs = [64, 64, 64, 64]
    seed = 90
    starting_pos = "standing"
    boxes = False 
    model_path = f"{base_dir}/final_model"
    norm_save_path = f"{base_dir}/vecnormalize_train.pkl"
    video_save_path = f"{base_dir}/videos_{starting_pos}"
    os.makedirs(video_save_path, exist_ok=True)
    record_videos(model_path, norm_save_path, starting_pos, boxes, video_save_path, archs, seed)

    
    base_dir = "./new_standing_expanding_ablations/width/4to64_lastlayer64_layers4_seed91_lrdc_0.7"
    archs = [64, 64, 64, 64]
    seed = 91
    starting_pos = "standing"
    boxes = False 
    model_path = f"{base_dir}/final_model"
    norm_save_path = f"{base_dir}/vecnormalize_train.pkl"
    video_save_path = f"{base_dir}/videos_{starting_pos}"
    os.makedirs(video_save_path, exist_ok=True)
    record_videos(model_path, norm_save_path, starting_pos, boxes, video_save_path, archs, seed)     


import typer
import gymnasium as gym
from gymnasium.envs.registration import register
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.utils import set_random_seed

from model import NGNActorCriticPolicy

register(
    id="HumanoidGetUp-v0",
    entry_point="environments.sitting:HumanoidGetUp",
    max_episode_steps=2000,
)


def make_env(env_id: str, rank: int, seed: int = 0, starting_pos: str = "standing"):
    def _init():
        if starting_pos == "knees":
            env = gym.make(env_id, starting_pos=starting_pos, xml_file="./environments/assets/humanoid_w_box.xml", healthy_z_range=(0.75, 2.0))
        elif starting_pos == "standing":
            env = gym.make(env_id, starting_pos=starting_pos, xml_file="./environments/assets/humanoid_w_2_boxes.xml", healthy_z_range=(0.75, 2.0))
        elif starting_pos == "prone":
            env = gym.make(env_id, starting_pos=starting_pos, terminate_when_unhealthy=False)
        else:
            env = gym.make(env_id, starting_pos=starting_pos)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def create_train_env(env_id: str, starting_pos: str, seed: int, n_envs: int):
    env_fns = [make_env(env_id, i, seed, starting_pos=starting_pos) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

def create_val_env(env_id: str, starting_pos: str):
    if starting_pos == "knees":
        env = DummyVecEnv([lambda: Monitor(gym.make(env_id, starting_pos=starting_pos, xml_file="./environments/assets/humanoid_w_box.xml", healthy_z_range=(0.75, 2.0)))])
    elif starting_pos == "standing":
        env = DummyVecEnv([lambda: Monitor(gym.make(env_id, starting_pos=starting_pos, xml_file="./environments/assets/humanoid_w_2_boxes.xml", healthy_z_range=(0.75, 2.0)))])
    elif starting_pos == "prone":
        env = DummyVecEnv([lambda: Monitor(gym.make(env_id, starting_pos=starting_pos, terminate_when_unhealthy=False))])
    else:
        env = DummyVecEnv([lambda: Monitor(gym.make(env_id, starting_pos=starting_pos))])
    return VecNormalize(env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

def create_eval_callback(val_env, algo: str, hidden_size: int, learning_rate: float, seed: int, pos: bool, eval_freq: int, n_envs: int):
    return EvalCallback(
        val_env,
        best_model_save_path=f"./logs/{algo.lower()}_{hidden_size}_{learning_rate:.0e}_{seed}_{pos}/best_model/",
        log_path=f"./logs/{algo.lower()}_{hidden_size}_{learning_rate:.0e}_{seed}_{pos}/",
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        verbose=False,
    )

def record_videos(model, norm_save_path: str, hidden_size: int, seed: int):
    video_folder = "./videos"
    os.makedirs(video_folder, exist_ok=True)

    def make_test_env(seed_value):
        env = gym.make(
            "HumanoidGetUp-v0",
            starting_pos="knees",
            render_mode="rgb_array",
            terminate_when_unhealthy=False,
            xml_file="./environments/assets/humanoid_w_box.xml"
        )
        env.reset(seed=seed_value)
        return env

    for ep in range(10):
        new_seed = seed + ep
        test_env = DummyVecEnv([lambda new_seed=new_seed: make_test_env(new_seed)])
        test_env = VecNormalize.load(norm_save_path, test_env)
        test_env.training = False
        test_env.norm_reward = False

        name_prefix = f"{hidden_size}_{ep}_{int(time.time())}"
        video_env = VecVideoRecorder(
            test_env,
            video_folder=video_folder,
            record_video_trigger=lambda episode_id: True,
            video_length=2000,
            name_prefix=name_prefix
        )

        obs = video_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = video_env.step(action)
            if done:
                break
        video_env.close()

def main(seed: int = 0, env_id: str = "HumanoidGetUp-v0", n_envs: int = 8, eval_freq: int = 10000, algo: str = "ppo", hidden_size: int = 32, learning_rate: float = 1e-4):
    train_env = create_train_env(env_id, "prone", seed, n_envs)
    val_env = create_val_env(env_id, "prone")
    eval_callback = create_eval_callback(val_env, algo, hidden_size, learning_rate, seed, "prone", eval_freq, n_envs)

    model1 = PPO(
        NGNActorCriticPolicy,
        train_env,
        batch_size=256,
        ent_coef=0.01,
        learning_rate=learning_rate,
        n_steps=int(2048 / n_envs),
        policy_kwargs={"optimizer_kwargs": {"weight_decay": 1e-4}, "n_units": [8, 8, 16],},
    )
    model1.learn(total_timesteps=100000, progress_bar=True, callback=eval_callback)

    norm_save_path_train = "./vecnormalize_train.pkl"
    train_env.save(norm_save_path_train)
    norm_save_path_val = "./vecnormalize_val.pkl"
    val_env.save(norm_save_path_val)
    model1.save("final_model1")

    train_env = create_train_env(env_id, "crawl", seed, n_envs)
    if os.path.exists(norm_save_path_train):
        train_env = VecNormalize.load(norm_save_path_train, train_env)
        train_env.training = True

    val_env = create_val_env(env_id, "crawl")
    if os.path.exists(norm_save_path_val):
        val_env = VecNormalize.load(norm_save_path_val, val_env)
        val_env.training = False
    eval_callback = create_eval_callback(val_env, algo, hidden_size, learning_rate, seed, "crawl", eval_freq, n_envs)

    pretrained_model = PPO.load("final_model1", env=train_env)
    pretrained_model.policy.mlp_extractor.ngn_trunk.expand(1, 8, 0.2)
    pretrained_model.policy.mlp_extractor.ngn_trunk.expand(2, 8, 0.2)
    model2 = PPO(
        NGNActorCriticPolicy,
        train_env,
        batch_size=256,
        ent_coef=0.01,
        learning_rate=learning_rate,
        n_steps=int(2048 / n_envs),
        policy_kwargs={
            "n_units": [16, 16, 16],
            "optimizer_kwargs": {"weight_decay": 1e-4},
        }
    )
    model2.policy.load_state_dict(pretrained_model.policy.state_dict())
    model2.learn(total_timesteps=100000, progress_bar=True, callback=eval_callback)

    train_env.save(norm_save_path_train)
    val_env.save(norm_save_path_val)
    model2.save("final_model2")

    train_env = create_train_env(env_id, "knees", seed, n_envs)
    if os.path.exists(norm_save_path_train):
        train_env = VecNormalize.load(norm_save_path_train, train_env)
        train_env.training = True

    val_env = create_val_env(env_id, "knees")
    if os.path.exists(norm_save_path_val):
        val_env = VecNormalize.load(norm_save_path_val, val_env)
        val_env.training = False
    eval_callback = create_eval_callback(val_env, algo, hidden_size, learning_rate, seed, "knees", eval_freq, n_envs)

    pretrained_model = PPO.load("final_model2", env=train_env)
    pretrained_model.policy.mlp_extractor.ngn_trunk.expand(1, 16, 0.2)
    pretrained_model.policy.mlp_extractor.ngn_trunk.expand(2, 16, 0.2)
    model3 = PPO(
        NGNActorCriticPolicy,
        train_env,
        batch_size=256,
        learning_rate=learning_rate,
        ent_coef=0.01,
        n_steps=int(2048 / n_envs),
        policy_kwargs={
            "n_units": [32, 32, 16],
            "optimizer_kwargs": {"weight_decay": 1e-4},
        }
    )
    model3.policy.load_state_dict(pretrained_model.policy.state_dict())
    model3.learn(total_timesteps=500000, progress_bar=True, callback=eval_callback)

    train_env.save(norm_save_path_train)
    model3.save("final_model_knees")
    record_videos(model3, norm_save_path_train, hidden_size, seed)

if __name__ == "__main__":
    typer.run(main)

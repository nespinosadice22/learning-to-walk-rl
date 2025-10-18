import os
import typer
import numpy as np
import optuna

import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from model import NGNActorCriticPolicy

# Register the custom environment.
register(
    id="HumanoidGetUp-v0",
    entry_point="environments.sitting:HumanoidGetUp",
    max_episode_steps=2000,
)

# ---------------------------
# Utility functions
# ---------------------------
def make_env(env_id: str, rank: int, seed: int = 0, starting_pos: str = "standing"):
    def _init():
        if starting_pos == "knees":
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                xml_file="./environments/assets/humanoid_w_box.xml",
                healthy_z_range=(0.75, 2.0)
            )
        elif starting_pos == "standing":
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                xml_file="./environments/assets/humanoid_w_2_boxes.xml",
                healthy_z_range=(0.75, 2.0)
            )
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
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos=starting_pos,
            xml_file="./environments/assets/humanoid_w_box.xml",
            healthy_z_range=(0.75, 2.0)
        ))])
    elif starting_pos == "standing":
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos=starting_pos,
            xml_file="./environments/assets/humanoid_w_2_boxes.xml",
            healthy_z_range=(0.75, 2.0)
        ))])
    elif starting_pos == "prone":
        env = DummyVecEnv([lambda: Monitor(gym.make(env_id, starting_pos=starting_pos, terminate_when_unhealthy=False))])
    else:
        env = DummyVecEnv([lambda: Monitor(gym.make(env_id, starting_pos=starting_pos))])
    return VecNormalize(env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

def create_eval_callback(val_env, algo: str, hidden_size: int, learning_rate: float, seed: int, pos: str, eval_freq: int, n_envs: int, base_dir: str = "./logs"):
    best_model_save_path = os.path.join(base_dir, f"{algo.lower()}_{hidden_size}_{learning_rate:.0e}_{seed}_{pos}", "best_model")
    log_path = os.path.join(base_dir, f"{algo.lower()}_{hidden_size}_{learning_rate:.0e}_{seed}_{pos}")
    return EvalCallback(
        val_env,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        verbose=False,
    )

def evaluate_model(model, env, n_eval_episodes=10):
    rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if isinstance(done, (list, np.ndarray)):
                done = done[0]
            if isinstance(reward, (list, np.ndarray)):
                total_reward += reward[0]
            else:
                total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

# ---------------------------
# Optuna objective function
# ---------------------------
def objective(trial: optuna.Trial) -> float:
    # Create a trial-specific directory to avoid collisions
    trial_dir = os.path.join("./optuna_trials", f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # File paths for VecNormalize statistics and model checkpoints
    norm_save_path_train = os.path.join(trial_dir, "vecnormalize_train.pkl")
    norm_save_path_val = os.path.join(trial_dir, "vecnormalize_val.pkl")
    model1_path = os.path.join(trial_dir, "final_model1")
    model2_path = os.path.join(trial_dir, "final_model2")
    model3_path = os.path.join(trial_dir, "final_model_knees")
    
    # ---------------------------
    # Hyperparameter search space
    # ---------------------------
    # Base learning rate and lr decrease factor for scheduling.
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    lr_decrease = trial.suggest_float("lr_decrease", 0.0, 1.0, step=0.05)
    
    n_steps = trial.suggest_int("n_steps", 1024, 4096, step=512)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 1, 20)
    gamma = trial.suggest_float("gamma", 0.90, 0.999, step=0.001)
    gae_lambda = trial.suggest_float("gae_lambda", 0.80, 1.0, step=0.01)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3, step=0.01)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02, step=0.001)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0, step=0.1)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0, step=0.1)
    use_sde = trial.suggest_categorical("use_sde", [True, False])
    
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    architecture_scale = trial.suggest_int("architecture_scale", 1, 4)
    eval_freq = 10000
    
    timesteps_phase1 = trial.suggest_int("timesteps_phase1", 10000, 100000, step=10000)
    timesteps_phase2 = trial.suggest_int("timesteps_phase2", 10000, 100000, step=10000)
    timesteps_phase3 = trial.suggest_int("timesteps_phase3", 100000, 1000000, step=50000)
    
    n_envs = 8
    seed = trial.suggest_int("seed", 0, 1000)
    
    # Define network architectures based on the suggested architecture_scale
    arch1 = [4 * architecture_scale, 4 * architecture_scale, 16]
    arch2 = [8 * architecture_scale, 8 * architecture_scale, 16]
    arch3 = [16 * architecture_scale, 16 * architecture_scale, 16]
    
    env_id = "HumanoidGetUp-v0"
    
    # ---------------------------
    # Phase 1: "prone"
    # ---------------------------
    train_env_phase1 = create_train_env(env_id, "prone", seed, n_envs)
    val_env_phase1 = create_val_env(env_id, "prone")
    
    model1 = PPO(
        NGNActorCriticPolicy,
        train_env_phase1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde,
        policy_kwargs={
            "optimizer_kwargs": {"weight_decay": weight_decay},
            "n_units": arch1,
        },
        verbose=0,
    )
    model1.learn(total_timesteps=timesteps_phase1, progress_bar=True)
    
    # Save normalization statistics and the model
    train_env_phase1.save(norm_save_path_train)
    val_env_phase1.save(norm_save_path_val)
    model1.save(model1_path)
    
    # Evaluate after phase 1 and report intermediate result to Optuna
    mean_reward_phase1 = evaluate_model(model1, val_env_phase1, n_eval_episodes=10)
    trial.report(mean_reward_phase1, step=1)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # ---------------------------
    # Phase 2: "crawl"
    # ---------------------------
    train_env_phase2 = create_train_env(env_id, "crawl", seed, n_envs)
    if os.path.exists(norm_save_path_train):
        train_env_phase2 = VecNormalize.load(norm_save_path_train, train_env_phase2)
        train_env_phase2.training = True
    val_env_phase2 = create_val_env(env_id, "crawl")
    if os.path.exists(norm_save_path_val):
        val_env_phase2 = VecNormalize.load(norm_save_path_val, val_env_phase2)
        val_env_phase2.training = False
    
    pretrained_model = PPO.load(model1_path, env=train_env_phase2)
    pretrained_model.policy.mlp_extractor.ngn_trunk.expand(1, 4 * architecture_scale, lr_decrease)
    pretrained_model.policy.mlp_extractor.ngn_trunk.expand(2, 4 * architecture_scale, lr_decrease)
    
    model2 = PPO(
        NGNActorCriticPolicy,
        train_env_phase2,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde,
        policy_kwargs={
            "n_units": arch2,
            "optimizer_kwargs": {"weight_decay": weight_decay},
        },
        verbose=0,
    )
    model2.policy.load_state_dict(pretrained_model.policy.state_dict())
    model2.learn(total_timesteps=timesteps_phase2, progress_bar=True)
    
    train_env_phase2.save(norm_save_path_train)
    val_env_phase2.save(norm_save_path_val)
    model2.save(model2_path)
    
    mean_reward_phase2 = evaluate_model(model2, val_env_phase2, n_eval_episodes=10)
    trial.report(mean_reward_phase2, step=2)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # ---------------------------
    # Phase 3: "knees"
    # ---------------------------
    train_env_phase3 = create_train_env(env_id, "knees", seed, n_envs)
    if os.path.exists(norm_save_path_train):
        train_env_phase3 = VecNormalize.load(norm_save_path_train, train_env_phase3)
        train_env_phase3.training = True
    val_env_phase3 = create_val_env(env_id, "knees")
    if os.path.exists(norm_save_path_val):
        val_env_phase3 = VecNormalize.load(norm_save_path_val, val_env_phase3)
        val_env_phase3.training = False
    
    eval_callback = create_eval_callback(
        val_env_phase3,
        "ppo",
        arch1[0],
        learning_rate,
        seed,
        "knees",
        eval_freq,
        n_envs,
        base_dir=os.path.join(trial_dir, "logs")
    )
    
    pretrained_model = PPO.load(model2_path, env=train_env_phase3)
    pretrained_model.policy.mlp_extractor.ngn_trunk.expand(1, 8 * architecture_scale, lr_decrease)
    pretrained_model.policy.mlp_extractor.ngn_trunk.expand(2, 8 * architecture_scale, lr_decrease)
    
    model3 = PPO(
        NGNActorCriticPolicy,
        train_env_phase3,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde,
        policy_kwargs={
            "n_units": arch3,
            "optimizer_kwargs": {"weight_decay": weight_decay},
        },
        verbose=0,
    )
    model3.policy.load_state_dict(pretrained_model.policy.state_dict())
    model3.learn(total_timesteps=timesteps_phase3, progress_bar=True, callback=eval_callback)
    
    train_env_phase3.save(norm_save_path_train)
    model3.save(model3_path)
    
    # Final evaluation on the "knees" validation environment
    final_mean_reward = evaluate_model(model3, val_env_phase3, n_eval_episodes=10)
    return final_mean_reward

# ---------------------------
# Main function to run the Optuna study
# ---------------------------
def main(n_trials: int = 10):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    print("Best trial:")
    print("  Value: ", best_trial.value)
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    typer.run(main)

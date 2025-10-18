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

def make_env(env_id: str, rank: int, starting_pos: str, boxes: bool, seed: int = 0):
    def _init():
        if starting_pos == "knees" or starting_pos == "knees_left": 
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                healthy_z_range=(0.6, 2.0) #changed to be a bit lower 
            )
        elif (starting_pos == "standing" and (not boxes)): 
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                healthy_z_range=(0.75, 2.0)
            )
        elif (starting_pos == "standing_left" and (not boxes)): 
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                healthy_z_range=(0.75, 2.0)
            )
        elif starting_pos == "crawl" or starting_pos == "crawl_perpendicular" or starting_pos == "crawl_more_perpendicular" or starting_pos == "seated": 
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                healthy_z_range=(0.25, 2.0) #not sure 
            )
        elif starting_pos == "prone" or starting_pos == "prone_2" or starting_pos == "supine": 
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                terminate_when_unhealthy=False
            )
        elif starting_pos == "standing" and boxes: 
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                xml_file="./environments/assets/humanoid_w_2_boxes.xml", 
                healthy_z_range=(0.75, 2.0)
            )
        elif starting_pos == "standing_left" and boxes: 
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                xml_file="./environments/assets/humanoid_w_2_boxes.xml", 
                healthy_z_range=(0.75, 2.0)
            )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def create_train_env(env_id: str,  starting_pos: str, boxes: bool, seed: int, n_envs: int):
    env_fns = [make_env(env_id, i, starting_pos, boxes, seed) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env) #added
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def create_val_env(env_id: str, starting_pos: str, boxes: bool):
    if starting_pos == "knees" or starting_pos == "knees_left": 
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos=starting_pos,
            healthy_z_range=(0.6, 2.0) #changed to be a bit lower 
        ))])
    elif (starting_pos == "standing" and (not boxes)): 
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos=starting_pos,
            healthy_z_range=(0.75, 2.0)
        ))])
    elif (starting_pos == "standing_left" and (not boxes)): 
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos=starting_pos,
            healthy_z_range=(0.75, 2.0)
        ))])
    elif starting_pos == "crawl" or starting_pos == "crawl_perpendicular" or starting_pos == "crawl_more_perpendicular"  or starting_pos == "seated": 
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos=starting_pos,
            healthy_z_range=(0.25, 2.0) #not sure 
        ))])
    elif starting_pos == "prone" or starting_pos == "prone_2" or starting_pos == "supine": 
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos=starting_pos,
            terminate_when_unhealthy=False
        ))])
    elif starting_pos == "standing" and boxes: 
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos= starting_pos,
            xml_file="./environments/assets/humanoid_w_2_boxes.xml", 
            healthy_z_range=(0.75, 2.0)
        ))])
    elif starting_pos == "standing_left" and boxes: 
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos= starting_pos,
            xml_file="./environments/assets/humanoid_w_2_boxes.xml", 
            healthy_z_range=(0.75, 2.0)
        ))])
    env = VecMonitor(env)
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
####----------------------------------------------#######

def objective(trial: optuna.Trial, run_config: dict) -> float:
    timesteps = trial.suggest_int("timesteps", run_config["timesteps_min"], run_config["timesteps_max"], step=run_config["step"]) 
    FIRST_CHUNK = run_config["FIRST_CHUNK"]  
    starting_pos = run_config["starting_pos"] 
    boxes = run_config["boxes"] 
    group_name = run_config["group_name"] 
    trial_dir = os.path.join(run_config["base_dir"], f"trial_{trial.number}") 
    #-----------------------Set up Paths------------------#
    os.makedirs(trial_dir, exist_ok=True)
    norm_save_path_train = os.path.join(trial_dir, "vecnormalize_train.pkl")
    norm_save_path_val = os.path.join(trial_dir, "vecnormalize_val.pkl")
    model_path = os.path.join(trial_dir, "final_model")
    #------------------STANDARD PPO PARAMS------------------------------
    #full searches 
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_int("n_steps", 1024, 4096, step=512)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    use_sde = trial.suggest_categorical("use_sde", [True, False])

    #suggestions implemented from 4/4 
    gamma = trial.suggest_categorical("gamma", [0.95, 0.99])
    clip_range = 0.2
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10])
    gae_lambda = 0.95 
    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.005])
    vf_coef = 0.5 
    max_grad_norm = 0.5 
    normalize_advantage = True 
    use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])
    if use_weight_decay:
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    else:
        weight_decay = 0.0
    
    #architecture 
    architecture_scale = trial.suggest_int("architecture_scale", 1, 4)
    arch = [16*architecture_scale, 16*architecture_scale, 16]
    hidden_size = 16*architecture_scale

    #timesteps defined above 

    #other 
    eval_freq = 10000
    n_envs = 8
    seed = trial.suggest_int("seed", 0, 1000) 
    env_id = "HumanoidGetUp-v0"
    
    #----------------------------------------------------------------------
    #add to wandb 
    wandb_run = wandb.init(
        entity="ne3496-princeton-university", 
        project="L2W-4-4-25",
        sync_tensorboard=True, 
        name=f"trial-{trial.number}",
        config={
            "trial_number": trial.number,
            "starting_pos": starting_pos, 
            "timesteps": timesteps,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "normalize_advantage": True, #added
            "use_sde": use_sde,
            "weight_decay": weight_decay,
            "hidden_size": hidden_size,
            "architecture_scale": architecture_scale,
            "seed": seed,
        },
        mode="offline", 
        dir=trial_dir, 
        group= group_name
    )

    # set up environments 
    train_env = create_train_env(env_id, starting_pos, boxes, seed, n_envs)
    val_env = create_val_env(env_id, starting_pos, boxes)
  
    #create callback 
    callback = create_eval_callback(
        val_env,
        "ppo",
        hidden_size, 
        learning_rate,
        seed,
        starting_pos,
        eval_freq,
        n_envs,
        base_dir=os.path.join(trial_dir, "logs")
    )
    #wandb
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(trial_dir, "wandb_model"),
        verbose=1, #changed
        log="all", 
    )
    
    model = PPO(
        NGNActorCriticPolicy,
        train_env,
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
        normalize_advantage=True,
        policy_kwargs={
            "optimizer_kwargs": {"weight_decay": weight_decay},
            "n_units": arch
        },
        verbose=0, 
        tensorboard_log=os.path.join(trial_dir, "tb_logs"), #added
        device="cuda"
    )

    #------------------PRUNING LOGIC-------------------------#
    #first_chunk defined above 
    model.learn(
        total_timesteps=FIRST_CHUNK,
        progress_bar=False,
        callback=[callback, wandb_callback],
        log_interval=1000,
        reset_num_timesteps=False
    )

    mean_reward_1m = evaluate_model(model, val_env, n_eval_episodes=10)
    trial.report(mean_reward_1m, step=1)

    if trial.should_prune():
        wandb_run.finish()
        raise optuna.TrialPruned()

    remaining_steps = timesteps - FIRST_CHUNK
    if remaining_steps > 0:
        model.learn(
            total_timesteps=remaining_steps,
            progress_bar=False,
            callback=[callback, wandb_callback],
            log_interval=1000,
            reset_num_timesteps=False
        )
    #----------------END PRUNING LOGIC -------------------#

    #model.learn(total_timesteps=timesteps, progress_bar = True, callback=[callback, wandb_callback], log_interval=100) 

    train_env.save(norm_save_path_train)
    model.save(model_path)

    #evaluate 
    final_mean_reward = evaluate_model(model, val_env, n_eval_episodes=10)

    #store in wandb 
    wandb.log({"final_eval_reward": final_mean_reward})
    wandb_run.finish()  

    print(f"[Trial {trial.number}] Final Reward: {final_mean_reward:.2f}")
    return final_mean_reward 

def record_videos(model_path, norm_save_path, starting_pos, boxes, video_folder, hidden_size, seed: int):

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

        name_prefix = f"baseline_{hidden_size}_{ep}_{int(time.time())}"
        video_env = VecVideoRecorder(
            test_env,
            video_folder=video_folder,
            record_video_trigger=lambda episode_id: True,
            video_length=10000,
            name_prefix=name_prefix
        )
        model = PPO.load(model_path, env=test_env)
        obs = video_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = video_env.step(action)
            if done: 
                print(info[0]["com_dist_from_feet_cumulative"])
                print(info[0]["energy_cumulative"])
                break
        video_env.close()

# main 
def main(): 
    #EDIT ME to change runs 
    run_config = {
        "timesteps_min": 1000000, 
        "timesteps_max": 1500000, 
        "step": 500000, 
        "FIRST_CHUNK": 1000000, 
        "starting_pos": "standing_left",
        "boxes": True,  
        "base_dir": "./baselines/standing_boxes_left_50", 
        "group_name": "baseline-standing_boxes_left-50", 
        "n_trials": 50, 
    }

    #pruner added 
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
            interval_steps=1
        )
    )
    study.optimize(lambda trial: objective(trial, run_config), n_trials=run_config["n_trials"] )

    best_trial = study.best_trial
    best_trial_num = best_trial.number
    hidden_size = best_trial.params["architecture_scale"]*16
    seed = best_trial.params["seed"]

    #print and save best trial info 
    base_dir = run_config["base_dir"] 
    output_lines = []
    output_lines.append(f"Best trial number: {best_trial_num}")
    output_lines.append(f"Final reward: {best_trial.value}")
    output_lines.append("Params:")
    for key, value in best_trial.params.items():
        output_lines.append(f"  {key}: {value}")
    print("\n".join(output_lines))
    save_path = f"./{base_dir}/best_trial_summary.txt"
    with open(save_path, "w") as f:
        f.write("\n".join(output_lines))

    #record 10 episodes of best trial 
    model_path = f"./{base_dir}/trial_{best_trial_num}/final_model"
    norm_save_path = f"./{base_dir}/trial_{best_trial_num}/vecnormalize_train.pkl"
    video_save_path = f"./{base_dir}/trial_{best_trial_num}/videos"
    os.makedirs(video_save_path, exist_ok=True)
    #record_videos(model_path, norm_save_path, "crawl", False, video_save_path, hidden_size, seed=seed)

if __name__ == "__main__":
    typer.run(main)


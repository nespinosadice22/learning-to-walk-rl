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

# Register the custom environment.
register(
    id="HumanoidGetUp-v0",
    entry_point="environments.sitting:HumanoidGetUp",
    max_episode_steps=2000,
)

def make_env(env_id: str, rank: int, starting_pos: str, boxes: bool, seed: int = 0):
    def _init():
        if starting_pos == "knees": 
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
        elif starting_pos == "crawl" or starting_pos == "crawl_perpendicular" or starting_pos == "crawl_more_perpendicular": 
            env = gym.make(
                env_id,
                starting_pos=starting_pos,
                healthy_z_range=(0.25, 2.0) #not sure 
            )
        elif starting_pos == "prone": 
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
    if starting_pos == "knees": 
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
    elif starting_pos == "crawl" or starting_pos == "crawl_perpendicular" or starting_pos == "crawl_more_perpendicular": 
        env = DummyVecEnv([lambda: Monitor(gym.make(
            env_id,
            starting_pos=starting_pos,
            healthy_z_range=(0.25, 2.0) #not sure 
        ))])
    elif starting_pos == "prone": 
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

#------------------------------------
def objective(trial: optuna.Trial, run_config: dict) -> float:
    #-------------------------EDIT ME TO CHANGE BASELINE--------------#
    group_name = run_config["group_name"] 
    trial_dir = os.path.join(run_config["base_dir"], f"trial_{trial.number}") 
    #-----------------------Set up Paths-------------------------------#
    os.makedirs(trial_dir, exist_ok=True)
    norm_save_path_train = os.path.join(trial_dir, "vecnormalize_train.pkl")
    norm_save_path_val = os.path.join(trial_dir, "vecnormalize_val.pkl")
    model_path = os.path.join(trial_dir, "final_model")

    #-------------  --BASELINE PARAMS----------------------------------#
    learning_rate = run_config["learning_rate"]
    n_steps = run_config["n_steps"]
    batch_size = run_config["batch_size"]
    n_epochs = run_config["n_epochs"]
    gamma = run_config["gamma"]
    gae_lambda = run_config["gae_lambda"]
    clip_range = run_config["clip_range"]
    ent_coef = run_config["ent_coef"]
    vf_coef = run_config["vf_coef"]
    max_grad_norm = run_config["max_grad_norm"]
    normalize_advantage = True 
    use_sde = run_config["use_sde"]
    weight_decay = run_config["weight_decay"]
    architecture_scale =  run_config["architecture_scale"]
    seed = run_config["seed"]

    #---------------------GENERAL PARAMS----------------------------#
    hidden_size = 16*architecture_scale 
    arch = [hidden_size, hidden_size, 16]

    eval_freq = 10000
    n_envs = 8
    env_id = "HumanoidGetUp-v0"
    #---------------------CURRICULUM PARAMS -------------------------#
    #possible curriculums (need to add to this)
    curriculum_options = {
        "P_S": ["prone", "standing"], 
        "C_S": ["crawl", "standing"], 
        "K_S": ["knees", "standing"], 
        "SB_S": ["standing_boxes", "standing"], 
        "P_C_S": ["prone", "crawl", "standing"], 
        "P_K_S": ["prone", "knees", "standing"], 
        "P_SB_S": ["prone", "standing_boxes", "standing"], 
        "C_K_S": ["crawl", "knees", "standing"], 
        "C_SB_S": ["crawl", "standing_boxes", "standing"], 
        "K_SB_S": ["knees", "standing_boxes", "standing"], 
        "P_C_K_S": ["prone", "crawl", "knees", "standing"],
        "P_C_SB_S": ["prone", "crawl", "standing_boxes", "standing"],
        "P_K_SB_S": ["prone", "knees", "standing_boxes", "standing"],
        "C_K_SB_S": ["crawl", "knees", "standing_boxes", "standing"],
        "P_C_K_SB_S": ["prone", "crawl", "knees", "standing_boxes", "standing"]
    }
    
    curriculum_key = trial.suggest_categorical("curriculum_key", list(curriculum_options.keys()))
    curriculum = curriculum_options[curriculum_key]

    reward_goals = run_config["reward_goals"]

    #-----------add to wandb------------#
    wandb_run = wandb.init(
        entity="ne3496-princeton-university", 
        project="L2W-4-4-25",
        sync_tensorboard=True, 
        name=f"trial-{trial.number}",
        config={
            "trial_number": trial.number,
            "curriculum_key": curriculum_key, 
            "reward_goals": reward_goals, 
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
            "normalize_advantage": True, 
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
    #----------------------------------------------
    model = None
    chunk_timesteps = run_config["chunk_timesteps"]
    total_timesteps = 0 
    print(f"Selected curriculum: {curriculum_key}")
    #---------CURRICULUM LOOP--------------------#
    for i, task in enumerate(curriculum): 
        print(f"Phase {i}: {task}")
        if task == "standing_boxes": 
            boxes = True 
            starting_pos = "standing"
        else: 
            boxes = False 
            starting_pos = task 

        #---------set up environments----------
        train_env = create_train_env(env_id, starting_pos, boxes, seed, n_envs)
        if os.path.exists(norm_save_path_train):
            train_env = VecNormalize.load(norm_save_path_train, train_env)
            train_env.training = True
        
        val_env_curr  = create_val_env(env_id, starting_pos, boxes)
        if os.path.exists(norm_save_path_val):
            val_env_loaded = VecNormalize.load(norm_save_path_val, val_env_curr)
            val_env_loaded.training = False
            val_env_curr = val_env_loaded
        
        val_env_prone = create_val_env(env_id, "prone", False)
        if os.path.exists(norm_save_path_val):
            val_env_loaded = VecNormalize.load(norm_save_path_val, val_env_prone) 
            val_env_loaded.training = False
            val_env_prone = val_env_loaded
        val_env_standing = create_val_env(env_id, "standing", False)
        if os.path.exists(norm_save_path_val):
            val_env_loaded = VecNormalize.load(norm_save_path_val, val_env_standing)
            val_env_loaded.training = False
            val_env_standing = val_env_loaded
        
        #-----------set up model-------------  
        if model is None: 
            wandb_callback = WandbCallback(
                gradient_save_freq=0,
                model_save_path=os.path.join(trial_dir, "wandb_model"),
                verbose=1,
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
                normalize_advantage = True, 
                policy_kwargs={
                    "optimizer_kwargs": {"weight_decay": weight_decay}, 
                    "n_units": arch
                },
                verbose=0,
                tensorboard_log=os.path.join(trial_dir, "tb_logs"), 
                device="cuda"
            )
        #otherwise, use same model but update train env (?) 
        else: 
            model.set_env(train_env)
        
        #----------------learning - task completion----------------
        
        #intermediate tasks 
        if task != "standing": 
            mean_reward_current = 0 
            phase_iteration = 0 
            while mean_reward_current < reward_goals[task]: 
                model.learn(
                    total_timesteps = chunk_timesteps, 
                    progress_bar = True, 
                    log_interval = 1000, 
                    reset_num_timesteps = False 
                )
                

                mean_reward_current = evaluate_model(model, val_env_curr)
                #wandb.log({f"phase_{i}_{task}/iteration_{phase_iteration}_mean_rew_curr": mean_reward_current})

                train_env.save(norm_save_path_train)
                val_env_curr.save(norm_save_path_val)

                phase_iteration += 1

                if phase_iteration > 30: 
                    break 
            
            #once learns, evaluate and log not only the current task but also comparatively to others
            num_timesteps_this_phase = chunk_timesteps*(phase_iteration)
            wandb.log({f"phase_{i}_{task}/timesteps": num_timesteps_this_phase})
            total_timesteps += num_timesteps_this_phase

            mean_reward_current = evaluate_model(model, val_env_curr)
            wandb.log({f"phase_{i}_{task}/final_mean_rew_curr": mean_reward_current})

            mean_reward_standing = evaluate_model(model, val_env_standing)
            wandb.log({f"phase_{i}_{task}/final_mean_rew_standing": mean_reward_standing})

            mean_reward_prone= evaluate_model(model, val_env_prone)
            wandb.log({f"phase_{i}_{task}/final_mean_rew_prone": mean_reward_prone})

            trial.report(mean_reward_standing, step=i)

        #on last task (standing), include a callback 
        else: 
            callback = create_eval_callback(
                val_env_standing,
                "ppo",
                hidden_size, 
                learning_rate,
                seed,
                task,
                eval_freq,
                n_envs,
                base_dir=os.path.join(trial_dir, "logs")
            )
            mean_reward_standing = 0 
            phase_iteration = 0 
            while mean_reward_standing < reward_goals["standing"]:
                model.learn(
                    total_timesteps=chunk_timesteps,
                    progress_bar=True,
                    callback=[callback, wandb_callback],
                    log_interval=1000,
                    reset_num_timesteps=False
                )
                model.save(model_path)
                train_env.save(norm_save_path_train)
                val_env_standing.save(norm_save_path_val)

                mean_reward_standing = evaluate_model(model, val_env_standing)
                #wandb.log({f"phase_{i}_{task}/iteration_{phase_iteration}_mean_rew_curr": mean_reward_standing})

                phase_iteration += 1

                if phase_iteration > 50: 
                    break 
            
            #log timesteps 
            num_timesteps_this_phase = chunk_timesteps*(phase_iteration)
            wandb.log({f"phase_{i}_{task}/timesteps": num_timesteps_this_phase})
            total_timesteps += num_timesteps_this_phase
            wandb.log({f"total_timesteps": total_timesteps})
            
            #log final mean reward standing
            wandb.log({"final_mean_reward_standing": mean_reward_standing})
            
            #log final mean reward from prone task 
            final_mean_reward_prone = evaluate_model(model, val_env_prone)
            wandb.log({"final_mean_reward_prone": final_mean_reward_prone})

            wandb_run.finish()
            print(f"[Trial {trial.number}] Final Reward (standing): {mean_reward_standing:.2f}")
            return mean_reward_standing

# main 
def main():
    study = optuna.create_study(direction="maximize")
    standing_run_config = { 
        "n_trials": 30, 
        "base_dir": "curriculum/best_standing_configs",
        "group_name": "my_curriculum_group",
        "reward_goals": { 
            "prone": 300, 
            "crawl": 1100, 
            "knees": 1500, 
            "standing_boxes": 2000, 
            "standing": 250, 
        }, 
        "chunk_timesteps": 100000, 
        "learning_rate": 3.3965316256683674e-05,
        "n_steps": 2560,
        "batch_size": 32,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "weight_decay": 0.0,
        "architecture_scale": 2,
        "seed": 89
    }
    run_config = standing_run_config
    study.optimize(lambda trial: objective(trial, run_config), n_trials=run_config["n_trials"] )

    best_trial = study.best_trial
    best_trial_num = best_trial.number
    curriculum_key = best_trial.params["curriculum_key"]

    #print and save best trial info 
    output_lines = []
    output_lines.append(f"Best trial number: {best_trial_num}")
    output_lines.append(f"Curriculum code: {curriculum_key}")
    output_lines.append(f"Final reward: {best_trial.value}")
    output_lines.append("Params:")
    for key, value in best_trial.params.items():
        output_lines.append(f"  {key}: {value}")
    for key, value in run_config.items():
        output_lines.append(f"  {key}: {value}")
    print("\n".join(output_lines))
    save_path = f"./{run_config['base_dir']}/best_trial_summary.txt"
    with open(save_path, "w") as f:
        f.write("\n".join(output_lines))

   
if __name__ == "__main__":
    typer.run(main)



''' Timestep logic (from before )
    #expanding timesteps 
    phase_timesteps = { } 
    for i, task in enumerate(curriculum): 
        timesteps_phase_i = trial.suggest_int(f"timesteps_phase{i}", 250000*(2**i), 500000*(2**i), step=100000*(2**i))
        phase_timesteps[task] = timesteps_phase_i

    EVEN TIMESTEPS 
    #even across stages -10M approach 
    MAX_TIMESTEPS = 10000000
    MAX_PER_TASK = MAX_TIMESTEPS / len(curriculum) 
    #naive 
    phase_timesteps = {
        task: trial.suggest_int(f"timesteps_{task}", 250000, MAX_PER_TASK, step=500000) for task in curriculum
    }

   NAIVE TIMESTEPS 
    phase_timesteps = {
        task: trial.suggest_int(f"timesteps_{task}", 250000, 2000000, step=500000) for task in curriculum
    }

    10MCAP AND LEFTOVER APPROACH 
    MAX_TIMESTEPS = 10000000
    phase_timesteps = {}
    leftover = MAX_TIMESTEPS

    n_phases = len(curriculum)
    min_each_phase = 250000
    for i, task in enumerate(curriculum):
        # Min required for the rest of the phases:
        remaining_phases = n_phases - i - 1
        min_for_rest = min_each_phase * remaining_phases

        # The max we can afford for this phase so we don't starve future phases:
        max_for_this_phase = leftover - min_for_rest

        if max_for_this_phase < min_each_phase:
            # We can't meet the minimum for this phase,
            # so prune or forcibly set it to min_each_phase
            raise optuna.TrialPruned("Not enough leftover timesteps")

        phase_timesteps[task] = trial.suggest_int(
            f"timesteps_{task}",
            min_each_phase,
            max_for_this_phase,
            step=500000
        )
        leftover -= phase_timesteps[task]
    '''
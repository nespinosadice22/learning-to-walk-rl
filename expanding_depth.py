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

import torch 
from model import NGNActorCriticPolicy


# Register the custom environment.
# Register the custom environment.
register(
    id="HumanoidGetUp-v0",
    entry_point="environments.sitting:HumanoidGetUp",
    max_episode_steps=2000,
)

#copy weights for layers 0... k-1 during depth expand 
def transplant_trunk_weights(src_trunk, dst_trunk):
    with torch.no_grad():
        for i, src_layer in enumerate(src_trunk.layers):
            if i >= len(dst_trunk.layers):
                # dst is deeper so stop copying
                break                  
            dst_layer = dst_trunk.layers[i]
            assert src_layer.weight.shape == dst_layer.weight.shape
            dst_layer.weight.copy_(src_layer.weight)
            if src_layer.bias is not None:
                dst_layer.bias.copy_(src_layer.bias)

def apply_lr_factor(params, factor: float):
    if factor >= 1.0:      
        return
    for p in params:
        if p is None:
            continue
        hooks = getattr(p, "_backward_hooks", None)
        if hooks:
            hooks.clear()

        p.register_hook(lambda grad, f=factor: grad * f)

def print_arch(model, tag=""):
    sizes = [layer.out_features for layer in model.policy.mlp_extractor.ngn_trunk.layers]
    print(f"{tag} NGN architecture: {sizes}")

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

def evaluate_model(model, env, n_eval_episodes):
    rewards = []
    base_seed = 89 
    for s in range (n_eval_episodes): 
        env.seed(base_seed + s)
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
    rewards = np.array(rewards, dtype=np.float32)
    return rewards.mean().item(), rewards.std(ddof=1).item()
#---------------------

def objective(run_config: dict):
    start_time = time.time()
    #-----------------BASELINE PARAMS -----------------------
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
    seed = run_config["seed"]
    
    #----general params-----
    eval_freq = 10000
    n_envs = 8
    env_id = "HumanoidGetUp-v0"

    #------EXPANDING NETWORK SPECIFIC PARAMS - pow2 search with arch scale from best configs-----------#
    num_expansions = 2

    #how much to decrease learning rate by 
    lr_decrease = run_config["lr_decrease"]

    archs = run_config["archs"]
    
    timesteps = [
        1000000,
        2000000,
        4000000,
    ]
    print("Architectures:", archs)
    print("Timesteps:", timesteps)
    #-----------------------Set up Paths------------------#
    trial_dir = run_config["base_dir"]
    starting_pos = run_config["starting_pos"]
    boxes = run_config["boxes"]

    os.makedirs(trial_dir, exist_ok=True)
    norm_save_path_train = os.path.join(trial_dir, "vecnormalize_train.pkl")
    norm_save_path_val = os.path.join(trial_dir, "vecnormalize_val.pkl")
    final_model_path = os.path.join(trial_dir, "final_model")
    #-------------------------------------------------------------------------------------#
    #intermediate model paths 
    model_paths = [os.path.join(trial_dir, f"model_phase_{i}") for i in range(num_expansions)]
    #---------add to wandb-------------
    wandb_run = wandb.init(
        entity="ne3496-princeton-university", 
        project="L2W-4-4-25",
        sync_tensorboard=True, 
        name=f"trial-{starting_pos}-{archs[0][0]}-num_layers_2to4-lr_dec-{lr_decrease}-seed-{seed}",
        config={
            "starting_pos": starting_pos, 
            "boxes": boxes, 
            "num_expansions": num_expansions, 
            "archs": archs, 
            "lr_decrease": lr_decrease, 
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
            "use_sde": use_sde,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        mode="offline", 
        dir=trial_dir, 
        group= run_config["group_name"]
    )
    #--------------------------------------------------------------------------
    #when i = 0, we aren't expanding yet (so we go to num_expansions+1)
    train_env = create_train_env(env_id, starting_pos, boxes, seed, n_envs)
    if os.path.exists(norm_save_path_train):
        train_env_loaded = VecNormalize.load(norm_save_path_train, train_env)
        train_env_loaded.training = True
        train_env = train_env_loaded
    val_env = create_val_env(env_id, starting_pos, boxes)
    if os.path.exists(norm_save_path_train):
        val_env_loaded = VecNormalize.load(norm_save_path_train, val_env)
        val_env_loaded.training = False
        val_env = val_env_loaded

    for i in range (0, num_expansions+1): 
        #-----SET UP-----
        #first model (no expansion)
        if i == 0: 
            wandb_callback = WandbCallback(
                gradient_save_freq=0,
                model_save_path=os.path.join(trial_dir, "wandb_model"),
                verbose=1, #changed
                log="all", 
            )
            model_i = PPO(
                NGNActorCriticPolicy,
                train_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                n_epochs = n_epochs, 
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                normalize_advantage = True, 
                use_sde=use_sde,
                policy_kwargs={
                    "optimizer_kwargs": {"weight_decay": weight_decay},
                    "n_units": archs[i] 
                },
                verbose=0,
                tensorboard_log=os.path.join(trial_dir, "tb_logs"), 
                device="cpu"
            )
        #subsequent models (expanding) 
        else: 
            #load previous
            pretrained_model = PPO.load(model_paths[i-1], env=train_env)
            #expand (doubles)
            #new_layer_size = archs[i-1][0] 
            #pretrained_model.policy.mlp_extractor.ngn_trunk.expand(1, new_layer_size, lr_decrease)
            #pretrained_model.policy.mlp_extractor.ngn_trunk.expand(2, new_layer_size, lr_decrease)
            #initialize 
            model_i = PPO(
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
                normalize_advantage = True, 
                use_sde=use_sde,
                policy_kwargs={
                    "n_units": archs[i],
                    "optimizer_kwargs": {"weight_decay": weight_decay},
                },
                verbose=0,
                tensorboard_log=os.path.join(trial_dir, "tb_logs"), 
                device="cpu"
            )
            transplant_trunk_weights(
                pretrained_model.policy.mlp_extractor.ngn_trunk,
                model_i.policy.mlp_extractor.ngn_trunk,
            )

            dst_pi = model_i.policy.mlp_extractor.policy_net[0]
            src_pi = pretrained_model.policy.mlp_extractor.policy_net[0]
            dst_pi.weight.data.copy_(src_pi.weight.data)
            dst_pi.bias.data.copy_(src_pi.bias.data)

            # first Linear layer of the value (critic) head
            dst_vf = model_i.policy.mlp_extractor.value_net[0]
            src_vf = pretrained_model.policy.mlp_extractor.value_net[0]
            dst_vf.weight.data.copy_(src_vf.weight.data)
            dst_vf.bias.data.copy_(src_vf.bias.data)

            # ---------------- collect "old" parameters for LR scaling (NEW version) ------
            old_depth = len(archs[i - 1])       # number of trunk layers that already existed
            old_layer_params = []

            # trunk layers that came from the previous model
            for j in range(old_depth):
                layer = model_i.policy.mlp_extractor.ngn_trunk.layers[j]
                old_layer_params.extend([layer.weight, layer.bias])

            # first layer of policy & value heads
            old_layer_params.extend([dst_pi.weight, dst_pi.bias, dst_vf.weight, dst_vf.bias])

            apply_lr_factor(old_layer_params, lr_decrease) 

        #-----LEARN------
        #last model 
        if i == (num_expansions): #final phase 
            eval_callback = create_eval_callback(
                val_env,
                "ppo",
                archs[i][0],
                learning_rate,
                seed,
                starting_pos,
                eval_freq,
                n_envs,
                base_dir=os.path.join(trial_dir, "logs")
            )
            #learn (with callback)
            model_i.learn(total_timesteps=timesteps[i], progress_bar=True, callback=[eval_callback, wandb_callback])
            #save (model to final path)
            model_i.save(final_model_path)
            train_env.save(norm_save_path_train)
            val_env = create_val_env(env_id, starting_pos, boxes)
            if os.path.exists(norm_save_path_train):
                val_env_loaded = VecNormalize.load(norm_save_path_train, val_env)
                val_env_loaded.training = False
                val_env = val_env_loaded
            #final evaluate 
            final_mean_reward, std_dev = evaluate_model(model_i, val_env, n_eval_episodes=20)
            print(final_mean_reward, std_dev)
            wandb.log({"final_mean_reward": final_mean_reward, "std_reward": std_dev})
           
     
            duration = time.time() - start_time  # duration in seconds
            wandb.log({"trial_duration_seconds": duration})
            
            print(f"Trial took {duration / 60:.2f} minutes.")
            wandb_run.finish()  
            return final_mean_reward, std_dev
        #intermediate models 
        else: 
            model_i.learn(total_timesteps=timesteps[i], progress_bar=True)
            print_arch(model_i, tag=f"Phase {i}") 
            #save 
            model_i.save(model_paths[i])
            train_env.save(norm_save_path_train)
            val_env = create_val_env(env_id, starting_pos, boxes)
            if os.path.exists(norm_save_path_train):
                val_env_loaded = VecNormalize.load(norm_save_path_train, val_env)
                val_env_loaded.training = False
                val_env = val_env_loaded
            #eval 
            mean_reward_i, std_dev_i = evaluate_model(model_i, val_env, n_eval_episodes=20)
            wandb.log({
                f"mean_reward_phase_{i}": mean_reward_i, 
                f"reward_phase_{i}_std": std_dev_i, 
            })
            print(mean_reward_i, std_dev_i)



# ---------------------------
# Main function to run the Optuna study
# ---------------------------
def main():
    seed = 93 
    lr_decreases = [0.1, 0.3, 0.5, 0.7, 0.9]
    for lr_dec in lr_decreases: 
        standing_run_config = {
            "base_dir": f"standing_expanding/depth/arch64_2-4layers_seed_{seed}_lr_dec_{lr_dec}",
            "group_name": "standing_expanding/depth",
            "starting_pos": "standing", 
            "seed": seed, 
            "lr_decrease": lr_dec, 
            "archs": [
                [64, 64], 
                [64, 64, 64],
                [64, 64, 64, 64],
            ],
            "boxes": False, 
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
        }
    
        run_config = standing_run_config
        final_rew, std_dev = objective(run_config)
        #print and save best trial info 
        output_lines = []
        output_lines.append(f"Final reward: {final_rew}")
        output_lines.append(f"std dev: {std_dev}")
        output_lines.append("CPU")
        output_lines.append("Params:")
        for key, value in run_config.items():
            output_lines.append(f"  {key}: {value}")
        print("\n".join(output_lines))
        save_path = f"./{run_config['base_dir']}/best_trial_summary.txt"
        with open(save_path, "w") as f:
            f.write("\n".join(output_lines))
        


if __name__ == "__main__":
    typer.run(main)



from typing import List, Tuple, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback

class ExpansionCallback(BaseCallback):
    def __init__(self, expand_every: int = 10000, expand_layer: Union[List[int], int] = [1, 2], expand_units: int = 2, verbose: int = 1):
        super().__init__(verbose)
        self.expand_every = expand_every
        self.expand_layer = expand_layer
        self.expand_units = expand_units

    def _on_step(self) -> bool:
        if self.num_timesteps % self.expand_every == 0:
            if isinstance(self.expand_layer, list):
                for layer in self.expand_layer:
                    self.model.policy.mlp_extractor.ngn_trunk.expand(layer, self.expand_units)
                    if self.verbose:
                        print(f"Expanded layer {layer} by {self.expand_units} units at step {self.num_timesteps}")
            else:
                self.model.policy.mlp_extractor.ngn_trunk.expand(self.expand_layer, self.expand_units)
                if self.verbose:
                    print(f"Expanded layer {self.expand_layer} by {self.expand_units} units at step {self.num_timesteps}")
        return True


class NGN(nn.Module):
    def __init__(self, n_obs: int, n_units: List[int], eps: float = 1e-3) -> None:
        super().__init__()
        n_units = [n_obs] + n_units
        self.layers = nn.ModuleList(
            [nn.Linear(n_units[i - 1], n_units[i]) for i in range(1, len(n_units))]
        )
        self.eps = eps

    def expand(self, layer: int, units: int, lr_factor: float = 1.0):
        """Simple expansion rule. Expand the weights at index [layer] and previous layer [layer - 1]
        to accommodate additional units with random inits, while scaling down the gradients
        for the old neurons by lr_factor."""
        assert layer > 0, "Cannot expand the input layer."

        # --- Expand the current layer ---
        old_layer = self.layers[layer]
        in_features = old_layer.in_features
        out_features = old_layer.out_features

        new_layer = nn.Linear(in_features + units, out_features)
        init.orthogonal_(new_layer.weight, gain=0.1)
        
        with torch.no_grad():
            new_layer.weight.data[:, :in_features] = old_layer.weight.data
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data
        # Register a hook that scales the gradient for the "old" portion of the weight:
        def hook_current(grad):
            grad[:, :in_features] *= lr_factor
            return grad
        new_layer.weight.register_hook(hook_current)
        
        self.layers[layer] = new_layer

        # --- Expand the previous layer ---
        old_layer = self.layers[layer - 1]
        in_features = old_layer.in_features
        out_features = old_layer.out_features

        new_layer_prev = nn.Linear(in_features, out_features + units)
        init.orthogonal_(new_layer_prev.weight, gain=0.1)
        with torch.no_grad():
            new_layer_prev.weight.data[:out_features, :] = old_layer.weight.data.clone()
            if old_layer.bias is not None:
                new_layer_prev.bias.data[:out_features] = old_layer.bias.data.clone()
        # Register a hook for the previous layer to scale the gradient for the old neurons:
        def hook_prev(grad):
            grad[:out_features, :] *= lr_factor
            return grad
        new_layer_prev.weight.register_hook(hook_prev)
        
        self.layers[layer - 1] = new_layer_prev


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

class NGNNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 16,
        last_layer_dim_vf: int = 16,
        n_units: List[int] = [16, 16, 16],
        eps: float = 1e-3,
    ):
        super().__init__()
        # Use NGN as the trunk feature extractor:
        self.ngn_trunk = NGN(n_obs=feature_dim, n_units=n_units, eps=eps)
        trunk_out_dim = n_units[-1]
        
        # Define the policy (actor) head:
        self.policy_net = nn.Sequential(
            nn.Linear(trunk_out_dim, last_layer_dim_pi),
            #nn.ReLU()
            #switch this to leaky relu, add tanh at the end 
            nn.LeakyReLU(negative_slope=0.01), #check whether param is right
            nn.Tanh()
        )
        # Define the value (critic) head:
        self.value_net = nn.Sequential(
            nn.Linear(trunk_out_dim, last_layer_dim_vf),
            #nn.ReLu() 
            nn.LeakyReLU(negative_slope=0.01) #check whether param is right
        )
        # Save the latent dimensions; these are used by SB3 to build distributions.
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the trunk representation once:
        x = self.ngn_trunk(features)
        latent_pi = self.policy_net(x)
        latent_vf = self.value_net(x)
        return latent_pi, latent_vf

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self(features)
        return latent_pi

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        _, latent_vf = self(features)
        return latent_vf

class NGNActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = True
        self.n_units = kwargs.pop("n_units", [16, 16, 16])
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = NGNNetwork(self.features_dim, last_layer_dim_pi=16, last_layer_dim_vf=16, n_units=self.n_units)

if __name__ == "__main__":
    expansion_callback = ExpansionCallback(expand_every=20000, expand_layer=[1, 2], expand_units=2, verbose=1)
    model = PPO(NGNActorCriticPolicy, "CartPole-v1", verbose=1)
    print(model.policy)
    model.learn(100000, callback=expansion_callback, progress_bar=True)
    print(model.policy)


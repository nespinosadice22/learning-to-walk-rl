from typing import Dict, Tuple, Union

import os
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}
CURRENT_DIR = os.path.dirname(__file__)
DEFAULT_XML_PATH = os.path.join(CURRENT_DIR, "assets", "humanoid.xml")


def mass_center(model, data):
    """Calculate center of mass and return x, y coordinates."""
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidWalking(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = DEFAULT_XML_PATH,  # strictly this xml file for now but we will accept others in the future
        frame_skip: int = 5,  # number of simulator steps to run between actions
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,  # are we making progress towards target
        reset_noise_scale: float = 1e-2,  # perturbate the starting condition
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        radius: float = 100.0,
        threshold: float = 1.0,
        predator: bool = True,
        predator_speed: float = 0.1,
        predator_delay: int = 10,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            reset_noise_scale,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._reset_noise_scale = reset_noise_scale
        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # sampling target from unit circle
        self._radius = radius
        self._threshold = threshold
        self.sample_reference()

        # calculate obs_size based on Mujoco return
        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        obs_size += 2  # for reference position

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "qpos": self.data.qpos.size,  # position of root + all joints
            "qvel": self.data.qvel.size,  # velocity of root + all joints
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6)
            * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
            "reference": 2,
        }

        # predator parameters
        self._predator = predator
        self._predator_speed = predator_speed
        self._predator_delay = predator_delay

        # initialize predator and agent distances for predator chasing logic
        self._reset_predator()

    def _reset_predator(self):
        """Reset predator and agent progress distances."""
        self._agent_distance = 0.0
        if self._predator:
            self._predator_distance = -self._predator_delay

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        reference = self.reference

        com_inertia = (
            self.data.cinert[1:].flatten()
            if self._include_cinert_in_observation
            else np.array([])
        )
        com_velocity = (
            self.data.cvel[1:].flatten()
            if self._include_cvel_in_observation
            else np.array([])
        )
        actuator_forces = (
            self.data.qfrc_actuator[6:].flatten()
            if self._include_qfrc_actuator_in_observation
            else np.array([])
        )
        external_contact_forces = (
            self.data.cfrc_ext[1:].flatten()
            if self._include_cfrc_ext_in_observation
            else np.array([])
        )

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
                reference,
            )
        )

    def _update_predator(self, velocity_toward_reference: float) -> bool:
        self._agent_distance += velocity_toward_reference
        self._predator_distance += self._predator_speed
        return self._predator_distance > self._agent_distance

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        direction_to_reference = self.reference - xy_position_after
        direction_to_reference /= np.linalg.norm(direction_to_reference)
        velocity_toward_reference = np.dot(xy_velocity, direction_to_reference)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(velocity_toward_reference)

        if (
            np.linalg.norm(self.reference - self.data.qpos[0:2], ord=2)
            < self._threshold
        ):
            self.sample_reference()

        if self._predator:
            terminated = self._update_predator(velocity_toward_reference)
        else:
            terminated = False

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_reference": np.linalg.norm(
                self.reference - self.data.qpos[0:2], ord=2
            ),
            "velocity_toward_reference": velocity_toward_reference,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # Note: truncation is handled by the TimeLimit wrapper
        return observation, reward, terminated, False, info

    def _get_rew(self, velocity: float):
        reward = self._forward_reward_weight * velocity

        reward_info = {
            "reward_forward": reward,
        }

        return reward, reward_info

    def sample_reference(self):
        theta = np.random.uniform(0, 2 * np.pi)
        self.reference = np.array(
            [self._radius * np.cos(theta), self._radius * np.sin(theta)]
        )

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self.sample_reference()
        self._reset_predator()

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_reference": np.linalg.norm(
                self.reference - self.data.qpos[0:2], ord=2
            ),
        }

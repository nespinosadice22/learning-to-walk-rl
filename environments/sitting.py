from typing import Dict, Tuple, Union, List

import os
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

from constants import KEYFRAMES

#same as walking.py 
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}
CURRENT_DIR = os.path.dirname(__file__)
DEFAULT_XML_PATH = os.path.join(CURRENT_DIR, "assets", "humanoid.xml")


class HumanoidGetUp(MujocoEnv, utils.EzPickle): 
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self, 
        xml_file: str = DEFAULT_XML_PATH, 
        frame_skip: int = 5, 
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        upright_reward_weight: float = 1,  # getting up
        ctrl_cost_weight: float = 0.1, # low energy
        reset_noise_scale: float = 1e-2, 
        healthy_z_range: Tuple[float, float] = (0.25, 2.0),
        weighted_sampling: bool = True,
        terminate_when_unhealthy: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        starting_pos: Union[List[str], str] = "standing",
        **kwargs,

    ): 
        utils.EzPickle.__init__(self, 
        xml_file,
        frame_skip, 
        default_camera_config,
        upright_reward_weight,
        reset_noise_scale, 
        include_cinert_in_observation,
        include_cvel_in_observation,
        include_qfrc_actuator_in_observation,
        include_cfrc_ext_in_observation,
        **kwargs)
        
        self._upright_reward_weight = upright_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale
        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._weighted_sampling = weighted_sampling and isinstance(starting_pos, list)
        if self._weighted_sampling and isinstance(starting_pos, list):
            self.p = np.zeros((len(starting_pos)))
            self.p[0] = 1.0

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

        # calculate obs_size based on Mujoco return
        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

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

        self.starting_pos = starting_pos
        #added for energy tracking
        self.energy_cumulative = 0.0  
        self.com_cumulative = 0.0

  

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

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
            )
        )
    
    


    def step(self, action): 
        head_id = self.model.geom("head").id
        head_pos = self.data.geom_xpos[head_id, 2]
        
        #----adding energy expenditure tracking-----
        joint_torques = self.data.qfrc_actuator
        joint_velocities = self.data.qvel
        inst_power = joint_torques * joint_velocities
        total_power = np.sum(np.abs(inst_power))
        energy_expenditure = total_power * self.dt 
        self.energy_cumulative += energy_expenditure 
        #-------------------------------------------
        self.do_simulation(action, self.frame_skip)
        reward, reward_info = self._get_rew(head_pos)
        observation = self._get_obs()
        if self._terminate_when_unhealthy and (
            head_pos < self._healthy_z_range[0] or head_pos > self._healthy_z_range[1]
        ):
            terminated = True
        else:
            terminated = False

        #--------------COM metric-------------------#
        total_mass = np.sum(self.model.body_mass)
        com = np.sum(self.data.xpos * self.model.body_mass[:, None], axis=0) / total_mass
        left_foot_id = self.model.geom("left_foot").id 
        left_foot_pos = self.data.geom_xpos[left_foot_id]
        right_foot_id = self.model.geom("right_foot").id
        right_foot_pos = self.data.geom_xpos[right_foot_id]
        foot_center = 0.5 * (left_foot_pos + right_foot_pos)
        com_dist_from_feet = np.linalg.norm(com[:2] - foot_center[:2])
        self.com_cumulative += com_dist_from_feet
        #---------------------------------------------#
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "head_pos": head_pos,
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "energy_expenditure": energy_expenditure, 
            "energy_cumulative": self.energy_cumulative, 
            "com_dist_from_feet": com_dist_from_feet, 
            "com_dist_from_feet_cumulative": self.com_cumulative, 
            **reward_info,
        }

        if self.render_mode == "human": 
            self.render() 

        return observation, reward, terminated, False, info 
    

    def _get_rew(self, head_pos): 
        up_rew = head_pos * self._upright_reward_weight
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        reward = up_rew - ctrl_cost

        reward_info = {
            "reward_head": reward,
        }
        
        return reward, reward_info
    
    def reset_model(self):
        if isinstance(self.starting_pos, str):
            if self.starting_pos == "standing":
                start = self.init_qpos
            else:
                start = KEYFRAMES[self.starting_pos]
        else:
            if self._weighted_sampling:
                start = self.np_random.choice(self.starting_pos, p=self.p)
            else:
                start = self.np_random.choice(self.starting_pos)
            if start == "standing":
                start = self.init_qpos
            else:
                start = KEYFRAMES[start]

        # shift prob mass left to right
        if self._weighted_sampling:
            for i in range(0, self.p.shape[0] - 1):
                if self.p[i] > 0.05:
                    self.p[i] -= 0.01
                    self.p[i + 1] += 0.01
                    self.p = np.abs(self.p)
                    self.p = self.p / np.sum(self.p)
                    break

        #energy added 
        self.energy_cumulative = 0.0 
        self.com_cumulative = 0.0 

        # apply a random rotation to the starting position
        #random_yaw = self.np_random.uniform(-np.pi, np.pi)
        #rotation = R.from_euler('x', random_yaw)
        #random_quat = rotation.as_quat()  # returns quaternion in [x, y, z, w] format
        #start[3:7] = random_quat

        qpos = np.copy(start) 
        

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos += self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        #------------CRAWL NOISE FOR ARMS--------------#
        if self.starting_pos == "crawl":
            left_shoulder_1_angle = self.np_random.uniform(0, np.pi / 2)
            right_shoulder_1_angle = -left_shoulder_1_angle 
            #range for right shoulder_1
            qpos[18] = right_shoulder_1_angle
            #range for left shoulder_1
            qpos[21] = left_shoulder_1_angle
        
        #--------KNEES_LEFT NOISE FOR KNEES----------#
        '''
        knee_configs = [ 
            [1, 0],
            [.9, -.3],
            [.8, -.6],
            [.7, -.9],
            [.6, -1.2]
        ]
        if self.starting_pos == "knees_left": 
            chosen = self.np_random.choice(len(knee_configs))
            root_z = knee_configs[chosen][0]
            angle = knee_configs[chosen][1]
            qpos[2] = root_z 
            qpos[8] = angle 
            qpos[13] = (-np.pi /2 ) + angle 
            qpos[17] = (-np.pi /2) + angle
        '''
        #-------------------------------------------#
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel) 

        observation = self._get_obs() 
        return observation

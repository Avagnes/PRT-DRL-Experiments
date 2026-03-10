import gymnasium
import sys
import os
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from rl_zoo3 import ALGOS
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from environments import SeedBase, ExecuteBase


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    # 1. Parameters for environment
    parser.add_argument("--reward_scale", type=float, default=1, help="reward scale factor")
    parser.add_argument("--action_type", type=str, default="continu", help="Options: continu/discret")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDistri",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[256,256,256])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")
    parser.add_argument("--value_min_log_std", type=int, default=-8)
    parser.add_argument("--value_max_log_std", type=int, default=8)

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256,256,256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_output_activation", type=str, default="linear", help="Options: linear/tanh")
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=0.0001)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0001)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
    # special parameter
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--alpha", type=bool, default=0.2)
    parser.add_argument("--delay_update", type=int, default=2)
    parser.add_argument("--TD_bound", type=float, default=1)
    parser.add_argument("--bound", default=True)
    args = vars(parser.parse_args())
    return args


class TanhGaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])
        self.EPS = 1e-6

    def sample(self):
        action = self.gauss_distribution.sample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            action
        ) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
                self.gauss_distribution.log_prob(action)
                - torch.log(1 + self.EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
                - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            action
        ) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
                self.gauss_distribution.log_prob(action)
                - torch.log(1 + self.EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
                - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def log_prob(self, action_limited) -> torch.Tensor:
        action = torch.atanh(
            (1 - self.EPS)
            * (2 * action_limited - (self.act_high_lim + self.act_low_lim))
            / (self.act_high_lim - self.act_low_lim)
        )
        log_prob = self.gauss_distribution.log_prob(action) - torch.log(
            (self.act_high_lim - self.act_low_lim)
            * (1 + self.EPS - torch.pow(torch.tanh(action), 2))
        ).sum(-1)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) + (
                self.act_high_lim + self.act_low_lim
        ) / 2

    def kl_divergence(self, other) -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


def get_apprfunc_dict(key: str, type=None, **kwargs):
    var = dict()
    var["apprfunc"] = kwargs[key + "_func_type"]
    var["name"] = kwargs[key + "_func_name"]
    var["obs_dim"] = kwargs["obsv_dim"]
    var["min_log_std"] = kwargs.get(key + "_min_log_std", float("-20"))
    var["max_log_std"] = kwargs.get(key + "_max_log_std", float("2"))
    var["std_type"] = kwargs.get(key + "_std_type", "mlp_shared")
    var["norm_matrix"] = kwargs.get("norm_matrix", None)

    apprfunc_type = kwargs[key + "_func_type"]
    if apprfunc_type == "MLP":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
    elif apprfunc_type == "CNN":
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
        var["conv_type"] = kwargs[key + "_conv_type"]
    elif apprfunc_type == "CNN_SHARED":
        if key == "feature":
            var["conv_type"] = kwargs["conv_type"]
        else:
            var["feature_net"] = kwargs["feature_net"]
            var["hidden_activation"] = kwargs[key + "_hidden_activation"]
            var["output_activation"] = kwargs[key + "_output_activation"]
    else:
        raise NotImplementedError

    if kwargs["action_type"] == "continu":
        var["act_high_lim"] = np.array(kwargs["action_high_limit"])
        var["act_low_lim"] = np.array(kwargs["action_low_limit"])
        var["act_dim"] = kwargs["action_dim"]
    else:
        raise NotImplementedError("DSAC don't support discrete action space!")

    var["action_distribution_cls"] = TanhGaussDistribution
    return var


def init_args(env, **args):
    # observation dimension
    if len(env.observation_space.shape) == 1:
        args["obsv_dim"] = env.observation_space.shape[0]
    else:
        args["obsv_dim"] = env.observation_space.shape

    if (
        args["action_type"] == "continu"
    ):  # get the dimension of continuous action or the num of discrete action
        args["action_dim"] = (
            env.action_space.shape[0]
            if len(env.action_space.shape) == 1
            else env.action_space.shape
        )
        args["action_high_limit"] = env.action_space.high.astype('float32')
        args["action_low_limit"] = env.action_space.low.astype('float32')
    else:
        raise NotImplementedError('DSAC do not support discrete action space! ')
    return args


def create_apprfunc(**kwargs):
    import importlib
    apprfunc_name = kwargs["apprfunc"]
    try:
        file = importlib.import_module('mlp')
    except NotImplementedError:
        raise NotImplementedError("This apprfunc does not exist")

    # name = kwargs['name'].upper()

    name = kwargs["name"]
    # print(name)
    # print(kwargs)

    if hasattr(file, name):  #
        apprfunc_cls = getattr(file, name)
        apprfunc = apprfunc_cls(**kwargs)
    else:
        raise NotImplementedError("This apprfunc is not properly defined")

    # print("--Initialize appr func: " + name + "...")
    return apprfunc


class ApproxContainer(torch.nn.Module):
    """Approximate function container for DSAC_V2.

    Contains one policy and one action value.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # create q networks
        q_args = get_apprfunc_dict("value", kwargs["value_func_type"], **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        # create policy network
        policy_args = get_apprfunc_dict("policy", kwargs["policy_func_type"], **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)

        # set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # create optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["value_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class SeedSpace(SeedBase):
    def __init__(self):
        super().__init__(
            high = np.array([0.01] * 47),
            low = -np.array([0.01] * 47),
            shape = (47,),
            dim = 47
        )
        self.qpos_dict = {
            # 根关节自由度 (Root joint - free type)
            0: "root_x_position",          # x轴位置 (米)
            1: "root_y_position",          # y轴位置 (米)  
            2: "root_z_position",          # z轴位置 (米) - 初始1.4米高
            3: "root_qw_orientation",      # 根姿态四元数 w分量
            4: "root_qx_orientation",      # 根姿态四元数 x分量
            5: "root_qy_orientation",      # 根姿态四元数 y分量
            6: "root_qz_orientation",      # 根姿态四元数 z分量
            
            # 躯干关节 (Torso joints)
            7: "abdomen_z_rotation",       # 腹部绕z轴旋转 (rad)
            8: "abdomen_y_rotation",       # 腹部绕y轴旋转 (rad)
            9: "abdomen_x_rotation",       # 腹部绕x轴旋转 (rad)
            
            # 右腿关节 (Right leg joints)
            10: "right_hip_x_rotation",    # 右髋关节绕x轴旋转 (rad)
            11: "right_hip_z_rotation",    # 右髋关节绕z轴旋转 (rad)
            12: "right_hip_y_rotation",    # 右髋关节绕y轴旋转 (rad)
            13: "right_knee_rotation",     # 右膝关节旋转 (rad)
            
            # 左腿关节 (Left leg joints)
            14: "left_hip_x_rotation",     # 左髋关节绕x轴旋转 (rad)
            15: "left_hip_z_rotation",     # 左髋关节绕z轴旋转 (rad)
            16: "left_hip_y_rotation",     # 左髋关节绕y轴旋转 (rad)
            17: "left_knee_rotation",      # 左膝关节旋转 (rad)
            
            # 右臂关节 (Right arm joints)
            18: "right_shoulder1_rotation", # 右肩关节1复合轴旋转 (rad)
            19: "right_shoulder2_rotation", # 右肩关节2复合轴旋转 (rad)
            20: "right_elbow_rotation",     # 右肘关节旋转 (rad)
            
            # 左臂关节 (Left arm joints)
            21: "left_shoulder1_rotation",  # 左肩关节1复合轴旋转 (rad)
            22: "left_shoulder2_rotation",  # 左肩关节2复合轴旋转 (rad)
            23: "left_elbow_rotation",      # 左肘关节旋转 (rad)
        }
        self.qvel_dict = {
            # 根关节线速度 (Root linear velocity)
            0: "root_vx_velocity",         # x轴线速度 (米/秒)
            1: "root_vy_velocity",         # y轴线速度 (米/秒)
            2: "root_vz_velocity",         # z轴线速度 (米/秒)
            
            # 根关节角速度 (Root angular velocity)
            3: "root_wx_angular_velocity",  # x轴角速度 (弧度/秒)
            4: "root_wy_angular_velocity",  # y轴角速度 (弧度/秒)
            5: "root_wz_angular_velocity",  # z轴角速度 (弧度/秒)
            
            # 躯干关节角速度 (Torso joint angular velocities)
            6: "abdomen_z_angular_velocity",  # 腹部z轴角速度
            7: "abdomen_y_angular_velocity",  # 腹部y轴角速度
            8: "abdomen_x_angular_velocity",  # 腹部x轴角速度
            
            # 右腿关节角速度 (Right leg joint angular velocities)
            9: "right_hip_x_angular_velocity",  # 右髋x轴角速度
            10: "right_hip_z_angular_velocity", # 右髋z轴角速度
            11: "right_hip_y_angular_velocity", # 右髋y轴角速度
            12: "right_knee_angular_velocity",  # 右膝角速度
            
            # 左腿关节角速度 (Left leg joint angular velocities)
            13: "left_hip_x_angular_velocity",  # 左髋x轴角速度
            14: "left_hip_z_angular_velocity",  # 左髋z轴角速度
            15: "left_hip_y_angular_velocity",  # 左髋y轴角速度
            16: "left_knee_angular_velocity",   # 左膝角速度
            
            # 右臂关节角速度 (Right arm joint angular velocities)
            17: "right_shoulder1_angular_velocity",  # 右肩1角速度
            18: "right_shoulder2_angular_velocity",  # 右肩2角速度
            19: "right_elbow_angular_velocity",      # 右肘角速度
            
            # 左臂关节角速度 (Left arm joint angular velocities)
            20: "left_shoulder1_angular_velocity",   # 左肩1角速度
            21: "left_shoulder2_angular_velocity",   # 左肩2角速度
            22: "left_elbow_angular_velocity",       # 左肘角速度
        }
        self.definition = deepcopy(self.qpos_dict)
        for i in range(23):
            self.definition[24+i] = self.qvel_dict[i]


class Execute(ExecuteBase):
    def __init__(self, seed=42, render=False, algo:str = None, model_path:str = None) -> None:
        self.env_name = "Humanoid-v4"
        if algo is None:
            self.algo = 'dsac-t'
        else:
            self.algo = algo.lower()
        if render:
            self.env = gymnasium.make(self.env_name, render_mode="human")
        else:
            self.env = gymnasium.make(self.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        
        if self.algo == 'dsac-t':
            self.reward_low_threshold = 0
            self.reward_high_threshold = 12000
            args = get_args()
            args = init_args(self.env, **args)
            self.networks = ApproxContainer(**args)
            self.networks.load_state_dict(torch.load('models/dsac-t/Humanoid-v4.pkl'))
        else:
            device = torch.device('cpu')
            self.reward_low_threshold = 0
            self.reward_high_threshold = 10000

            kwargs = dict(seed=seed)
            off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
            if algo in off_policy_algos:
                # Dummy buffer size as we don't need memory to enjoy the trained agent
                kwargs.update(dict(buffer_size=1))

            # Check if we are running python 3.8+
            # we need to patch saved model under python 3.6/3.7 to load them
            newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

            custom_objects = {}
            if newer_python_version:
                custom_objects = {
                    "learning_rate": 0.0,
                    "lr_schedule": lambda _: 0.0,
                    "clip_range": lambda _: 0.0,
                    # load models with different obs bounds
                    # Note: doesn't work with channel last envs
                    # "observation_space": env.observation_space,
                }
            model_path = f'models/{algo}/{self.env_name}.zip'
            self.model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)
    
    def __call__(self, test_case):
        if self.algo == 'dsac-t':
            return self.execute_dsac(test_case)
        else:
            return self.execute_sb3(test_case)

    def execute_dsac(self, seed):
        observation, info = self.env.reset(seed=42, options={'noise': seed})
        sequence = [observation]
        actions = []
        episode_over = False
        episode_reward = 0
        while not episode_over:
            batch_obs = torch.from_numpy(np.expand_dims(observation, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
            observation, reward, terminated, truncated, info = self.env.step(action)
            sequence.append(observation)
            actions.append(action)
            episode_reward = episode_reward + reward
            episode_over = terminated or truncated
        info['state_sequence'] = sequence
        info['action_sequence'] = actions
        return episode_reward, terminated, info

    def execute_sb3(self, seed):
        episode_start = np.ones((1,), dtype=bool)
        lstm_states = None
        observation, info = self.env.reset(seed=42, options={'noise': seed})
        sequence = [observation]
        actions = []
        episode_over = False
        episode_reward = 0
        while not episode_over:
            action, lstm_states = self.model.predict(
                observation,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            observation, reward, terminated, truncated, info = self.env.step(action)
            sequence.append(observation)
            actions.append(action)
            episode_reward = episode_reward + reward
            episode_over = terminated or truncated
        info['state_sequence'] = sequence
        info['action_sequence'] = actions
        return episode_reward, terminated, info


if __name__ == "__main__":
    seedspace = SeedSpace()
    execute = Execute(render=True)
    seed = seedspace.low
    episode_reward, terminated, info = execute(seed)
    print(episode_reward, terminated)
    seed = seedspace.high
    episode_reward, terminated, info = execute(seed)
    print(episode_reward, terminated)
    mutate_seed = seedspace.mutate(seed)
    episode_reward, terminated, info = execute(mutate_seed)
    print(episode_reward, terminated)
    print("successfully testing!")
import gymnasium
from rl_zoo3 import ALGOS
import sys
import torch
import numpy as np
from environments import SeedBase, ExecuteBase


class SeedSpace(SeedBase):
    def __init__(self):
        super().__init__(
            high = np.array([0.9, 0.3, 3*np.pi/180, 2*np.pi/180]),
            low = -np.array([0.9, 0.3, 3*np.pi/180, 2*np.pi/180]),
            shape = (4,),
            dim = 4
        )


class Execute(ExecuteBase):
    def __init__(self, seed=42, render=False, algo:str = None, model_path: str = None) -> None:
        if render:
            self.env = gymnasium.make("CartPole-v1", render_mode="human")
        else:
            self.env = gymnasium.make("CartPole-v1")
        if algo is None:
            algo = "ppo"
        device = torch.device('cpu')
        self.action = None
        self.obs_dim = self.env.observation_space.shape[0]
        self.reward_low_threshold = 0
        self.reward_high_threshold = 500
        self.observation_space = gymnasium.spaces.Box(np.array([-4.8, -4, -0.41887903, -4]), np.array([4.8, 4, 0.41887903, 4]), (4,), np.float32)

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
        if model_path is None:
            model_path = f'models/{algo}/CartPole-v1.zip'
        self.model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)
    
    def __call__(self, seed):
        episode_start = np.ones((1,), dtype=bool)
        lstm_states = None
        observation, info = self.env.reset(seed=42, options={'state': seed})
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
    print(terminated)
    seed = seedspace.high
    episode_reward, terminated, info = execute(seed)
    print(terminated)
    mutate_seed = seedspace.mutate(seed)
    execute(mutate_seed)
    print("successfully testing!")
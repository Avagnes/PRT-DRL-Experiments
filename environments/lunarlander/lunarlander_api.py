import gymnasium
from rl_zoo3 import ALGOS
import sys
import torch
import numpy as np
from environments import SeedBase, ExecuteBase


class SeedSpace(SeedBase):
    def __init__(self):
        super().__init__(
            low=-np.array([1500, 1500]),
            high=np.array([1500, 1500]),
            shape=(2,),
            dim=2
        )


class Execute(ExecuteBase):
    def __init__(self, seed=42, render=False, algo:str = None, model_path: str = None) -> None:
        if render:
            self.env = gymnasium.make("LunarLanderContinuous-v3", render_mode="human")
        else:
            self.env = gymnasium.make("LunarLanderContinuous-v3")
        if algo is None:
            algo = "ppo"
        device = torch.device('cpu')
        self.obs_dim = self.env.observation_space.shape[0]
        self.reward_low_threshold = -200
        self.reward_high_threshold = 400

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
            model_path = f'models/{algo}/LunarLanderContinuous-v3.zip'

        self.model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)
    
    def __call__(self, seed):
        episode_start = np.ones((1,), dtype=bool)
        lstm_states = None
        observation, info = self.env.reset(seed=42, options={'force': seed})
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
        return episode_reward, info['crash'], info


if __name__ == "__main__":
    seedspace = SeedSpace()
    execute = Execute(render=True)
    seed = seedspace.low
    episode_reward, failure, info = execute([0,0])
    print(failure)
    seed = seedspace.high
    episode_reward, failure, info = execute(seed)
    print(failure)
    mutate_seed = seedspace.mutate(seed)
    episode_reward, failure, info = execute(mutate_seed)
    print(failure)
    print("successfully testing!")
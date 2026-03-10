import gymnasium
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class SeedBase:
    mutation_rate = 0.1
    def __init__(self, low: NDArray, high: NDArray, shape: tuple, dim: int):
        self.shape = shape  
        self.low = low
        self.high = high
        self.dim = dim
        self.magnitude = self.mutation_rate * (self.high - self.low) / 2

    def random_generate(self, number=None) -> NDArray:
        if number is None:
            return np.random.uniform(self.low, self.high)
        else:
            return np.random.uniform(self.low, self.high, size=(number, *self.shape))

    def mutate(self, seed) -> NDArray:
        seed = np.array(seed)
        delta = np.random.uniform(low=-self.magnitude, high=self.magnitude, size=self.dim)
        mutate_seed = (seed + delta).clip(min=self.low, max=self.high)
        return mutate_seed


class ExecuteBase(ABC):
    @abstractmethod
    def __init__(self, execute_random_seed: int, render: bool, algo:str = None, model_path: str = None):
        pass

    @abstractmethod
    def __call__(self, test_case) -> tuple[float, bool, dict]:
        episode_reward = 0.0
        failure = False
        info = {'success': True, 'state_sequence': [], 'action_sequence': []}
        return episode_reward, failure, info
    
    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return self.env.observation_space
    
    @observation_space.setter
    def observation_space(self, value):
        if isinstance(value, gymnasium.spaces.Box):
            self.env.observation_space = value
        else:
            raise ValueError("observation_space must be a gymnasium.spaces.Box")


from . import cartpole, mountaincar, lunarlander, humanoid
ENVS = {
    "CartPole-v1": [cartpole.SeedSpace, cartpole.Execute],
    "MountainCar-v0": [mountaincar.SeedSpace, mountaincar.Execute],
    "LunarLanderContinuous-v3": [lunarlander.SeedSpace, lunarlander.Execute],
    "LunarLander-v3": [lunarlander.SeedSpace, lunarlander.Execute],
    "Humanoid-v4": [humanoid.SeedSpace, humanoid.Execute]
}
from dataclasses import dataclass
from environments import SeedBase, ExecuteBase
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import time
import tqdm
import pickle
import os


@dataclass
class Args:
    env: str
    framework: str
    save_path: str = ""
    random_seed: int = 0
    render: bool = False
    number_cost: int = None
    time_cost: float = None


class Framework(ABC):
    def __init__(self, args: Args, SeedSpace: SeedBase, Execute: ExecuteBase, algo:str = None, model_path:str = None) -> None:
        self.all_test_cases: list[NDArray] = []
        self.result: list[NDArray] = []
        self.failure_time: list[float] = []
        self.args: Args = args
        self.seed_space: SeedBase = SeedSpace()
        self.executer: ExecuteBase = Execute(seed=args.random_seed, render=args.render, algo=algo, model_path=model_path)
        self.efficiency: float = None
        self.pbar: tqdm.tqdm = None

        print(args)
        import numpy as np
        np.random.seed(args.random_seed)
        import torch
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        import random
        random.seed(args.random_seed)
    
    @abstractmethod
    def test(self) -> None:
        raise NotImplementedError("Please construct your test method")
        # e.g. random testing
        start_test_time = time.time()
        while not self.terminate(time.time()-start_test_time):
            seed = self.seed_space.random_generate()
            episode_reward, failure, info = self.executer(seed)  # info['sequence']: list[NDArray]
            if 'success' in info and not info['success']:
                continue
            self.all_test_cases.append(seed)
            if failure:
                self.result.append(seed)
                self.failure_time.append((time.time()-start_test_time)/3600)
            if self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_postfix({'Found': len(self.result)})
        self.efficiency = len(self.all_test_cases) / (time.time()-start_test_time)
        if self.pbar is not None:
            self.pbar.close()

    @abstractmethod
    def terminate(self, test_time) -> bool:
        # examples can be referred in rq1, rq2, rq3
        raise NotImplementedError("Please construct your terminate condition")
    
    def save(self) -> None:
        os.makedirs(os.path.dirname(self.args.save_path), exist_ok=True)
        with open(self.args.save_path, 'wb') as f:
            pickle.dump((self.args, self.all_test_cases, self.result, self.failure_time, self.efficiency), f)


from .PRT import PRT_B, PRT_M
from .fuzz import GenerateSeeds, MDPFuzz, CureFuzz
from .RT import RT
from .g_model import G_Model
from .map_elites import Map_Elites
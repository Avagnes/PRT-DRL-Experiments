import torch
import numpy as np
import random
import tqdm
import os
import pickle
import copy
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from environments import ENVS, SeedBase, ExecuteBase
from frameworks import Framework
from abc import ABC, abstractmethod


@dataclass
class ExecuteData:
    last_reward: float
    last_final_state: Any
    current_reward: float
    failure: bool
    sequence: list
    actions: list


class GenerateSeeds:
    NUMBER = 2000
    REPEAT = None
    def generate(self, target_envs:list, threads:int = None):
        from multiprocessing import get_context
        root = Path(__file__).parent / 'seeds'
        args = []
        for env_id in target_envs:
            for random_seed in range(self.REPEAT):
                exist = False
                for file in root.glob(env_id+'_'+str(random_seed)+'*.pkl'):
                    number = int(file.stem.split('_')[-1])
                    if number >= self.NUMBER:
                        exist = True
                        break
                if not exist:
                    args.append((env_id, random_seed, self.NUMBER))
        if len(args):
            print(f"Generate Seeds: {len(args)}", )
            if threads is None:
                with get_context("spawn").Pool(maxtasksperchild=1) as pool:
                    pool.map(self._generate, args)
            else:
                with get_context("spawn").Pool(threads, maxtasksperchild=1) as pool:
                    pool.map(self._generate, args)

    @staticmethod
    def _generate(data):
        env_id, random_seed, number = data
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        random.seed(random_seed)
        SeedSpace, Execute = ENVS[env_id]
        seed_space: SeedBase = SeedSpace()
        executer = Execute(seed=random_seed)

        seeds = []
        info_seeds = []
        pbar = tqdm.tqdm(total=number)
        while pbar.n < number:
            seed = seed_space.random_generate()
            reward, collision, info = executer(seed)
            if 'success' in info and not info['success']:
                continue
            mutate_seed = seed_space.mutate(seed)
            mutate_reward, mutate_collision, mutate_info = executer(mutate_seed)
            if 'success' in mutate_info and not mutate_info['success']:
                continue
            if collision or mutate_collision:
                continue
            seeds.append(seed)
            execution_data = ExecuteData(
                last_reward=mutate_reward,
                last_final_state=mutate_info['state_sequence'][-1],
                current_reward=reward,
                failure=collision,
                sequence=info['state_sequence'],
                actions=info['action_sequence']
            )
            info_seeds.append((seed, execution_data))
            pbar.update(1)
        root = Path(__file__).parent
        os.makedirs(root / "seeds", exist_ok=True)
        with open(root / "seeds" / (env_id+'_'+str(random_seed)+'_'+str(number)+'.pkl'), 'wb') as f:
            pickle.dump(info_seeds, f)


class Fuzz(Framework):
    SeedNumber = GenerateSeeds.NUMBER
    def __init__(self, args, SeedSpace, Execute, algo:str = None, model_path:str = None):
        super().__init__(args, SeedSpace, Execute, algo, model_path)
        root = Path(__file__).parent / 'seeds'
        self.fuzzer = Fuzzer()
        self.guidance: GuidanceBase = None
        self.init_seeds = None

        for file in root.glob(args.env+'_'+str(args.random_seed)+'*.pkl'):
            number = int(file.stem.split('_')[-1])
            if number >= self.SeedNumber:
                with open(file, 'rb') as f:
                    self.init_seeds = pickle.load(f)[:self.SeedNumber]
        if self.init_seeds is None:
            GenerateSeeds._generate((args.env, args.random_seed, self.SeedNumber))
            with open(root / (args.env+'_'+str(args.random_seed)+'_'+str(self.SeedNumber)+'.pkl'), 'rb') as f:
                self.init_seeds = pickle.load(f)[:self.SeedNumber]

    def test(self):
        print('begin testing')
        start_test_time = time.time()
        while not self.terminate(time.time()-start_test_time): 
            seed, last_reward, last_final_state = self.fuzzer.get_pose()
            mutate_seed = self.seed_space.mutate(seed)
            episode_reward, failure, info = self.executer(mutate_seed)
            if 'success' in info and not info['success']:
                self.fuzzer.drop_current()
                continue
            self.all_test_cases.append(mutate_seed)
            if failure:
                self.result.append(mutate_seed)
                self.failure_time.append((time.time()-start_test_time)/3600)
            else:
                samplePr, add = self.guidance.analyze(ExecuteData(last_reward, last_final_state, episode_reward, failure, info['state_sequence'], info['action_sequence']))
                if add:
                    self.fuzzer.further_mutation((mutate_seed, episode_reward, info['state_sequence'][-1]), samplePr)
            if self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_postfix({'Found': len(self.result), 'corpus': len(self.fuzzer.corpus)})
        self.efficiency = len(self.all_test_cases) / (time.time()-start_test_time)
        if self.pbar is not None:
            self.pbar.close()


class Fuzzer:
    TIMES = 100
    def __init__(self):
        self.corpus = []
        self.count = []
        self.samplePr = []
        self.failures = []
        self.track_times = []
        self.track_pop = []
        self.track_failures = []
        self.current_index = None

    def get_pose(self):
        probs = np.array(self.samplePr)
        choose_index = np.random.choice(range(len(self.corpus)), 1, p=probs/probs.sum())[0]
        self.count[choose_index] -= 1
        self.current_index = choose_index
        self.current_pose = self.corpus[choose_index]
        self.track_times[choose_index][1] = self.track_times[choose_index][1] + 1
        if self.count[choose_index] <= 0:
            self.corpus.pop(choose_index)
            self.count.pop(choose_index)
            self.samplePr.pop(choose_index)
            self.track_pop.append(self.track_times.pop(choose_index))
            self.current_index = None
        return self.current_pose

    def add_crash(self, failures_pose):
        self.failures.append(failures_pose)
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.count.pop(choose_index)
            self.samplePr.pop(choose_index)
            self.track_failures.append(self.track_times.pop(choose_index))
            self.current_index = None
    
    def further_mutation(self, current_pose, samplePr):
        choose_index = self.current_index
        copy_pose = copy.deepcopy(current_pose)
        if choose_index != None:
            self.corpus[choose_index] = copy_pose
            self.samplePr[choose_index] = samplePr
            self.count[choose_index] = self.TIMES
        else:
            self.corpus.append(copy_pose)
            self.samplePr.append(samplePr)
            self.track_times.append([len(self.track_times), 0])
            self.count.append(self.TIMES)
    
    def drop_current(self):
        choose_index = self.current_index
        self.corpus.pop(choose_index)
        self.count.pop(choose_index)
        self.samplePr.pop(choose_index)
        self.track_failures.append(self.track_times.pop(choose_index))
        self.current_index = None


class GuidanceBase(ABC):
    def __init__(self, executer: ExecuteBase):
        pass

    @abstractmethod
    def analyze(self, data: ExecuteData) -> tuple[float, bool]:
        samplePr = 1.0
        add = True
        return samplePr, add
    
    def __call__(self, data):
        self.analyze(data)


from .MDPFuzz import MDPFuzz
from .CureFuzz import CureFuzz
from pathlib import Path
import tqdm
from environments import ENVS
import importlib
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import traceback
from frameworks import Args, Framework
from multiprocessing import get_context


SAVE_FOLDER = 'RQ3'
NUMBER_COST = 100000


class Terminate(Framework):
    def terminate(self, test_time) -> bool:
        if len(self.all_test_cases) >= NUMBER_COST:
            return True
        else:
            return False


def multi_test(data):
    try:
        Test, Terminate, args, SeedSpace, Execute, algo, model_path = data
        class CombinationClass(Test, Terminate):
            pass
        test_instance = CombinationClass(args, SeedSpace, Execute, algo, model_path)
        test_instance.pbar = tqdm.tqdm(total=NUMBER_COST)
        test_instance.test()
        test_instance.save()
        print("================================")
    except Exception as e:
        with open('error.log', 'a') as f:
            print(args, file=f)
            traceback.print_exc(file=f)


def execute_test(data):
    try:
        test_cases, Execute, algo, model_path = data
        executer = Execute(algo=algo, model_path=model_path)
        result = []
        for test_case in tqdm.tqdm(test_cases):
            episode_reward, failure, info = executer(test_case)
            if failure:
                result.append(test_case)
        return (list(test_cases), result)
    except Exception as e:
        with open('error.log', 'a') as f:
            print(model_path, file=f)
            traceback.print_exc(file=f)


if __name__ == "__main__":
    root = Path(__file__).parent

    algos = {
        'CartPole-v1': ['a2c', 'qrdqn', 'ppo'],
        'MountainCar-v0': ['ddpg', 'tqc', 'trpo'],
        'LunarLanderContinuous-v3': ['ppo', 'sac', 'td3'],
    }
    arg_list = []
    for env, framework in [("CartPole-v1", "PRT_B"), ("LunarLanderContinuous-v3", "PRT_B"), ("MountainCar-v0", "PRT_M")]:
        SeedSpace, Execute = ENVS[env]
        for algo in algos[env]:
            save_path = (root / SAVE_FOLDER / f"{framework}_{env}_{algo}_0.pkl").absolute()
            model_path = root / 'models' / algo / env
            args = Args(
                env,
                framework,
                random_seed=0,
                save_path=save_path,
                number_cost=NUMBER_COST
            )
            if os.path.exists(save_path):
                print('exist: ', save_path)
                continue
            module = importlib.import_module(f'frameworks')
            Test = getattr(module, framework)
            class CombinationClass(Test, Terminate):
                pass
            arg_list.append((Test, Terminate, args, SeedSpace, Execute, algo, model_path))
    # multi_test(arg_list[0])
    with get_context("spawn").Pool() as pool:
        pool.map(multi_test, arg_list)
    
    # failure patterns plot
    frameworks = ["PRT", "MDPFuzz", "CureFuzz", "G-Model", "QD", "RT"]
    plt.rcParams['text.usetex'] = True
    files = []
    for env in algos.keys():
        if env == "CartPole-v1":
            fig = plt.figure(figsize=(18,3.7), dpi=400)
        else:
            fig = plt.figure(figsize=(10,3.7), dpi=400)
        seed_space = ENVS[env][0]()
        for i, algo in enumerate(algos[env]):
            seeds = []
            failures = []
            for file in (root / 'RQ3').glob(f'*{env}*{algo}*.pkl'):
                with open(file, 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                seeds += all_test_cases
                failures += failure
            if len(seeds) == 0:
                continue
            seeds = np.array(seeds)
            failures = np.array(failures)
            if env == "CartPole-v1":
                fig.add_axes([0.04+0.15*i, 0.13, 0.14, 0.78])
                plt.scatter(seeds[:,0], seeds[:,2], s=1, color="green", alpha=0.15)
                plt.scatter(failures[:,0], failures[:,2], s=1, color="red", alpha=0.6)
                plt.xlim([seed_space.low[0], seed_space.high[0]])
                plt.ylim([seed_space.low[2], seed_space.high[2]])
                plt.title(algo.upper(), fontsize=18)
                if i == 0:
                    plt.ylabel(r'$\dot{x}$', fontsize=15)
                else:
                    plt.yticks([])
                plt.xlabel(r'$x$', fontsize=15)
                fig.add_axes([0.08+0.155*(i+3), 0.13, 0.14, 0.78])
                plt.scatter(seeds[:,1], seeds[:,3], s=1, color="green", alpha=0.15)
                plt.scatter(failures[:,1], failures[:,3], s=1, color="red", alpha=0.6)
                plt.xlim([seed_space.low[1], seed_space.high[1]])
                plt.ylim([seed_space.low[3], seed_space.high[3]])
                plt.title(algo.upper(), fontsize=18)
                if i == 0:
                    plt.ylabel(r'$\dot{\theta}$', fontsize=15)
                else:
                    plt.yticks([])
                plt.xlabel(r'$\theta$', fontsize=15)
            elif env == "LunarLanderContinuous-v3":
                fig.add_axes([0.078+0.314*i, 0.13, 0.28, 0.78])
                plt.scatter(seeds[:,0], seeds[:,1], s=1, color="green", alpha=0.15)
                plt.scatter(failures[:,0], failures[:,1], s=1, color="red", alpha=0.5)
                plt.xlim([seed_space.low[0], seed_space.high[0]])
                plt.ylim([seed_space.low[1], seed_space.high[1]])
                plt.title(algo.upper(), fontsize=18)
                if i == 0:
                    plt.ylabel(r'$f_y$', fontsize=15)
                else:
                    plt.yticks([])
                plt.xlabel(r'$f_x$', fontsize=15)
            elif env == "MountainCar-v0":
                fig.add_axes([0.08+0.31*i, 0.13, 0.28, 0.78])
                plt.scatter(seeds[:,0], seeds[:,1], s=1, color="green", alpha=0.15)
                plt.scatter(failures[:,0], failures[:,1], s=1, color="red", alpha=0.5)
                plt.xlim([seed_space.low[0], seed_space.high[0]])
                plt.ylim([seed_space.low[1], seed_space.high[1]])
                plt.title(algo.upper(), fontsize=18)
                if i == 0:
                    plt.ylabel(r'$v$', fontsize=15)
                else:
                    plt.yticks([])
                plt.xlabel(r'$x$', fontsize=15)
        plt.savefig(f'{env}_failure_pattern.jpg')
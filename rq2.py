from pathlib import Path
import tqdm
from environments import ENVS
import importlib
import os
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, no GUI required
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from frameworks import Args, Framework
from multiprocessing import get_context


SAVE_FOLDER = 'RQ2'
REPEAT_TIMES = 20


class Terminate(Framework):
    def terminate(self, test_time) -> bool:
        if len(self.all_test_cases) >= 10000:
            return True
        else:
            return False


def multi_test(data):
    try:
        Test, Terminate, args, SeedSpace, Execute = data
        class CombinationClass(Test, Terminate):
            pass
        if args.env == "Humanoid-v4" and args.framework == "MDPFuzz":
            test_instance = CombinationClass(args, SeedSpace, Execute, use_freshness=False)
        else:
            test_instance = CombinationClass(args, SeedSpace, Execute)
        test_instance.pbar = tqdm.tqdm(total=10000)
        test_instance.test()
        test_instance.save()
        print("================================")
    except Exception as e:
        with open('error.log', 'a') as f:
            print(args, file=f)
            traceback.print_exc(file=f)


if __name__ == "__main__":
    root = Path(__file__).parent

    arg_list = []
    for env in ["CartPole-v1", "LunarLander-v3", "Humanoid-v4"]:
        SeedSpace, Execute = ENVS[env]
        for framework in ["PRT_B", "MDPFuzz", "CureFuzz", "G_Model", "Map_Elites", "RT"]:
            for random_seed in range(REPEAT_TIMES):
                save_path = (root / SAVE_FOLDER / f"{framework}_{env}_{random_seed}.pkl").absolute()
                args = Args(
                    env,
                    framework,
                    save_path=save_path,
                    random_seed=random_seed,
                    number_cost=10000
                )
                if os.path.exists(save_path):
                    print('exist: ', save_path)
                    continue
                module = importlib.import_module(f'frameworks')
                Test = getattr(module, framework)
                arg_list.append((Test, Terminate, args, SeedSpace, Execute))
    for env in ["MountainCar-v0"]:
        SeedSpace, Execute = ENVS[env]
        for framework in ["PRT_M", "MDPFuzz", "CureFuzz", "G_Model", "Map_Elites", "RT"]:
            for random_seed in range(REPEAT_TIMES):
                save_path = (root / SAVE_FOLDER / f"{framework}_{env}_{random_seed}.pkl").absolute()
                args = Args(
                    env,
                    framework,
                    save_path=save_path,
                    random_seed=random_seed,
                    number_cost=10000
                )
                if os.path.exists(save_path):
                    print('exist: ', save_path)
                    continue
                module = importlib.import_module(f'frameworks')
                Test = getattr(module, framework)
                arg_list.append((Test, Terminate, args, SeedSpace, Execute))
    with get_context("spawn").Pool() as pool:
        pool.map(multi_test, arg_list)

    arg_list = []
    for env in ["Humanoid-v4"]:
        SeedSpace, Execute = ENVS[env]
        for framework in ["PRT_B", "MDPFuzz", "CureFuzz", "G_Model", "Map_Elites", "RT"]:
            for random_seed in range(REPEAT_TIMES):
                save_path = (root / SAVE_FOLDER / f"{framework}_{env}_{random_seed}.pkl").absolute()
                args = Args(
                    env,
                    framework,
                    save_path=save_path,
                    random_seed=random_seed,
                    number_cost=10000
                )
                if os.path.exists(save_path):
                    print('exist: ', save_path)
                    continue
                module = importlib.import_module(f'frameworks')
                Test = getattr(module, framework)
                arg_list.append((Test, Terminate, args, SeedSpace, Execute))
    print(len(arg_list))
    with get_context("spawn").Pool(2) as pool:
        pool.map(multi_test, arg_list)

    frameworks = ["PRT", "MDPFuzz", "CureFuzz", "G-Model", "QD", "RT"]
    NAMES = {"G-Model": "G_Model", "QD": "Map_Elites"}
    
    # distribution plot
    plt.rcParams['text.usetex'] = True
    for env in ["CartPole", "LunarLander", "MountainCar", "Humanoid"]:
        files = []
        for framework in frameworks:
            f1_time = np.inf
            for file in (root / "RQ2").glob(f'{NAMES[framework] if framework in NAMES else framework}*{env}*.pkl'):
                with open(file, 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                if len(failure) == 0:
                    continue
                if failure_time[0] < f1_time:
                    f1_time = failure_time[0]
                    target_file = file
            files.append(target_file)
        fig = plt.figure(figsize=(18,3.7), dpi=300)
        seed_space = ENVS[args.env][0]()
        if env == "CartPole":
            for i, framework in enumerate(frameworks):
                with open(files[i], 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                fig.add_axes([0.05+0.16*i, 0.13, 0.14, 0.78])
                all_test_cases = np.array(all_test_cases)
                failure = np.array(failure)
                plt.scatter(all_test_cases[:,0], all_test_cases[:,2], s=3, color="green", alpha=0.45)
                plt.scatter(failure[:,0], failure[:,2], s=3, color="red", alpha=0.7)
                plt.xlim([seed_space.low[0], seed_space.high[0]])
                plt.ylim([seed_space.low[2], seed_space.high[2]])
                plt.title(framework, fontsize=24)
                if i == 0:
                    plt.ylabel(r'$\dot{x}$', fontsize=17)
                else:
                    plt.yticks([])
                plt.xlabel(r'$x$', fontsize=17)
            plt.savefig('cartpole_x_seeds_pattern.jpg')
            fig = plt.figure(figsize=(18,3.7), dpi=300)
            for i, framework in enumerate(frameworks):
                with open(files[i], 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                fig.add_axes([0.05+0.16*i, 0.13, 0.14, 0.78])
                all_test_cases = np.array(all_test_cases)
                failure = np.array(failure)
                plt.scatter(all_test_cases[:,1], all_test_cases[:,3], s=3, color="green", alpha=0.4)
                plt.scatter(failure[:,1], failure[:,3], s=3, color="red", alpha=0.7)
                plt.xlim([seed_space.low[1], seed_space.high[1]])
                plt.ylim([seed_space.low[3], seed_space.high[3]])
                plt.title(framework, fontsize=24)
                if i == 0:
                    plt.ylabel(r'$\dot{\theta}$', fontsize=17)
                else:
                    plt.yticks([])
                plt.xlabel(r'$\theta$', fontsize=17)
            plt.savefig('cartpole_theta_seeds_pattern.jpg')
        elif env == "LunarLander":
            for i, framework in enumerate(frameworks):
                with open(files[i], 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                fig.add_axes([0.05+0.16*i, 0.13, 0.14, 0.78])
                all_test_cases = np.array(all_test_cases)
                failure = np.array(failure)
                plt.scatter(all_test_cases[:,0], all_test_cases[:,1], s=3, color="green", alpha=0.4)
                plt.scatter(failure[:,0], failure[:,1], s=10, color="red", alpha=0.7)
                plt.xlim([seed_space.low[0], seed_space.high[0]])
                plt.ylim([seed_space.low[1], seed_space.high[1]])
                plt.title(framework, fontsize=24)
                if i == 0:
                    plt.ylabel(r'$f_y$', fontsize=17)
                else:
                    plt.yticks([])
                plt.xlabel(r'$f_x$', fontsize=17)
            plt.savefig('lunarlander_seeds_pattern.jpg')
        elif env == "MountainCar":
            for i, framework in enumerate(frameworks):
                with open(files[i], 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                fig.add_axes([0.05+0.16*i, 0.13, 0.14, 0.78])
                all_test_cases = np.array(all_test_cases)
                failure = np.array(failure)
                plt.scatter(all_test_cases[:,0], all_test_cases[:,1], s=3, color="green", alpha=0.4)
                plt.scatter(failure[:,0], failure[:,1], s=6, color="red", alpha=0.7)
                plt.xlim([seed_space.low[0], seed_space.high[0]])
                plt.ylim([seed_space.low[1], seed_space.high[1]])
                plt.title(framework, fontsize=24)
                if i == 0:
                    plt.ylabel(r'$v$', fontsize=17)
                else:
                    plt.yticks([])
                plt.xlabel(r'$x$', fontsize=17)
            plt.savefig('mountaincar_seeds_pattern.jpg')
        elif env == "Humanoid":
            for i, framework in enumerate(frameworks):
                with open(files[i], 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                all_test_cases = np.array(all_test_cases)
                failure = np.array(failure)

                from entropy import knn_entropy
                import itertools
                entropy_rank = []
                data_normalized = (all_test_cases-seed_space.low) / (seed_space.high-seed_space.low)
                for combo in tqdm.tqdm(list(itertools.combinations(range(47), 2))):
                    data = data_normalized[:,combo]
                    entropy = knn_entropy(data)
                    entropy_rank.append((combo, entropy))
                entropy_rank.sort(key=lambda x: x[1])
                
                xdim, ydim = entropy_rank[0][0]
                fig = plt.figure('min_e', figsize=(18,3.7), dpi=300)
                fig.add_axes([0.05+0.16*i, 0.13, 0.14, 0.78])
                plt.scatter(all_test_cases[:,xdim], all_test_cases[:,ydim], s=3, color="green", alpha=0.4)
                plt.scatter(failure[:,xdim], failure[:,ydim], s=6, color="red", alpha=0.7)
                plt.xlim([seed_space.low[xdim], seed_space.high[xdim]])
                plt.ylim([seed_space.low[ydim], seed_space.high[ydim]])
                plt.title(f'{framework} (E={entropy_rank[0][1]:.2f})', fontsize=24)
                if i == 0:
                    plt.ylabel(f'{seed_space.definition[ydim]}', fontsize=17)
                else:
                    plt.ylabel(f'{seed_space.definition[ydim]}', fontsize=17)
                    plt.yticks([])
                plt.xlabel(f'{seed_space.definition[xdim]}', fontsize=17)

                xdim, ydim = entropy_rank[-1][0]
                fig = plt.figure('max_e', figsize=(18,3.7), dpi=300)
                fig.add_axes([0.05+0.16*i, 0.13, 0.14, 0.78])
                plt.scatter(all_test_cases[:,xdim], all_test_cases[:,ydim], s=3, color="green", alpha=0.4)
                plt.scatter(failure[:,xdim], failure[:,ydim], s=6, color="red", alpha=0.7)
                plt.xlim([seed_space.low[xdim], seed_space.high[xdim]])
                plt.ylim([seed_space.low[ydim], seed_space.high[ydim]])
                plt.title(f'{framework} (E={entropy_rank[-1][1]:.2f})', fontsize=24)
                if i == 0:
                    plt.ylabel(f'{seed_space.definition[ydim]}', fontsize=17)
                else:
                    plt.ylabel(f'{seed_space.definition[ydim]}', fontsize=17)
                    plt.yticks([])
                plt.xlabel(f'{seed_space.definition[xdim]}', fontsize=17)
            fig = plt.figure('min_e')
            plt.savefig('humanoid_mine_seeds_pattern.jpg')
            fig = plt.figure('max_e')
            plt.savefig('humanoid_maxe_seeds_pattern.jpg')


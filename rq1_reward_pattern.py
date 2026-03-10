import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from multiprocessing import Pool, get_context
from numpy.typing import NDArray
from environments import ENVS, SeedBase, ExecuteBase
import tqdm
import pickle
from matplotlib.colors import LinearSegmentedColormap

root = Path(__file__).parent

def main():
    colors_humanoid = [
        (0.0, '#440154'),   
        (0.8, '#3B528B'),   
        # (0.94, '#21918C'),   
        # (0.97, '#35B779'),   
        (0.99, '#90D743'),   
        (0.995, '#FDE725'),   
        (1.0, '#FFD700')    
    ]
    colors_mountaincar = [
        (0.0, '#440154'),   
        (0.8, '#3B528B'),   
        # (0.94, '#21918C'),   
        # (0.97, '#35B779'),   
        (0.95, '#90D743'),   
        (0.99, '#FDE725'),   
        (1.0, '#FFD700')    
    ]
    colors_lunarlander = [
        (0.0, '#440154'),   
        (0.2, '#3B528B'),   
        # (0.5, '#21918C'),   
        # (0.97, '#35B779'),   
        (0.6, '#90D743'),   
        (0.8, '#FDE725'),   
        (1.0, '#FFD700')    
    ]
    custom_cmap = LinearSegmentedColormap.from_list('enhanced_yellow_green', colors_lunarlander)
    NUMBER = 10000
    
    env = 'CartPole-v1'
    SeedSpace, Execute = ENVS[env]
    seed_space = SeedSpace()
    result = _calculate_reward(env, NUMBER)
    with open(f'{env}_reward.pkl', 'wb') as f:
        pickle.dump(result, f)
    # import sys
    # sys.exit('0')
    # with open(f'LunarLander-v3_reward.pkl', 'rb') as f:
    #     result = pickle.load(f)
    result.sort(key=lambda x: x[1], reverse=True)
    # result = result[-10000:]
    tcases, rewards = zip(*result)
    tcases = np.array(tcases)#[:,[4,5]]

    fig = plt.figure(figsize=(4.1, 3.3), dpi=600)
    plt.rcParams['text.usetex'] = True
    ax = fig.add_axes([0.205, 0.15, 0.71, 0.75])
    scatter = plt.scatter(tcases[:,0], tcases[:,1], c=rewards, cmap=custom_cmap, s=2, alpha=0.9)
    plt.title('Lunar Lander', fontsize=24)
    plt.xlabel(r'$f_x$', fontsize=16)
    plt.ylabel(r'$f_y$', fontsize=16)
    plt.xlim([seed_space.low[0], seed_space.high[0]])
    plt.ylim([seed_space.low[1], seed_space.high[1]])
    # ax = fig.add_axes([0.56, 0.12, 0.41, 0.79])
    # scatter = plt.scatter(tcases[:,1], tcases[:,3], c=rewards, cmap=custom_cmap, s=6, alpha=0.6)
    # plt.title('Humanoid', fontsize=18)
    # plt.xlabel(r'$\theta$', fontsize=16)
    # plt.ylabel(r'$\dot{\theta}$', fontsize=16)
    # plt.xlim([seed_space.low[1], seed_space.high[1]])
    # plt.ylim([seed_space.low[3], seed_space.high[3]])
    colors = plt.colorbar(scatter)
    colors.set_label('reward', fontsize=16)
    plt.savefig('lunarlander_reward_pattern.jpg')


def _calculate_reward(env, number):
    SeedSpace, Execute = ENVS[env]
    seed_space: SeedBase = SeedSpace()
    executer: ExecuteBase = Execute()
    result: list[tuple[NDArray, float]] = []
    for _ in tqdm.tqdm(list(range(number))):
        tcase = seed_space.random_generate()
        reward, failure, info = executer(tcase)
        result.append((tcase, reward))
    return result


if __name__ == "__main__":
    main()
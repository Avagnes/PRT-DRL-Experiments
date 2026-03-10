from pathlib import Path
import pickle
import numpy as np
from scipy.stats import mannwhitneyu
import scipy.stats as stats
from typing import Literal


def calculate_effect_size(group1, group2, method: Literal['A12', 'A21']='A12'):
    x = np.asarray(group1)
    y = np.asarray(group2)
    comparisons = np.sign(x[:, np.newaxis] - y[np.newaxis, :])
    if method == 'A12':
        p_greater = np.mean(comparisons > 0)
    else:
        p_greater = np.mean(comparisons < 0)
    p_equal = np.mean(comparisons == 0)
    ps = p_greater + 0.5 * p_equal
    return ps


if __name__ == "__main__":
    root = Path(__file__).parent

    f1_result = {'number': {}, 'time': {}}
    for env in ["CartPole-v1", "LunarLander-v3", "MountainCar-v0", "Humanoid-v4"]:
        f1_result['number'][env] = {}
        f1_result['time'][env] = {}
        for framework in ["PRT", "CureFuzz", "G_Model", "RT", "Map_Elites", "MDPFuzz"]:
            number = []
            time = []
            for file in (root / 'RQ1').glob(f'{framework}*{env}*.pkl'):
                with open(file, 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                if len(failure):
                    for k, tcase in enumerate(all_test_cases):
                        if (tcase == failure[0]).all():
                                break
                    number.append(k+1)
                else:
                    # print(f'No failures: {args}')
                    number.append(len(all_test_cases))
                if len(failure_time):
                    time.append(failure_time[0])
                else:
                    time.append(len(all_test_cases) / efficiency)
            f1_result['number'][env][framework] = number
            f1_result['time'][env][framework] = time

    for env in ["CartPole-v1", "LunarLander-v3", "MountainCar-v0", "Humanoid-v4"]:
        print("PRT", env, np.average(f1_result['number'][env]["PRT"]), np.average(f1_result['time'][env]["PRT"])*3600)
        for framework in ["CureFuzz", "G_Model", "RT", "Map_Elites", "MDPFuzz"]:
            print(framework, env, np.average(f1_result['number'][env][framework]), np.average(f1_result['time'][env][framework])*3600)
            # print(f1_result['number'][env]["PRT"])
            # print(f1_result['number'][env][framework])
            # print(f1_result['time'][env]["PRT"])
            # print(f1_result['time'][env][framework])
            t_stat, p_value = stats.mannwhitneyu(f1_result['number'][env][framework], f1_result['number'][env]["PRT"], alternative='greater')
            effect_size = calculate_effect_size(f1_result['number'][env][framework], f1_result['number'][env]["PRT"])
            print(f"({p_value:.3f},{effect_size:.2f})", end="|")
            t_stat, p_value = stats.mannwhitneyu(f1_result['time'][env][framework], f1_result['time'][env]["PRT"], alternative='greater')
            effect_size = calculate_effect_size(f1_result['time'][env][framework], f1_result['time'][env]["PRT"])
            print(f"({p_value:.3f},{effect_size:.2f})")
        print('\n')
    

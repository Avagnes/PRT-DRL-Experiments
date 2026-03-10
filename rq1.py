from pathlib import Path
import tqdm
from environments import ENVS
import importlib
import os
import pickle
import numpy as np
from frameworks import Args, Framework, GenerateSeeds  # for fuzzing


SAVE_FOLDER = 'RQ1'
REPEAT_TIMES = 20


class Terminate(Framework):
    def terminate(self, test_time) -> bool:
        if len(self.result) or len(self.all_test_cases) >= 100000:
            return True
        else:
            return False


if __name__ == "__main__":
    GenerateSeeds.REPEAT = REPEAT_TIMES
    seed_generator = GenerateSeeds()
    seed_generator.generate(["CartPole-v1", "LunarLander-v3", "MountainCar-v0"])
    seed_generator.generate(["Humanoid-v4"], threads=2)
    
    root = Path(__file__).parent
    for env in ["CartPole-v1", "LunarLander-v3", "Humanoid-v4"]:
        SeedSpace, Execute = ENVS[env]
        for framework in ["PRT_B", "RT", "CureFuzz", "G_Model", "Map_Elites", "MDPFuzz"]:
            for random_seed in range(REPEAT_TIMES):
                save_path = (root / SAVE_FOLDER / f"{framework}_{env}_{random_seed}.pkl").absolute()
                args = Args(
                    env,
                    framework,
                    save_path=save_path,
                    random_seed=random_seed
                )
                if os.path.exists(save_path):
                    print('exist: ', save_path)
                    continue
                module = importlib.import_module(f'frameworks')
                Test = getattr(module, framework)
                class CombinationClass(Test, Terminate):
                    pass
                if env == "Humanoid-v4" and framework == "MDPFuzz":
                    test_instance = CombinationClass(args, SeedSpace, Execute, use_freshness=False)
                else:
                    test_instance = CombinationClass(args, SeedSpace, Execute)
                test_instance.pbar = tqdm.tqdm(total=100000)
                test_instance.test()
                test_instance.save()
                print("================================")
    for env in ["MountainCar-v0"]:
        SeedSpace, Execute = ENVS[env]
        for framework in ["PRT_M", "MDPFuzz", "CureFuzz", "G_Model", "Map_Elites", "RT"]:
            for random_seed in range(REPEAT_TIMES):
                save_path = (root / SAVE_FOLDER / f"{framework}_{env}_{random_seed}.pkl").absolute()
                args = Args(
                    env,
                    framework,
                    save_path=save_path,
                    random_seed=random_seed
                )
                if os.path.exists(save_path):
                    print('exist: ', save_path)
                    continue
                module = importlib.import_module(f'frameworks')
                Test = getattr(module, framework)
                class CombinationClass(Test, Terminate):
                    pass
                test_instance = CombinationClass(args, SeedSpace, Execute)
                test_instance.pbar = tqdm.tqdm(total=100000)
                test_instance.test()
                test_instance.save()
                print("================================")
    
    f1_result = {'number': {}, 'time': {}}
    for env in ["CartPole-v1", "LunarLander-v3", "MountainCar-v0", "Humanoid-v4"]:
        f1_result['number'][env] = {}
        f1_result['time'][env] = {}
        for framework in ["PRT", "MDPFuzz", "CureFuzz", "G_Model", "Map_Elites", "RT"]:
            number = []
            time = []
            for file in (root / SAVE_FOLDER).glob(f'{framework}*{env}*.pkl'):
                with open(file, 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                if len(failure):
                    for k, tcase in enumerate(all_test_cases):
                        if (tcase == failure[0]).all():
                                break
                    number.append(k+1)
                else:
                    print(f'No failures: {args}')
                    number.append(len(all_test_cases))
                if len(failure_time):
                    time.append(failure_time[0])
                else:
                    time.append(len(all_test_cases) / efficiency)
            f1_result['number'][env][framework] = np.mean(number)
            f1_result['time'][env][framework] = np.mean(time)*3600
    import json
    with open('f1_result.json', 'w', encoding='utf-8') as f:
        json.dump(f1_result, f, indent=2)
    print(f1_result)
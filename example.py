import argparse
import tqdm
from environments import ENVS
import importlib
from frameworks import Args, Framework


SAVE_FOLDER = 'your_folder'
REPEAT_TIMES = 10


class Terminate(Framework):
    def terminate(self, test_time) -> bool:
        if len(self.result) or len(self.all_test_cases) >= 10000:
            return True
        else:
            return False
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", choices=["CartPole-v1", "MountainCar-v0", "LunarLander-v3"])
    parser.add_argument("--framework", type=str, default="RT", choices=["PRT_B", "PRT_M", "MDPFuzz", "CureFuzz", "G_Model", "Map_Elites", "RT"])
    parser.add_argument("--save_path", type=str, default=SAVE_FOLDER)
    parser.add_argument("--random_seed", help="Random Seed", type=int, default=0)
    parser.add_argument("--render", help="display screen", default=False, action="store_true")
    args = parser.parse_args()
    args = Args(**dict(args._get_kwargs()))

    SeedSpace, Execute = ENVS[args.env]
    module = importlib.import_module(f'frameworks')
    Test = getattr(module, args.framework)
    class CombinationClass(Test, Terminate):
        pass
    test_instance = CombinationClass(args, SeedSpace, Execute)
    test_instance.pbar = tqdm.tqdm(total=10000)
    test_instance.test()
    test_instance.save()
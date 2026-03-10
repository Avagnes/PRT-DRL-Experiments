import time
from frameworks import Framework


class RT(Framework):
    def test(self):
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


import time
import numpy as np
from environments import SeedBase
from numpy.typing import NDArray
import bisect
from frameworks import Framework


class PRT_B(Framework):
    def test(self):
        generator = NewGenerator(self.seed_space)

        start_test_time = time.time()
        while not self.terminate(time.time()-start_test_time):
            if len(generator.corpus[0]) < 1:
                seed = self.seed_space.random_generate()
            else:
                seed = generator.generate()
            episode_reward, failure, info = self.executer(seed)
            generator.add(seed)
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


class PRT_M(Framework):
    def test(self):
        generator = NewGenerator(self.seed_space)

        start_test_time = time.time()
        while not self.terminate(time.time()-start_test_time):
            if len(generator.corpus[0]) < 1:
                seed = self.seed_space.random_generate()
                tseed = seed
            else:
                seed = generator.generate()
                # mapping `seed` -> `tseed`, here mapping the boundary to the middle
                tseed = seed.copy()
                larger_dimension = seed > (self.seed_space.low + self.seed_space.high) / 2
                tseed[larger_dimension] = (self.seed_space.low[larger_dimension] + 3 * self.seed_space.high[larger_dimension]) / 2 - seed[larger_dimension]
                tseed[~larger_dimension] = (3 * self.seed_space.low[~larger_dimension] + self.seed_space.high[~larger_dimension]) / 2 - seed[~larger_dimension]
            episode_reward, failure, info = self.executer(tseed)
            generator.add(seed)
            if 'success' in info and not info['success']:
                continue
            self.all_test_cases.append(tseed)
            if failure:
                self.result.append(tseed)
                self.failure_time.append((time.time()-start_test_time)/3600)
            if self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_postfix({'Found': len(self.result)})
        self.efficiency = len(self.all_test_cases) / (time.time()-start_test_time)
        if self.pbar is not None:
            self.pbar.close()


class NewGenerator:
    MAXN = 10**6  # maximal generated test cases
    LAMBDA = 20
    def __init__(self, seed_space: SeedBase):
        self.dim = seed_space.dim
        self.high = seed_space.high.flatten()
        self.low = seed_space.low.flatten()
        self.seed_space = seed_space
        self.seed_length = self.high - self.low
        self.corpus: list[list[float]] = [[] for _ in range(self.dim)]
        self.corpus_id: list[list[int]] = [[] for _ in range(self.dim)]
        self.id2corpus = np.zeros((self.MAXN, self.dim), dtype=np.int32)
        self.count = 1
        self.resolution = None
    
    def add(self, seed):
        seed = seed.flatten()
        for s, corpus, id, index in zip(seed, self.corpus, self.corpus_id, self.id2corpus.T):
            i = bisect.bisect_right(corpus, s)
            index[id[i:]] += 1
            corpus.insert(i, s)
            id.insert(i, self.count)
            index[self.count] = i
        self.count += 1
        self.resolution = 1/(self.count**(1/self.dim))
    
    def generate(self) -> NDArray:
        result = np.zeros(self.dim)
        self._reduction(np.array(self.corpus), np.array(self.corpus), np.array(self.corpus_id), np.array(self.corpus_id), list(range(self.dim)), result)
        result = result.reshape(self.seed_space.shape)
        return result
    
    def _reduction(self, corpus:NDArray, pdata: NDArray, corpus_id:NDArray, pid: NDArray, dimensions: list, result: NDArray):
        if pdata.shape[1] == 1:
            interval = np.zeros(len(dimensions))
            solution = np.zeros(len(dimensions))
        else:
            solution = np.zeros(len(dimensions))
            interval = (pdata[:,1:] - pdata[:,:-1]) / 2
            maxk = interval.argmax(axis=1)
            interval = interval[range(len(dimensions)), maxk]
            prob = np.random.uniform(size=len(dimensions))
            small_mask = prob < 0.5
            big_mask = ~small_mask
            prob[big_mask] = 1 - prob[big_mask]
            solution[small_mask] = pdata[small_mask, maxk[small_mask]] + (0.5-(2**(self.LAMBDA-1))*((0.5-prob[small_mask])**self.LAMBDA))*interval[small_mask]
            solution[big_mask] = pdata[big_mask, maxk[big_mask]+1] - (0.5-(2**(self.LAMBDA-1))*((0.5-prob[big_mask])**self.LAMBDA))*interval[big_mask]
        infimum = interval <= pdata[:,0] - self.low[dimensions]
        if infimum.any():
            interval[infimum] = pdata[infimum,0]-self.low[dimensions][infimum]
            prob = np.random.uniform(0,0.5,size=infimum.sum())
            solution[infimum] = pdata[infimum,0] - (1-(2**self.LAMBDA)*((0.5-prob)**self.LAMBDA))*interval[infimum]
        supermum = interval <= self.high[dimensions] - pdata[:,-1]
        if supermum.any():
            interval[supermum] = self.high[dimensions][supermum]-pdata[supermum,-1]
            prob = np.random.uniform(0,0.5,size=supermum.sum())
            solution[supermum] = pdata[supermum,-1] + (1-(2**self.LAMBDA)*((0.5-prob)**self.LAMBDA))*interval[supermum]
        interval = interval / self.seed_length[dimensions]
        result_mask = interval > self.resolution
        if result_mask.any():
            result_id = np.where(result_mask)[0].tolist()
            result_id.reverse()
            for id in result_id:
                result[dimensions[id]] = solution[id]
                dimensions.pop(id)
            if len(dimensions) == 0:
                return
            pdata = pdata[~result_mask,:]
            pid = pid[~result_mask,:]
            interval = interval[~result_mask]
            solution = solution[~result_mask]
        dimension = interval.argmax()
        result[dimensions[dimension]] = solution[dimension]
        if len(dimensions) == 1:
            return

        dist = np.abs(pdata[dimension,:]-solution[dimension])
        threshold = np.quantile(dist, self.resolution, method='higher')
        similar_mask = dist <= threshold
        dimensions.pop(dimension)
        similar_id = pid[dimension,similar_mask]
        pix = self.id2corpus[np.ix_(similar_id, dimensions)]
        index_matrix = np.zeros(corpus.shape, dtype=np.bool_)
        index_matrix[dimensions, pix] = True
        pdata = corpus[index_matrix].reshape(len(dimensions), len(pix))
        pid = corpus_id[index_matrix].reshape(len(dimensions), len(pix))
        self._reduction(corpus, pdata, corpus_id, pid, dimensions, result)


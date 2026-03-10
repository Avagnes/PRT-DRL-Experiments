import numpy as np
import time
from collections import Counter
import random
from frameworks import Framework


class Map_Elites(Framework):
    def test(self):
        INIT_BUDGET = 100
        BINS = 20
        
        start_test_time = time.time()
        behaviors = []
        rewards = []
        while len(behaviors) < INIT_BUDGET and not self.terminate(time.time()-start_test_time):
            tcase = self.seed_space.random_generate()
            episode_reward, collision, info = self.executer(tcase)
            if 'success' in info and not info['success']:
                continue
            self.all_test_cases.append(tcase)
            rewards.append(episode_reward)
            if collision:
                self.result.append(tcase)
                self.failure_time.append((time.time()-start_test_time)/3600)
            behavior = get_behavior(self.args.env, info['state_sequence'], info['action_sequence'])
            behaviors.append(behavior)
            if self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_postfix({'Found': len(self.result)})
        if self.terminate(time.time()-start_test_time):
            return
        behaviors = np.array(behaviors)
        _H, xedges, yedges = np.histogram2d(behaviors[:, 0], behaviors[:, 1], bins=BINS)
        edge = np.stack((xedges, yedges))
        behavior_id = np.tile(behaviors.reshape(len(behaviors),2,1),(1,1,len(xedges))) - edge
        behavior_id = np.int32((behavior_id >= 0).sum(axis=2) - 1)

        corpus: dict[tuple, dict[str, list]] = {}
        ids: list[tuple] = []
        for k in range(INIT_BUDGET):
            id = tuple(behavior_id[k])
            insert(corpus, ids, id, self.all_test_cases[k], rewards[k])

        print('begin testing')
        while not self.terminate(time.time()-start_test_time):
            id = random.sample(ids, k=1)[0]
            min_reward = np.min(corpus[id]['rewards'])
            idxs = np.where(corpus[id]['rewards'] == min_reward)[0]
            min_reward_index = random.sample(idxs.tolist(), k=1)[0]
            seed = corpus[id]['test_cases'][min_reward_index]
            mutate_seed = self.seed_space.mutate(seed)
            episode_reward, collision, info = self.executer(mutate_seed)
            if 'success' in info and not info['success']:
                for v in corpus[id].values():
                    v.pop(min_reward_index)
                if len(v) == 0:
                    ids.remove(id)
                    del corpus[id]
                continue
            self.all_test_cases.append(mutate_seed)
            if collision:
                self.result.append(mutate_seed)
                self.failure_time.append((time.time()-start_test_time)/3600)
            behavior = get_behavior(self.args.env, info['state_sequence'], info['action_sequence'])
            behavior_id = tuple(np.int32((np.tile(behavior.reshape(-1,1), (1,len(xedges))) - edge).sum(axis=1) - 1))
            insert(corpus, ids, behavior_id, mutate_seed, episode_reward)
            if self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_postfix({'Found': len(self.result)})
        self.efficiency = len(self.all_test_cases) / (time.time()-start_test_time)
        if self.pbar is not None:
            self.pbar.close()


def get_behavior(env_id, sequence, actions):
    actions = np.array(actions).tolist()
    if env_id == "CartPole-v1":
        cnt = Counter(actions)
        return np.array([cnt[0], cnt[1]])
    elif env_id == "LunarLander-v3":
        seq_array = np.array(sequence)
        flag = np.bool_(seq_array[:,6]) | np.bool_(seq_array[:,7])
        index = np.where(flag)[0][0]
        return seq_array[index,[0,3]]
    elif env_id == "MountainCar-v0":
        if 2 in actions:
            cnt = Counter(actions)
            return np.array([cnt[0], cnt[2]])
        else:
            action_array = np.array(actions)
            return np.array([(action_array < 0).sum(), (action_array > 0).sum()])
    elif env_id == "Humanoid-v4":
        action_array = np.array(actions)
        return np.array([(action_array < 0).sum(), (action_array > 0).sum()])
    else:
        raise KeyError(f"please set behavior definition for {env_id}")


def insert(corpus, ids, id, tcase, reward):
    if id not in corpus:
        corpus[id] = {'test_cases': [tcase], 'rewards': [reward]}
        ids.append(id)
    else:
        corpus[id]['test_cases'].append(tcase)
        corpus[id]['rewards'].append(reward)
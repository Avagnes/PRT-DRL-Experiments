import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree
from math import gamma, pi


def knn_entropy(X, k=10):
    """
    Kozachenko–Leonenko kNN entropy estimator.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Data points in [0, 1]^d
    k : int
        Number of nearest neighbors

    Returns
    -------
    H : float
        Estimated differential entropy
    """
    X = np.asarray(X)
    N, d = X.shape

    # Build KD-tree
    tree = cKDTree(X)

    # Query k+1 because the nearest neighbor is the point itself
    distances, _ = tree.query(X, k=k + 1, p=2)

    # Distance to k-th nearest neighbor
    eps = distances[:, k]

    # Volume of d-dimensional unit ball
    c_d = (pi ** (d / 2)) / gamma(d / 2 + 1)

    H = (
        digamma(N)
        - digamma(k)
        + np.log(c_d)
        + (d / N) * np.sum(np.log(eps + 1e-12))
    )
    return H


if __name__ == "__main__":
    from environments import ENVS
    from pathlib import Path
    import pickle
    import numpy as np
    import pandas as pd


    # with open('RQ2/G_Model_Humanoid-v4_1.pkl', 'rb') as f:
    #     args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
    # data = np.array(all_test_cases)
    # SeedSpace, _ = ENVS[args.env]
    # seed_space = SeedSpace()
    # data_normalized = (data - seed_space.low) / (seed_space.high - seed_space.low)
    # print(args.env, args.framework, knn_entropy(data_normalized))

    root = Path('.')
    
    records = []
    for framework in ["PRT", "MDPFuzz", "CureFuzz", "G_Model", "Map_Elites", "RT"]:
        result_dict: dict[str, str] = {}
        for env in ["CartPole-v1", "LunarLander-v3", "MountainCar-v0", "Humanoid-v4"]:
            entropy = []
            for file in (root / 'RQ2').glob(f'{framework}*{env}*.pkl'):
                with open(file, 'rb') as f:
                    args, all_test_cases, failure, failure_time, efficiency = pickle.load(f)
                if len(all_test_cases) != 10000:
                    print(file)
                data = np.array(all_test_cases)
                SeedSpace, _ = ENVS[args.env]
                seed_space = SeedSpace()
                data_normalized = (data - seed_space.low) / (seed_space.high - seed_space.low)
                H = knn_entropy(data_normalized)
                entropy.append(H)
            # records.append({
            #     "env": args.env,
            #     "framework": args.framework,
            #     "entropy": np.average(entropy),
            #     "std": np.std(entropy)
            # })
            result_dict[env] = f"${np.average(entropy):.4f} \\pm {np.std(entropy):.4f}$"
        records.append(result_dict)
    df_entropy = pd.DataFrame.from_records(records, index=["PRT", "MDPFuzz", "CureFuzz", "G_Model", "Map_Elites", "RT"])

    print(df_entropy)
    df_entropy.to_latex('entropy.tex')
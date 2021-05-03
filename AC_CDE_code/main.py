import numpy as np, multiprocessing
from multiprocessing import Pipe
from utils import ME, DE, CDE, AC_CDE, Env, G1, G2, G3, save_data

r = 0.15
n_runs = 2
id = None
alg_id = None
alg_ops = [ME(), DE(), CDE(), AC_CDE()]
p_ops = [G1(), G2(), G3()]

def f(n_action, p_op_idx, n_sample, n_repeat, alg_idx, _i, _j, K=None, pipe=None, seed=None):
    print("alg: %d, %d start."%(_i, _j))
    np.random.seed(seed)
    results = np.zeros(n_repeat)
    p_op = p_ops[p_op_idx[0]] if isinstance(p_op_idx, list) else p_ops[p_op_idx]
    for i in range(n_repeat):
        p = p_op(p_op_idx[1], n_action) if isinstance(p_op_idx, list) else p_op(n_action)
        env = Env(n_action, p)
        samples = np.zeros((n_action, n_sample))
        for j in range(n_action):
            samples[j] = env.sample(j, n_sample)
        real = np.max(p)

        if alg_idx in [3]:
            results[i] = alg_ops[alg_idx](samples, n_action, K, r) - real
        else:
            results[i] = alg_ops[alg_idx](samples) - real

    bias = np.mean(results)
    bias2 = bias ** 2
    variance = np.var(results)
    mse = bias2 + variance
    print("alg: %d, %d ok."%(_i, _j))
    pipe.send([ _i, _j , np.array([bias2, variance, mse, bias])])

def mp(args, n_core, results):
    n = len(args)
    pipes = [Pipe() for _ in range(n_core)]
    pool = multiprocessing.Pool(processes=n_core)
    for i in range(0, n, n_core):
        print("start %d"%(i))
        ps = []
        seeds = np.random.randint(1, 10000, len(args[i: i + n_core]))
        for j, arg in enumerate(args[i: i+n_core]):
            ps.append(pool.apply_async(f, args=arg + [pipes[j][1]] + [seeds[j]]))
        [p.get() for p in ps]

        for pipe in pipes[:len(args[i: i + n_core])]:
            m, n, data = pipe[0].recv()
            results[m][n] = data
    return results

def setting1():
    n_action = 30
    n_samples = 1000 * (np.arange(10) + 1)
    n_repeat = 2000
    p_op_idx = 0
    alg_idxs = [alg_id]
    n_core = 10
    results = np.zeros([len(alg_ops), n_samples.size, 4])
    args = [[n_action, p_op_idx, n_sample, n_repeat, alg_idx, i, j, None] for j, n_sample in enumerate(n_samples) for i, alg_idx in enumerate(alg_idxs)]
    results = mp(args, n_core, results)
    save_data(results, "./results/setting%d_alg%d_id%d.pth"%(1, alg_idxs[0], id))

def setting2():
    n_actions = 10 * (np.arange(10) + 1)
    n_sample = 10000
    n_repeat = 2000
    p_op_idx = 0
    alg_idxs = [alg_id]
    n_core = 10
    results = np.zeros([len(alg_ops), n_actions.size, 4])
    args = [[n_action, p_op_idx, n_sample, n_repeat, alg_idx, i, j, None] for j, n_action in enumerate(n_actions) for i, alg_idx in enumerate(alg_idxs)]
    results = mp(args, n_core, results)
    save_data(results, "./results/setting%d_alg%d_id%d.pth"%(2, alg_idxs[0], id))

def setting3():
    n_action = 30
    n_sample = 10000
    n_repeat = 2000
    p_op_idx = 1
    p_num = 9
    n_core = 10
    alg_idxs = [alg_id]
    results = np.zeros([len(alg_ops), p_num, 4])
    args = [[n_action, [p_op_idx, p_idx], n_sample, n_repeat, alg_idx, i, j, None] for j, p_idx in enumerate(range(p_num)) for i, alg_idx in enumerate(alg_idxs)]
    results = mp(args, n_core, results)
    save_data(results, "./results/setting%d_alg%d_id%d.pth"%(3, alg_idxs[0], id))

def main(setting_idx):
    ops = [setting1, setting2, setting3]
    ops[setting_idx]()

if __name__ == "__main__":
    for idx in range(n_runs):
        id = idx + 1
        for setting_idx in [0, 1, 2]:
            for _alg_id in [0, 1, 2, 3]:
                np.random.seed(idx * 10)
                alg_id = _alg_id
                main(setting_idx)

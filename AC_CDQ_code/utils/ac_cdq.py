from utils.step import step
from utils.commons import save_data
from utils.play_mp import play_mp
import multiprocessing as mp, numpy as np, pdb

def _ac_cdq(opts, reward_array, repeat_n, seed, which_Q, pipe_send):
    np.random.seed(seed)

    Q = np.zeros((opts.n_state, opts.n_action, 2))
    Q_q = np.zeros((opts.n_state, opts.n_action))
    current_state = opts.start
    n_eps = np.zeros(opts.n_state)
    n_alpha = np.zeros((opts.n_state, opts.n_action, 2))

    n_alpha_q = np.zeros((opts.n_state, opts.n_action))

    rewards = np.zeros(opts.n_step)
    max_Q0 = np.zeros(opts.n_step)

    for i in range(opts.n_step):
        idx_Q = which_Q[i]
        n_eps[current_state] += 1
        action, reward, next_state = step(opts, Q, current_state, n_eps, reward_array, i)
        n_alpha[current_state][action][idx_Q] += 1
        n_alpha_q[current_state][action] += 1

        if current_state == opts.goal:
            delta = reward - Q[current_state][action][idx_Q]
        else:
            idxs = Q[next_state, :, 1 - idx_Q].argsort()[-max(int(opts.n_action / 2), opts.K): ]
            max_action = idxs[np.argmax(Q[next_state, idxs, idx_Q])]
            delta = reward + opts.gamma * np.minimum(np.max(Q[next_state, max_action, 1 - idx_Q]), np.max(Q[next_state, :, idx_Q])) - Q[current_state][action][idx_Q]

        Q[current_state][action][idx_Q] = Q[current_state][action][idx_Q] + (1 / np.power(n_alpha[current_state][action][idx_Q], opts.exp)) * delta

        rewards[i] = reward
        max_Q0[i] = np.max(np.mean(Q[opts.start], 1), 0)
        current_state = next_state

    pipe_send.send([repeat_n, rewards, max_Q0])

def ac_cdq(opts, reward_array):
    pool = mp.Pool(processes=opts.n_core)
    which_Q = (np.random.rand(opts.n_repeat, opts.n_step) >= 0.5).astype(np.uint8)
    args = [[reward_array[i], i, np.random.choice(1000000), which_Q[i]] for i in range(opts.n_repeat)]
    rewards, max_Q0 = play_mp(opts, pool, args, _ac_cdq)
    save_data([np.mean(rewards, 0), np.mean(max_Q0, 0)], opts.save_path % "ac_cdq")
    pool.close()

from utils.step import step
from utils.commons import save_data
from utils.play_mp import play_mp
import multiprocessing as mp, numpy as np

def _q(opts, reward_array, repeat_n, seed, pipe_send):

    np.random.seed(seed)
    Q = np.zeros((opts.n_state, opts.n_action))
    current_state = opts.start
    n_eps = np.zeros(opts.n_state)
    n_alpha = np.zeros((opts.n_state, opts.n_action))
    rewards = np.zeros(opts.n_step)
    max_Q0 = np.zeros(opts.n_step)
    for i in range(opts.n_step):
        n_eps[current_state] += 1
        action, reward, next_state = step(opts, Q, current_state, n_eps, reward_array, i)
        n_alpha[current_state][action] += 1
        if current_state == opts.goal:
            delta = reward - Q[current_state][action]
        else:
            delta = reward + opts.gamma * np.max(Q[next_state]) - Q[current_state][action]

        Q[current_state][action] = Q[current_state][action] + (
                1 / np.power(n_alpha[current_state][action], opts.exp)) * delta
        rewards[i] = reward
        max_Q0[i] = np.max(Q[opts.start], 0)
        current_state = next_state

    pipe_send.send([repeat_n, rewards, max_Q0])

def q(opts, reward_array):
    pool = mp.Pool(processes=opts.n_core)
    args = [[reward_array[i], i, np.random.choice(1000000)] for i in range(opts.n_repeat)]
    rewards, max_Q0 = play_mp(opts, pool, args, _q)
    save_data([rewards, max_Q0], opts.save_path % "q")
    pool.close()


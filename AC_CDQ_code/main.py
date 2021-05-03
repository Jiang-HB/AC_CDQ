import numpy as np
from utils import opts, q, dq, cdq, ac_cdq

def main():
    assert opts.setting in [1, 2, 3], "Unknown Reward Setting Type."

    # reward setting
    reward_array = np.zeros((opts.n_repeat, opts.n_step))
    random_array = np.random.rand(opts.n_repeat, opts.n_step)
    reward_array[np.where(random_array >= 0.5)] = -6
    reward_array[np.where(random_array < 0.5)] = 4

    # run algorithms
    q(opts, reward_array)       # q-learning
    dq(opts, reward_array)      # double q-learning
    cdq(opts, reward_array)     # clipped double q-learning
    ac_cdq(opts, reward_array)  # action candidate based clipped double q-learning

if __name__ == '__main__':
    main()
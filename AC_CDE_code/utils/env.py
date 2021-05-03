import numpy as np

class Env:
    def __init__(self, n_action, p):
        self.n_action = n_action
        self.p = p

    def sample(self, action, n_sample):
        return (np.random.uniform(0, 1, n_sample) < self.p[action]).astype(np.uint8)

def G1():
    def _op(n_action):
        p = np.random.uniform(0.02, 0.05, n_action)
        return p
    return _op

def G2():
    def _op(idx, n_action):
        p_upper = 0.02 + idx * 0.01
        p = np.random.uniform(0.02, p_upper, n_action)
        return p
    return _op

def G3():
    def _op(n_action):
        p = np.zeros(n_action) + 0.05
        return p
    return _op
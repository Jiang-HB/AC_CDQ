import numpy as np, heapq

def ME():
    def _op(samples):
        _mus = np.mean(samples, 1)
        return np.max(_mus)
    return _op

def DE():
    def _op(samples):
        n_sample = samples.shape[1]
        _mus_A = np.mean((samples[:, :int(n_sample/2)]), 1)
        _mus_B = np.mean((samples[:, int(n_sample/2):]), 1)
        estimator = (_mus_B[np.argmax(_mus_A)] + _mus_A[np.argmax(_mus_B)])/2
        return estimator
    return _op

def CDE():
    def _op(samples):
        n_sample = samples.shape[1]
        clip = np.max(np.mean(samples, 1))
        _mus_A = np.mean((samples[:, :int(n_sample/2)]), 1)
        _mus_B = np.mean((samples[:, int(n_sample/2):]), 1)
        estimator = (np.minimum(_mus_B[np.argmax(_mus_A)], clip) + np.minimum(_mus_A[np.argmax(_mus_B)], clip)) / 2
        return estimator
    return _op

def AC_CDE():
    def _op(samples, n_action, K, r):
        n_sample = samples.shape[1]
        _mus_A = np.mean((samples[:, :int(n_sample/2)]), 1)
        _mus_B = np.mean((samples[:, int(n_sample/2):]), 1)

        KK = K if K is not None else int(n_action * r)

        idxs = [x[0] for x in heapq.nlargest(KK, enumerate(_mus_A), key=lambda x: x[1])]
        max_actionA = idxs[np.argmax(_mus_B[idxs])]

        idxs = [x[0] for x in heapq.nlargest(KK, enumerate(_mus_B), key=lambda x: x[1])]
        max_actionB = idxs[np.argmax(_mus_A[idxs])]

        estimator = (_mus_A[max_actionA] + _mus_B[max_actionB]) / 2

        _mus = np.mean(samples, 1)
        estimator_q =  np.max(_mus)

        estimator = np.minimum(estimator_q, estimator)

        return estimator

    return _op
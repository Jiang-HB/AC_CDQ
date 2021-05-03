import numpy as np

def step(opts, Q, current_state, n_eps, reward_array, step_n, probs=None):

    if probs is not None:
        cums = np.cumsum(probs)
        probs = cums / np.max(cums)
        action = np.random.choice(np.where(np.random.rand() <= probs)[0])

    elif opts.policy == "eps":
        if np.random.rand() > 1 / np.sqrt(n_eps[current_state]):
            if len(Q.shape) == 3:
                action = np.argmax(np.mean(Q, 2)[current_state])
            if len(Q.shape) == 2:
                action = np.argmax(Q[current_state])
        else:
            action = np.random.randint(0, 4, 1)[0]

    if current_state != opts.goal:
        if action == 0:
            next_state = current_state - opts.n_col
            if next_state < 0:
                next_state = current_state
        elif action == 1:
            next_state = current_state + opts.n_col
            if next_state >= opts.n_state:
                next_state = current_state
        elif action == 2:
            next_state = current_state - 1
            if (next_state + 1) % opts.n_col == 0:
                next_state = current_state
        elif action == 3:
            next_state = current_state + 1
            if next_state % opts.n_col == 0:
                next_state = current_state
        reward = reward_array[step_n]
    else:
        reward = np.random.choice([-30, 40])
        next_state = opts.start

    return action, reward, next_state











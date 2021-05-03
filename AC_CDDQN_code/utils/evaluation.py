import numpy as np
from minatar import Environment
from utils.commons import get_state

def max_q(policy, state):
    return policy(state).max(1)[0].item()

def evaluation(opts, policy, timesteps):
    scores = []
    max_qs, real_qs = [], []
    env = Environment(opts.game, random_seed=10)
    for seed in range(opts.n_eval_episodes):
        env.reset()
        state = get_state(env.state())
        score, done = 0., False
        max_qs.append(max_q(policy, state))
        discount_score = 0.
        while not done:
            action = policy(state).max(1)[1].view(1, 1)
            reward, done = env.act(action)
            score += reward
            discount_score = opts.gamma * discount_score + reward
            state = get_state(env.state())
        scores.append(score)
        real_qs.append(discount_score)

    print("timesteps %d, mean score %.4f, mean max_q %.4f, real_q %.4f" % (timesteps, np.mean(scores), np.mean(max_qs), np.mean(real_qs)))

    return np.asarray(scores), np.asarray(max_qs), np.asarray(real_qs)


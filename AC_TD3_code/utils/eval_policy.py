import gym, numpy as np

def eval_policy(opts, policy, idx, timesteps):

    env = gym.make(opts.env_nm)
    env.seed(opts.seed)
    avg_reward = 0.
    max_qs, real_qs = [], []
    for _ in range(opts.n_eval_episodes):
        state, done = env.reset(), False
        discount_score, t = 0., 0
        max_qs.append(policy.max_q(np.array(state)))
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            discount_score += (opts.discount ** t) * reward
            t += 1
        real_qs.append(discount_score)


    avg_reward /= opts.n_eval_episodes

    print("=== [%d / %d] Evaluation over %d, episodes: avg_reward: %.3f ===" % (timesteps, idx, opts.n_eval_episodes, avg_reward))

    return avg_reward, np.mean(max_qs), np.mean(real_qs)
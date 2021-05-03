import gym, numpy as np, pdb, torch, time
from utils import ReplayBuffer, eval_policy, Recoder
from AC_TD3.ac_td3 import AC_TD3 as Method

def run(opts, seed):

    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # recoder
    recoder = Recoder(opts.results_dir, seed)

    # env
    env = gym.make(opts.env_nm)
    env.seed(seed)
    opts.state_dim = env.observation_space.shape[0]
    opts.action_dim = env.action_space.shape[0]
    opts.max_action = float(env.action_space.high[0])

    opts.policy_noise = opts.policy_noise * opts.max_action
    opts.noise_clip = opts.noise_clip * opts.max_action

    # policy setting
    policy = Method(opts, seed)

    # replay buffer
    replay_buffer = ReplayBuffer(opts)

    state, done = env.reset(), False
    episode_reward, episode_timesteps, episode_num, eval_idx = 0, 0, 0, 0
    t1 = time.time()

    for t in range(int(opts.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < opts.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(np.array(state)) + np.random.normal(0, opts.max_action * opts.expl_noise, size=opts.action_dim)
                      ).clip(-opts.max_action, opts.max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= opts.start_timesteps:
            policy.train(replay_buffer)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print("- Seed: %d, Total T: %d, Episode Num: %d, Episode T: %d, Reward: %.3f, Time: %.2f, %s -" % (
                seed, t + 1, episode_num + 1, episode_timesteps + 1, episode_reward, time.time() - t1, opts.tag))
            recoder.add_result(episode_reward, "train_return")
            # recoder.add_result(episode_reward, "train_return")
            recoder.save_result()

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if opts.is_eval and (t + 1) % opts.eval_freq == 0:
            eval_scores, max_qs, real_qs = eval_policy(opts, policy, eval_idx + 1, t + 1)
            print("eval_scores", eval_scores, "max_qs", max_qs, "real_qs", real_qs)
            recoder.add_result({"eval_scores": eval_scores, "max_qs": max_qs, "real_qs": real_qs}, "test_return")
            recoder.save_result()

    if opts.save_model:
        policy.save()

    recoder.save_result()




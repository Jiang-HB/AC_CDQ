from utils import AttrDict

opts = AttrDict()
opts.alg_nm = "AC_TD3"
opts.env_nm = "Walker2d-v2"
opts.seed = 0

# train
opts.buffer_size = int(1e6)
opts.start_timesteps = 1e4
opts.max_timesteps = 1e6
opts.expl_noise = 0.1
opts.batch_size = 256
opts.discount = 0.99
opts.tau = 0.005
opts.policy_noise = 0.1
opts.noise_clip = 0.5
opts.policy_freq = 2

opts.std_dev_stop = 0.1
opts.ac_td3_std_dev = 0.05
opts.top_candidates = 32

# test
opts.is_eval = True
opts.eval_freq = 5e3
opts.n_eval_episodes = 10

opts.save_model = False
opts.tag = "%s_%s_k%d" % (opts.alg_nm, opts.env_nm, opts.top_candidates)
opts.results_dir = "./results/%s" % (opts.tag)
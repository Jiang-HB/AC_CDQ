from utils import AttrDict

opts = AttrDict()

# general setting
opts.gamma = 0.95
opts.n_repeat = 10000
opts.n_step = 10000
opts.noise_gamma = 10
opts.setting = 1
opts.policy = "eps"
opts.exp = 0.8
opts.n_core = 20

# env setting
opts.n_row = 3
opts.n_col = 3
opts.start = 6
opts.goal = 2

# opts.n_row = 4
# opts.n_col = 4
# opts.start = 12
# opts.goal = 3

# opts.n_row = 5
# opts.n_col = 5
# opts.start = 20
# opts.goal = 4

# opts.n_row = 6
# opts.n_col = 6
# opts.start = 30
# opts.goal = 5

opts.n_state = opts.n_row * opts.n_col
opts.n_action = 4
opts.K = 2

opts.tag = "r%dc%dset%dk%d" % (opts.n_row, opts.n_col, opts.setting, opts.K)
opts.save_path = "./results/%s_" + opts.tag + ".pth"

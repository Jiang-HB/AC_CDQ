from utils import AttrDict

opts = AttrDict()
opts.ids = [0, 1, 2, 3, 4]
opts.N = 2

opts.agent_nm = "AC_CDDQN"
opts.env_nm = "space_invaders"
opts.has_target = False

opts.batch_size = 32
opts.replay_buffer_size = 100000
opts.target_network_update_freq = 1000
opts.training_freq = 1
opts.num_frames = 5000000
opts.first_n_frames = 100000
opts.replay_start_size = 5000
opts.end_epsilon = 0.1
opts.step_size = 0.00025
opts.grad_momentum = 0.95
opts.squared_grad_momentum = 0.95
opts.min_squared_grad = 0.01
opts.discount = 0.99
opts.epsilon = 1.0

opts.feature_dim = 128
opts.hidden_layers = []
opts.gradient_clip = -1
opts.optim_nm = "RMSprop"
opts.step_size = 0.00025
opts.grad_momentum = 0.95
opts.squared_grad_momentum = 0.95
opts.min_squared_grad = 0.01

opts.n_eval_episodes = 20 #20
opts.eval_iterval = 25000

opts.alg_tag = "%s_K%d" % (opts.agent_nm, opts.N)
opts.store_intermediate_result = True
import torch, random, numpy as np, pickle

def get_state(s):
    return (torch.tensor(s).permute(2, 0, 1)).unsqueeze(0).float().cuda()

def to_numpy(t):
  return t.cpu().detach().numpy()

def world_dynamics(opts, t, s, env, policy_net1, policy_net2):
    # A uniform random policy is run before the learning starts
    if t < opts.replay_start_size:
        action = torch.tensor([[random.randrange(opts.num_actions)]]).cuda()
    else:
        epsilon = opts.end_epsilon if t - opts.replay_start_size >= opts.first_n_frames \
            else ((opts.end_epsilon - opts.epsilon) / opts.first_n_frames) * (t - opts.replay_start_size) + opts.epsilon

        if np.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(opts.num_actions)]]).cuda()
        else:
            with torch.no_grad():
                action = (policy_net1(s) + policy_net2(s)).max(1)[1].view(1, 1)

    reward, terminated = env.act(action)
    s_prime = get_state(env.state())

    # return s_prime.cpu(), action.cpu(), torch.tensor([[reward]]).float(), torch.tensor([[terminated]])
    return s_prime, action, torch.tensor([[reward]]).float().cuda(), torch.tensor([[terminated]]).cuda()

def load_data(path):
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def save_data(path, data):
    file = open(path, "wb")
    pickle.dump(data, file)
    file.close()
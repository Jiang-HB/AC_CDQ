import torch, numpy as np

class ReplayBuffer(object):
    def __init__(self, opts):

        self.state = np.zeros((opts.buffer_size, opts.state_dim))
        self.action = np.zeros((opts.buffer_size, opts.action_dim))
        self.next_state = np.zeros((opts.buffer_size, opts.state_dim))
        self.reward = np.zeros((opts.buffer_size, 1))
        self.not_done = np.zeros((opts.buffer_size, 1))

        self.ptr = 0
        self.size = 0
        self.opts = opts

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.opts.buffer_size
        self.size = min(self.size + 1, self.opts.buffer_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.opts.device),
            torch.FloatTensor(self.action[ind]).to(self.opts.device),
            torch.FloatTensor(self.next_state[ind]).to(self.opts.device),
            torch.FloatTensor(self.reward[ind]).to(self.opts.device),
            torch.FloatTensor(self.not_done[ind]).to(self.opts.device)
        )

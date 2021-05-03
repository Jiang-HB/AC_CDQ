import torch, torch.nn as nn, torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, opts):
        super(Actor, self).__init__()

        # A1
        self.l1 = nn.Linear(opts.state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, opts.action_dim)

        # A2
        self.l4 = nn.Linear(opts.state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, opts.action_dim)

        self.max_action = opts.max_action

    def A1(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def A2(self, state):
        a = F.relu(self.l4(state))
        a = F.relu(self.l5(a))
        return self.max_action * torch.tanh(self.l6(a))

    def forward(self, state):
        return self.A1(state)

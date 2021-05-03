from utils.base_dqn import BaseDQN
from utils.commons import to_numpy
import torch

class AC_CDDQN(BaseDQN):

    def __init__(self, opts):

        super().__init__(opts)
        self.target_network_update_freq = opts.target_network_update_freq
        self.Q_net = [self.creatNN(self.input_type).to(self.device) for _ in range(2)]
        self.optimizer = [torch.optim.RMSprop(self.Q_net[i].parameters(), lr=opts.step_size, alpha=opts.squared_grad_momentum,
                                        centered=True, eps=opts.min_squared_grad) for i in range(2)]

    def get_action_selection_q_values(self, state):
        q_values = self.Q_net[0](state) + self.Q_net[1](state)
        q_values = to_numpy(q_values).flatten()
        return q_values

    def learn(self):
        self.update_Q_net_index = 0
        super().learn()
        self.update_Q_net_index = 1
        super().learn()

    def compute_q_target(self, next_states, rewards, dones):

        with torch.no_grad():
            q_next = self.Q_net[1 - self.update_Q_net_index](next_states)
            N = self.opts.N
            idxs = q_next.argsort(1)[:, -N:]  # [B, N]
            tmp = torch.zeros(idxs.size()).cuda() # # [B, N]
            q_next_tmp = self.Q_net[self.update_Q_net_index](next_states)
            ords = torch.arange(len(q_next_tmp)).long()
            for i in range(idxs.size(1)):
                tmp[:, i] = q_next_tmp[ords, idxs[:, i]]
            best_actions = torch.argmax(tmp, dim=-1)
            best_actions = idxs[ords, best_actions]
            Q_s_prime_a_prime = q_next[ords, best_actions].view(-1)

            clip_Q = q_next.max(1)[0].view(-1)

            q_target = rewards + self.discount * torch.min(Q_s_prime_a_prime, clip_Q) * (1 - dones)

            return q_target
import copy, torch, torch.nn.functional as F, os, pdb
from AC_TD3.modules import Actor, Critic

class AC_TD3(object):

    def __init__(self, opts, seed):

        self.actor = Actor(opts).to(opts.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(opts).to(opts.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.actor_save_path = os.path.join(opts.results_dir, "actor_seed%d.path" % seed)
        self.critic_save_path = os.path.join(opts.results_dir, "critic_seed%d.path" % seed)

        self.opts = opts
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.opts.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def max_q(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.opts.device)
        return (self.critic.Q1(state, self.actor.A1(state)) + self.critic.Q2(state, self.actor.A2(state))).item() / 2.

    def train(self, replay_buffer):

        self.total_it += 1

        # sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(self.opts.batch_size)
        next_states = next_state.unsqueeze(1).repeat(1, self.opts.top_candidates, 1).reshape(self.opts.batch_size * self.opts.top_candidates, self.opts.state_dim)
        with torch.no_grad():

            noise = (torch.randn_like(action) * self.opts.policy_noise).clamp(- self.opts.noise_clip, self.opts.noise_clip)
            next_actions = (self.actor_target.A2(next_state) + noise).clamp(-self.opts.max_action, self.opts.max_action)
            next_actions = next_actions.unsqueeze(1)
            next_actions = next_actions + self.opts.ac_td3_std_dev * torch.randn(self.opts.batch_size, self.opts.top_candidates, self.opts.action_dim).cuda()  # [B, TopK, D]
            next_actions = next_actions.reshape(self.opts.batch_size * self.opts.top_candidates, self.opts.action_dim)

            max_action_values_Q1 = self.critic_target.Q1(next_states, next_actions).view(self.opts.batch_size, self.opts.top_candidates)
            max_action_idxs_Q1 = (max_action_values_Q1.argmax(1) + self.opts.top_candidates * torch.arange(0, self.opts.batch_size, dtype=torch.int64).cuda()).view(-1)
            max_actions_Q1 = next_actions[max_action_idxs_Q1] # [B, D]
            target_Q1 = self.critic_target.Q2(next_state, max_actions_Q1)

            if self.opts.is_clip:
                clip_target_Q1 = self.critic_target.Q1(next_state, self.actor_target.A1(next_state))
                target_Q = reward + not_done * self.opts.discount * torch.min(target_Q1, clip_target_Q1)
            else:
                target_Q = reward + not_done * self.opts.discount * target_Q1

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.opts.policy_freq == 0:

            # Compute actor losse
            actor_loss = - self.critic.Q1(state, self.actor.A1(state)).mean() - self.critic.Q2(state, self.actor.A2(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.opts.tau * param.data + (1 - self.opts.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.opts.tau * param.data + (1 - self.opts.tau) * target_param.data)

    def save(self):
        torch.save(self.critic.state_dict(), self.critic_save_path)
        torch.save(self.actor.state_dict(), self.actor_save_path)

    def load(self):
        self.critic.load_state_dict(torch.load(self.critic_save_path))
        self.actor.load_state_dict(torch.load(self.actor_save_path))
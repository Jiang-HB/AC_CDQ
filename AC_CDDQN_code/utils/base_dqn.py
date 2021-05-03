from minatar import Environment
from utils.modules import Conv2d_MinAtar, MLP, NetworkGlue
from utils.replay import Replay
from utils.commons import to_numpy, get_state
import torch.nn as nn, torch, numpy as np, torch.optim as optim, torch.nn.functional as f, os, time, random
from utils.recorder import Recoder

class BaseDQN:

    def __init__(self, opts):

        self.opts = opts
        self.env_name = opts.env_nm
        self.agent_name = opts.agent_nm
        self.max_episode_steps = opts.num_frames
        self.device = opts.device
        self.batch_size = opts.batch_size
        self.discount = opts.discount
        self.gradient_clip = opts.gradient_clip

        # env
        self.env = Environment(self.env_name)
        self.action_size = self.env.num_actions()
        self.state_shape = self.env.state_shape()
        self.state_size = self.get_state_size()
        self.history_length = self.state_shape[2]

        # network
        self.input_type = opts.input_type
        self.layer_dims = [opts.feature_dim] + opts.hidden_layers + [self.action_size]
        self.Q_net = [None]
        self.Q_net[0] = self.creatNN(self.input_type).to(self.device)

        # optimizer
        self.optimizer = [None]
        self.optimizer[0] = optim.RMSprop(self.Q_net[0].parameters(), lr=opts.step_size, alpha=opts.squared_grad_momentum, centered=True, eps=opts.min_squared_grad)

        # normalizer
        self.state_normalizer = lambda x: x
        self.reward_normalizer = lambda x: x

        # replay buffer
        self.replay_buffer = Replay(opts.replay_buffer_size, self.batch_size, self.device)

        # update
        self.loss = f.smooth_l1_loss
        self.update_Q_net_index = 0
        self.sgd_update_frequency = opts.training_freq

    def get_state_size(self):
        return int(np.prod(self.state_shape))

    def creatNN(self, input_type):
        feature_net = Conv2d_MinAtar(in_channels=self.history_length, feature_dim=self.layer_dims[0])
        value_net = MLP(layer_dims=self.layer_dims, hidden_activation=nn.ReLU())
        NN = NetworkGlue(feature_net, value_net)
        return NN

    def comput_q(self, states, actions):
        actions = actions.long()
        q = self.Q_net[self.update_Q_net_index](states).gather(1, actions).squeeze()
        return q

    def compute_q_target(self, next_states, rewards, dones):
        q_next = self.Q_net[0](next_states).detach().max(1)[0]
        q_target = rewards + self.discount * q_next * (1 - dones)
        return q_target

    def learn(self):
        states, actions, next_states, rewards, dones = self.replay_buffer.sample()

        # Compute q target
        q_target = self.compute_q_target(next_states, rewards, dones)
        # Compute q
        q = self.comput_q(states, actions)
        # Take an optimization step
        loss = self.loss(q, q_target)
        self.optimizer[self.update_Q_net_index].zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.Q_net[self.update_Q_net_index].parameters(), self.gradient_clip)
        self.optimizer[self.update_Q_net_index].step()

    def save_experience(self, state, action, next_state, reward, done):
        # Saves recent experience to replay buffer
        experience = [state, action, next_state, reward, done]
        self.replay_buffer.add([experience])

    def get_action_selection_q_values(self, state):
        q_values = self.Q_net[0](state)
        q_values = to_numpy(q_values).flatten()
        return q_values

    def get_action(self, state, is_test=False):

        if not is_test:
            if self.step_count < self.opts.replay_start_size:
                action = random.randrange(self.action_size)
            else:
                epsilon = self.opts.end_epsilon if self.step_count - self.opts.replay_start_size >= self.opts.first_n_frames \
                    else ((self.opts.end_epsilon - self.opts.epsilon) / self.opts.first_n_frames) * (self.step_count - self.opts.replay_start_size) + self.opts.epsilon

                if np.random.binomial(1, epsilon) == 1:
                    action = random.randrange(self.action_size)
                else:
                    with torch.no_grad():
                        q_values = self.get_action_selection_q_values(state)
                        action = np.argmax(q_values)
        else:
            with torch.no_grad():
                q_values = self.get_action_selection_q_values(state)
                action = np.argmax(q_values)

        return action

    def evaluation(self):

        def max_q(state):
            return self.Q_net[0](state).max(1)[0].item()

        scores = []
        max_qs, real_qs = [], []
        env = Environment(self.opts.env_nm)
        for seed in range(self.opts.n_eval_episodes):
            # env = Environment(self.opts.env_nm, random_seed=10*seed)
            env.reset()
            state = get_state(env.state())
            score, done = 0., False
            max_qs.append(max_q(state))
            discount_score, t = 0., 0
            while not done:
                action = self.get_action(state, is_test=True)
                reward, done = env.act(action)
                reward = reward.item() if not isinstance(reward, int) else reward
                score += reward
                discount_score += (self.opts.discount ** t) * reward
                t += 1
                state = get_state(env.state())
            scores.append(score)
            real_qs.append(discount_score)

        print("timesteps %d, mean score %.4f, mean max_q %.4f, real_q %.4f" % (
        self.step_count, np.mean(scores), np.mean(max_qs), np.mean(real_qs)))

        return np.asarray(scores), np.asarray(max_qs), np.asarray(real_qs)

    def run_steps(self):
        # Set initial values
        data_return, frame_stamp, avg_return = [], [], 0.
        t_start = time.time()
        self.step_count, self.episode_count, self.policy_net_update_counter = 0, 0, 0
        recoder = Recoder(self.opts.save_dir, seed=0)

        while self.step_count < self.opts.num_frames:
            print("%d / %d: %.4f %s" % (self.step_count, self.opts.num_frames, self.step_count / self.opts.num_frames, self.opts.tag))
            G = 0.0
            self.env.reset()
            state = self.state_normalizer(self.env.state())
            done = False

            while (not done) and self.step_count < self.opts.num_frames:

                action = self.get_action(get_state(state))
                reward, done = self.env.act(action)
                next_state = self.state_normalizer(self.env.state())
                reward = self.reward_normalizer(reward)
                # reward = reward.item() if not isinstance(reward, int) else reward

                self.save_experience(state.transpose(2, 0, 1), action, next_state.transpose(2, 0, 1), reward, done)

                if self.step_count > self.opts.replay_start_size and self.step_count % self.sgd_update_frequency == 0:
                    self.policy_net_update_counter += 1
                    self.learn()

                if self.step_count % self.opts.eval_iterval == 0:
                    eval_scores, max_qs, real_qs = self.evaluation()
                    recoder.add_result({"eval_scores": eval_scores, "max_qs": max_qs, "real_qs": real_qs}, "test_return")
                    recoder.save_result()

                G += reward
                self.step_count += 1
                state = next_state

            self.episode_count += 1
            data_return.append(G)
            frame_stamp.append(self.step_count)

            avg_return = 0.99 * avg_return + 0.01 * G
            if self.episode_count % 50 == 0:
                print("Episode " + str(self.episode_count) + " | Return: " + str(G) + " | Avg return: " +
                             str(np.around(avg_return, 2)) + " | Frame: " + str(self.step_count) + " | Time per frame: " + str(
                    (time.time() - t_start) / self.step_count))

            # Save model data and other intermediate data if the corresponding flag is true
            if self.opts.store_intermediate_result and self.episode_count % 50 == 0:
                torch.save({
                    'episode': self.episode_count,
                    'frame': self.step_count,
                    'policy_net_update_counter': self.policy_net_update_counter,
                    'avg_return': avg_return,
                    'return_per_run': data_return,
                    'frame_stamp_per_run': frame_stamp,
                    'replay_buffer': []
                }, os.path.join(self.opts.save_dir, "checkpoint.pth"))
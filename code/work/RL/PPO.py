# libraries
import os
from typing import Any
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from pathlib import Path


class PPOMemory:
    def __init__(self, batch_size):
        self.states, self.actions, self.vals = [], [], []
        self.rewards, self.probs, self.dones = [], [], []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), \
            np.array(self.probs), np.array(self.vals), \
            np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = [] 


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc_dims, chkpt_dir=Path('tmp/ppo')):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = chkpt_dir / "actor_torch_ppo"

        layers, prev = [], input_dims
        for dim in fc_dims:
            layers += [nn.Linear(prev, dim), nn.Tanh()]
            prev = dim
        layers.append(nn.Linear(prev, n_actions))

        self.actor_mean = nn.Sequential(*layers)
        self.actor_log_std = nn.Parameter(T.zeros(n_actions)-0.5)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        mu = self.actor_mean(state)
        std = self.actor_log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist

    def save_checkpoint(self):
        Path(self.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)   # create the folder if missing
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,
                     fc_dims, chkpt_dir=Path('tmp/ppo')):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = chkpt_dir / "critic_torch_ppo"

        layers, prev = [], input_dims
        for dim in fc_dims:
            layers += [nn.Linear(prev, dim), nn.Tanh()]
            prev = dim
        layers.append(nn.Linear(prev, 1))

        self.critic = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        Path(self.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)   # create the folder if missing
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, action_low, action_high,
                 gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10,
                 entropy_coef=0.01, actor_dims=(256, 256), critic_dims=(256, 256),
                 save_dir=Path.cwd()):
        """actor_dims / critic_dims: hidden layer sizes, one entry per layer,
        e.g. (64, 64) -> input -> 64 -> 64 -> output."""
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.save_dirs = save_dir

        self.actor = ActorNetwork(n_actions, input_dims, alpha, fc_dims=list(actor_dims), chkpt_dir=self.save_dirs)
        self.critic = CriticNetwork(input_dims, alpha, fc_dims=list(critic_dims), chkpt_dir=self.save_dirs)
        self.memory = PPOMemory(batch_size)

        self.action_low = T.tensor(action_low, dtype=T.float).to(self.actor.device)
        self.action_high = T.tensor(action_high, dtype=T.float).to(self.actor.device)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('saving models.....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('loading models.....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_actions(self, observations):
        """Sample an action for one observation.

        Returns (clipped_action, raw_action, log_prob, value):
          clipped_action  clamped to [action_low, action_high] -> send this to the environment
          raw_action      the unclipped sample, shape (n_actions,) -> store this with remember();
                          log_prob belongs to this sample, so storing the clipped one would
                          make PPO's probability ratio wrong whenever clipping happens
        """
        state = T.tensor(observations, dtype=T.float).unsqueeze(0).to(self.actor.device)

        with T.no_grad():
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()                                   # (1, n_actions)

            prob = dist.log_prob(action).sum(axis=-1).item()
            value = value.item()
            clipped = T.clamp(action, self.action_low, self.action_high)

        # squeeze only the batch dimension, so the action keeps shape (n_actions,)
        raw_action = action.squeeze(0).cpu().numpy()
        clipped_action = clipped.squeeze(0).cpu().numpy()

        return clipped_action, raw_action, prob, value

    def compute_gae(self, rewards, values, dones, last_value):
        adv, gae = np.zeros(len(rewards), dtype=np.float32), 0.0
        for t in reversed(range(len(rewards))):
            next_v = last_value if t == len(rewards) - 1 else values[t + 1]
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * nonterm - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterm * gae    # resets at an episode end
            adv[t] = gae
        return adv

    def learn(self, last_value):
        
        state_arr, action_arr, old_prob_arr, vals_arr, \
        reward_arr, dones_arr, _ = self.memory.generate_batches()

        advantage = self.compute_gae(reward_arr, vals_arr, dones_arr.astype(np.float32), last_value)

        advantage = T.tensor(advantage, dtype=T.float).to(self.actor.device)
        values = T.tensor(vals_arr, dtype=T.float).to(self.actor.device)

        for _ in range(self.n_epochs):
            _, _, _, _, _, _, batches = self.memory.generate_batches()   # fresh shuffle each epoch

            for batch in batches:
                states = T.tensor(state_arr[batch],
                                  dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(action_arr[batch],
                                   dtype=T.float).to(self.actor.device)
                actions = actions.reshape(len(batch), -1)       # (batch, n_actions), matches dist
                dist = self.actor(states)
                critic_value = self.critic(states).squeeze(-1)  # (batch,), also for a batch of 1

                new_probs = dist.log_prob(actions).sum(axis=-1)
                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantage[batch]*prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip
                )*advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                entropy_bonus = dist.entropy().sum(axis=-1).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss - self.entropy_coef*entropy_bonus

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()




    
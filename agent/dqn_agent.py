import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from .per_buffer import PrioritizedReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=100_000,
        target_update_freq=500,
        use_per=True
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_per = use_per

        self.learn_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay Buffer
        if use_per:
            self.memory = PrioritizedReplayBuffer(memory_size)
            self.beta = 0.4
            self.beta_increment = 0.001
        else:
            self.memory = deque(maxlen=memory_size)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        if self.use_per:
            self.memory.add((state, action, reward, next_state, done))
        else:
            self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        if self.use_per:
            batch, indices, weights = self.memory.sample(
                self.batch_size, beta=self.beta
            )
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = np.ones(len(batch))

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q = self.model(states).gather(1, actions)

        # DOUBLE DQN
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        if self.use_per:
            td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
            loss = (weights * (current_q - target_q) ** 2).mean()
        else:
            loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Atualizar prioridades se usar PER
        if self.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors.flatten())

        # Atualizar target network
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.update_target_network()

        # Decaimento epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.learn_step = checkpoint.get('learn_step', 0)

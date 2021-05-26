import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

class ConvDuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, env):
        super(ConvDuelingDQN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # NN is divided into state-values (self.values) and state-dependent action advantages (self.advantages)
        self.values = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantages = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, state):
        # Forward pass convolutional layers
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        # Forward pass state-values
        values = self.values(features)
        # Forward pass state-dependent action advantages
        advantages = self.advantages(features)
        # Merge values and advantages to get Q values
        Q_values = values + (advantages - advantages.mean())

        return Q_values

    def feature_size(self):

        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

    def get_action(self, state, epsilon):
        if np.random.randn() > epsilon:
            return self.env.action_space.sample()

        else:
            state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
            q_values = self.forward(state).to(self.device)
            action = q_values.max(1)[1].item()

            return action
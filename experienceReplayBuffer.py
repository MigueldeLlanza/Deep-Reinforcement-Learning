from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_capacity):
        """
        Buffer with limit capacity
        """
        self.buffer = deque(maxlen=max_capacity)

    def add_past_exp(self, state, action, reward, next_state, done):
        """
        Add past experiences to the buffer
        """
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample_past_exp(self, batch_size):
        """
        Sample from the past experiences
        """
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

"""
"""
import copy
import torch
import random
import numpy as np
from collections import namedtuple, deque

import config
random.seed(config.SEED)

device = torch.device(config.device)

class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=config.BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        if len(self.memory) <= config.BATCH_SIZE:
            return None

        experiences = random.sample(self.memory, k=config.BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class OUNoise:
    def __init__(self, action_dims, mu=0., theta=0.15, sigma=0.2):
        self.action_dims = action_dims        
        self.mu = mu * np.ones(action_dims)
        self.theta = theta
        self.sigma = sigma        
        
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.action_dims)
        self.state = x + dx
        
        return self.state
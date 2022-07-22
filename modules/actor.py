"""
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
from network.actor import ActorNN

device = torch.device(config.device)

class Actor:

    def __init__(self, input_dims, action_dims, memory, noise):   
        self.local = ActorNN(input_dims, action_dims).to(device)
        self.target = ActorNN(input_dims, action_dims).to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=config.LR_ACTOR)

        self.noise = noise
        self.memory = memory

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.local.eval()
        with torch.no_grad():
            action = self.local(state).cpu().data.numpy()
        self.local.train()        
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
        
    def reset(self):
        self.noise.reset()
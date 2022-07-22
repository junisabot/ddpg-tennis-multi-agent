"""
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
from network.critic import CriticNN

device = torch.device(config.device)

class Critic:

    def __init__(self, input_dims, action_dims):   
        self.local = CriticNN(input_dims, action_dims).to(device)
        self.target = CriticNN(input_dims, action_dims).to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=config.LR_CRITIC)

    def step(self, actor, memory):
        experiences = memory.sample()        
        if not experiences:
            return
        self.learn(actor, experiences)

    def learn(self, actor, experiences):
        states, actions, rewards, next_states, dones = experiences

        actions_next = actor.target(next_states)
        Q_targets_next = self.target(next_states, actions_next)
        Q_targets = rewards + (config.GAMMA * Q_targets_next * (1 - dones))

        Q_expected = self.local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        actions_pred = actor.local(states)
        actor_loss = - self.local(states, actions_pred).mean()

        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()

        self.model_update(self.local, self.target)
        self.model_update(actor.local, actor.target)

    def model_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(config.TAU*local_param.data + (1.0-config.TAU)*target_param.data)
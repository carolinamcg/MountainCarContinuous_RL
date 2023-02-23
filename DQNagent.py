import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# https://arxiv.org/pdf/1509.02971.pdf
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQNAC(nn.Module):  # DQN actor critic
    def __init__(self, n_observations):
        super(DQNAC, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.activ = nn.ELU()
        self.actor = nn.Linear(128, 1)
        self.actor_activ = nn.Tanh()
        self.critic = nn.Linear(128, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.activ((self.layer1(x)))
        x = self.activ((self.layer2(x)))
        return self.actor_activ(self.actor(x)), self.critic(x)  # action, value

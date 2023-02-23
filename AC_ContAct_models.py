import numpy as np

import torch
import torch.nn as nn
#from torch.distributions.normal import Normal
from trunc_normal_dist import trunc_normal_

class Actor(nn.Module):
    def __init__(self, nb_states, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        #self.BN1 = nn.InstanceNorm1d(hidden1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        #self.BN2 = nn.InstanceNorm1d(hidden2)

        self.mu = nn.Linear(hidden2, 1)
        self.sigma = nn.Linear(hidden2, 1)
        #self.elu = nn.ELU()
        self.tanh = nn.Tanh()
    
    ''' 
    def sampler(self, mu, sigma):
        policy = Normal(mu, sigma) #USE TRUNCATED DIST ?????????????????????????
        action = policy.rsample()
        action = self.tanh()
        log_prob = policy.log_prob(action)

        return action, log_prob
    '''
    def sampler(self, mu, sigma): #select action
        action = torch.empty(1)
        action, prob = trunc_normal_(action, mean=mu, std=sigma, a=-1, b=1)
        return action, prob

    def forward(self, x, eps=1e-6):
        out = self.tanh(self.fc1(x))
        out = self.tanh(self.fc2(out))

        mu_, sigma_ = self.mu(out), self.sigma(out)
        #mu_ = self.tanh(mu_) #mu between -1 and 1, cause that's the action range
        #sigma = self.elu(self.sigma(out)) + 1 #has to be non-negative
        sigma_ =  torch.exp(sigma_ + eps)

        action, prob = self.sampler(mu_, sigma_)

        return action, prob


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        #self.BN1 = nn.InstanceNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        #self.BN2 = nn.InstanceNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ELU()
        #self.init_weights(init_w)

    def forward(self, x):
        #the critic just predicts the value function (V(s))
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

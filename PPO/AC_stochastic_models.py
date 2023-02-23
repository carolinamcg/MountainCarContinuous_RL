import numpy as np
import math

import torch
import torch.nn as nn
#from torch.distributions.normal import Normal
from trunc_normal_dist import trunc_normal_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.relu = nn.ReLU()
    
    ''' 
    def sampler(self, mu, sigma):
        policy = Normal(mu, sigma) #USE TRUNCATED DIST ?????????????????????????
        action = policy.rsample()
        action = self.tanh()
        log_prob = policy.log_prob(action)

        return action, log_prob
    '''
    def forward(self, x, eps=1e-6): #select action
        mu, sigma = self.get_dist(x, eps=eps)
        if sigma <0 :
            print("wtd")
        action = torch.empty(mu.size()).to(device) #action_dim=1
        action, prob = trunc_normal_(action, mean=mu, std=sigma, a=-1, b=1)
        if prob.isnan():
            print(action)
        return action, torch.log(prob)

    def get_dist(self, x, eps=1e-6):
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        out = self.tanh(self.fc1(x))
        out = self.tanh(self.fc2(out))

        mu_, sigma_ = self.mu(out), self.sigma(out)
        #mu_ = self.tanh(mu_) #mu between -1 and 1, cause that's the action range
        #sigma = self.elu(self.sigma(out)) + 1 #has to be non-negative
        sigma_ =  torch.exp(sigma_ + eps)

        if mu_.isnan().any() or sigma_.isnan().any():
            print("wtf")

        return mu_, sigma_ #chould the model also predict the sigma, or shoul we set it??????????

    def norm_pdf(self, action, mu, sigma):
        # Computes standard normal probability density function
        return ( 1/ (sigma * math.sqrt(2*torch.pi)) ) * torch.exp( (-1/2) * ((action - mu) / sigma)**2 )
    
    def norm_pdf_entropy(self, sigma):
        #https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
        return (1/2)*torch.log(2*torch.pi*(sigma**2)) + (1/2)


class Critic(nn.Module):
    def __init__(self, nb_states, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        #self.BN1 = nn.InstanceNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        #self.BN2 = nn.InstanceNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ELU()
        #self.init_weights(init_w)

    def forward(self, x):
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        #the critic just predicts the value function (V(s))
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

import math
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
from collections import namedtuple

from AC_ContAct_replaymemorybuffer import ReplayMemory
from utils import soft_update

# from DQNagent import DQNAC
from PPO.AC_stochastic_models import Actor, Critic

random.seed(42)
torch.autograd.set_detect_anomaly(True)
Transition = namedtuple("Transition", ("state", "action", 'prob', "next_state", "reward"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
#https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
class Train(object):
    def __init__(
        self,
        n_observations,
        n_actions,
        env,
        LR=[10e-4, 10e-3],
        memory_length=100000,
        h1=400,
        h2=300,
        BATCH_SIZE=64,
        GAMMA=0.99,
        TAU=0.0001,
    ) -> None:

        self.actor = Actor(
            n_observations, hidden1=h1, hidden2=h2
        ).to(device)
        self.actor_target = Actor(
            n_observations, hidden1=h1, hidden2=h2
        ).to(device)
        self.actor_target.load_state_dict(
            self.actor.state_dict()
        )  # initialize both with the same weights

        self.critic = Critic(
            n_observations, n_actions, hidden1=h1, hidden2=h2
        ).to(device)
        self.critic_target = Critic(
            n_observations, n_actions, hidden1=h1, hidden2=h2
        ).to(device)
        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )  # initialize both with the same weights

        self.env = env
        self.action_size = n_actions
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=LR[0], amsgrad=True)
        self.critic_optim = optim.AdamW(
            self.critic.parameters(), lr=LR[1], amsgrad=True, weight_decay=0.01
        )
        self.criterion = nn.SmoothL1Loss()
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.memory = ReplayMemory(memory_length)
        assert memory_length >= BATCH_SIZE, "ERROR: memory length has to be >= batch_size"
        self.TAU = TAU

        self.episode_durations = []



    def optimize_model(self):
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(
            *zip(*transitions)
        )  # organizes transitions as a list of len=4 (state, action, nex_state, reward)
        # each element in the list contains 64 steps=batch_size

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        prob_batch = torch.cat(batch.prob)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.critic([state_batch, action_batch])

        # Compute Q(s_{t+1}, a_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" critic_target; selecting the Q value, when the "older" actor_target is selecting the actions
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros((self.BATCH_SIZE, 1), device=device)
        with torch.no_grad():
            actions = torch.clip(self.actor_target(non_final_next_states).rsample(), -1, 1)
            next_state_values[non_final_mask] = self.critic_target(
                [non_final_next_states, actions]
            )
        # Compute the expected Q values
        # Q(s_t,a_t) = r_t + gamma*Q(s', a'), where s' and a' are from t+1 to final state
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        # target Gt is also estimated but w/ another NN -> bootstrapping (?)
        # Semi-gradient TD(0) for estimating Q bootstraping)

        advantage = expected_state_action_values - state_action_values #compute advantage

        # Critic and Actor update
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        # Compute Huber loss
        Q_loss = self.criterion(state_action_values, expected_state_action_values) #Correct by the loss formula: (predictions, targets)
        #BUT: the VE is (targets-predictions)**2 (doesn't matter for the result, cause tehy use absolute value in the self.criterion)

        #Policy Loss
        policy_loss = -torch.sum(torch.log(prob_batch) * advantage.flatten())

        loss = Q_loss + policy_loss
        loss.backward() #retain_graph=True
        self.critic_optim.step()
        self.actor_optim.step()

        return Q_loss, policy_loss

    def train(self, num_episodes, max_steps=500, w_path="./weights"):
        max_ER, min_no_steps = -100000, max_steps
        results =  {"Episode": [], "No of steps": [], "Final position": [], "Reward Sum": [], "Q_loss": [], "Policy_loss": []}
        for i_episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            state = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(
                0
            )
            for t in count():
                #get policy dist and randomly sample the action from it
                action, prob = self.actor(state)
                #action = torch.clip(policy.rsample(), -1, 1) #sampling from gaussian and clipping 
                #rsample to be differntiable: https://stackoverflow.com/questions/72925722/strange-behavior-from-normal-log-prob-when-calculating-gradients-in-pytorch

                observation, reward, terminated, truncated, _ = self.env.step(
                    np.array([action.item()], dtype=np.float32)
                )
                # episode_reward += reward
                # reward = torch.tensor([[reward]], device=device)
                done = terminated or truncated

                if terminated:
                    reward = 0
                    next_state = None  # if terminated, the agent was successfull
                else:
                    reward = -1
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                episode_reward += reward
                reward = torch.tensor([[reward]], device=device)
                # Store the transition in memory
                self.memory.push(state, action, prob , next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if len(self.memory) >= self.BATCH_SIZE:
                    Q_loss, policy_loss = self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                soft_update(self.actor_target, self.actor, self.TAU)
                soft_update(self.critic_target, self.critic, self.TAU)

                # [optional] save intermideate model
                if (t + 1) % int(max_steps / 2) == 0:
                    self.save_model(w_path, t + 1, i_episode)

                if done:
                    self.episode_durations.append(t + 1)
                    
                    results_list = [i_episode, t+1, round(state[0,0].item(), 4) if state is not None else state, episode_reward, Q_loss.item(), policy_loss.item()]
                    for v, k in zip(results_list, results.keys()):
                        results[k].append(v)

                    print(
                        f"Episode {results_list[0]}: steps={results_list[1]}, pos_T={results_list[2]}, episode_reward={round(episode_reward, 4)}, Q_loss={round(Q_loss.item(), 4)}, P_loss={round(policy_loss.item(), 4)}"
                    )

                    # plot_durations()
                    if episode_reward > max_ER:
                        self.save_model(w_path, t + 1, i_episode)
                        max_ER = episode_reward
                    elif (t+1) < min_no_steps:
                        self.save_model(w_path, t + 1, i_episode)
                        min_no_steps = t+1
                    
                    df = pd.DataFrame(results) #, columns=results.keys())
                    df.to_excel(w_path + "/results.xlsx", index=True)

                    break

        print("Complete")
        # plot_durations(show_result=True)
        # plt.ioff()
        # plt.show()

    def save_model(self, output, step, episode):
        os.makedirs(output, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            "{}/actor_E{}_step{}.pkl".format(output, episode, step),
        )
        torch.save(
            self.critic.state_dict(),
            "{}/critic_E{}_step{}.pkl".format(output, episode, step),
        )

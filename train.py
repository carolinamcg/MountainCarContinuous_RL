import math
import random
import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
from collections import namedtuple

from replaymemorybuffer import ReplayMemory
from utils import soft_update, to_numpy, plot_TDE

# from DQNagent import DQNAC
from DDPG_models import Actor, Critic
from OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcess

#random.seed(42)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://github.com/ghliu/pytorch-ddpg/blob/e9db328ca70ef9daf7ab3d4b44975076ceddf088/ddpg.py
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
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
        init_w=3e-3,
        BATCH_SIZE=64,
        GAMMA=0.99,
        epsilon = 50000, #'linear decay of exploration policy'
        warmup_steps=100, #'time without training but only filling the replay memory'
        warmup_decay=200, #to decrease the warmup steps number at each episode
        noise_theta=0.15,
        noise_sigma=0.2,
        sigma_min=None, 
        n_steps_annealing=100,
        eval_episodes=30,
        #EPS_END=0.05,
        #EPS_START=0.9,
        #EPS_DECAY=1000, ##decay approaches linear function and decreases slower with the number of steps, as this number is higher. it takes more nsteps to reach the EPS_END
        TAU=0.0001,
    ) -> None:

        self.actor = Actor(
            n_observations, n_actions, hidden1=h1, hidden2=h2, init_w=init_w
        ).to(device)
        self.actor_target = Actor(
            n_observations, n_actions, hidden1=h1, hidden2=h2, init_w=init_w
        ).to(device)
        self.actor_target.load_state_dict(
            self.actor.state_dict()
        )  # initialize both with the same weights

        self.critic = Critic(
            n_observations, n_actions, hidden1=h1, hidden2=h2, init_w=init_w
        ).to(device)
        self.critic_target = Critic(
            n_observations, n_actions, hidden1=h1, hidden2=h2, init_w=init_w
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
        self.memory = ReplayMemory(memory_length)
        assert memory_length >= BATCH_SIZE, "ERROR: memory length has to be >= batch_size"
        self.noise = OrnsteinUhlenbeckProcess(theta=noise_theta, sigma=noise_sigma, size=n_actions, 
                                            sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)

        #hyperparameters
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.depsilon = 1.0 / epsilon
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True
        self.warmup = warmup_steps
        self.eval_episodes = eval_episodes

        #self.steps_done = 0
        #self.EPS_END = EPS_END
        #self.EPS_START = EPS_START
        #self.EPS_DECAY = EPS_DECAY
        self.end = -0.05
        self.start = warmup_steps
        self.warmup_decay = warmup_decay

        self.episode_durations = []
        self.episode=0
        self.save_path=None

        self.check={"Episode": [], "Gt": [], "Predicted Value": [], "Q_loss": [], "P_loss": []}

    def set_warmup(self):  # force policy to be exploratory at the beginning
        new_warmup = self.end + (self.start - self.end) * math.exp(
            -1.0 * len(self.episode_durations) / self.warmup_decay)
        self.warmup = new_warmup

    '''
    def select_action(
        self, state, previous_action
    ):  # force policy to be exploratory at the beginning
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:  # rare at the first steps, cause eps is high
            with torch.no_grad():
                self.actor.eval()
                return self.actor(state)
        else:
            if (
                previous_action is not None and sample < eps_threshold * 0.8
            ):  # sample being smaller will be more often in teh first steps
                # maintain direction of previous action (important in mountain car problem)
                return (previous_action / torch.abs(previous_action)) * torch.tensor(
                    [np.abs(self.env.action_space.sample())],
                    device=device,
                    dtype=torch.float32,
                )
            else:
                return torch.tensor(
                    [self.env.action_space.sample()], device=device, dtype=torch.float32
                )
    '''

    def optimize_model(self, terminated=False):
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
            next_state_values[non_final_mask] = self.critic_target(
                [non_final_next_states, self.actor_target(non_final_next_states)]
            )
        # Compute the expected Q values
        # Q(s_t,a_t) = r_t + gamma*Q(s', a'), where s' and a' are from t+1 to final state
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        # target Gt is also estimated but w/ another NN -> bootstrapping (?)

        # Critic update
        # Compute Huber loss
        Q_loss = self.criterion(state_action_values, expected_state_action_values)
        self.critic_optim.zero_grad()
        Q_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.train()
        self.actor_optim.zero_grad()
        policy_loss = self.critic([state_batch, self.actor(state_batch)])
        ''' 
        # DON'T KNOW IF THIS MAKES SENSE HERE, cause our states are not drawn from the policy (self.actor), but from the env
        # multiplying by these frequencis below is not equivalent to the expected value under our policy (see pg. 326 - sutton and barto)
        # THE METHOD USED HERE IS NOT ON-POLICY
        #compute on-policy dist for the batch
        #here, is the normalized number of times the state has been visited
        with torch.no_grad():
            policy_dist = {}
            for s in state_batch:
                k = (s[0].item(), s[1].item())
                if k not in policy_dist.keys():
                    policy_dist[k] = 1
                else:
                    policy_dist[k] += 1
        #wieght/average the Q values of (s, actor_actions) with these frequencies
        policy_loss = 0
        for i,s in enumerate(state_batch):
            k = (s[0].item(), s[1].item())
            policy_loss -= policy_values[i] * policy_dist[k]/state_batch.shape[0]
            #we are saying which states we care more about
            # as we're maximizing teh action-state value (minimizing the policy loss), for lower frequency/w for one state, 
            # more penalized will be that state. It will have to increase more its value than others more visited
        '''
        policy_loss = -policy_loss.mean()  # deterministic policy = p(a|s) = 1
        policy_loss.backward()
        self.actor_optim.step()

        if terminated:
            #PLOT STATE and ACTION vs TDE 
            TDE = expected_state_action_values - next_state_values
            plot_TDE(state_batch, action_batch, TDE, self.episode, self.save_path)
            del TDE

        if len([s for s in batch.next_state if s is None])>0:
            #print("INCLUDED terminal state in memory buffer")
            i = torch.where(non_final_mask==False)[0]
            self.check["Episode"].append(self.episode)
            self.check["Gt"].append(expected_state_action_values[i].cpu().numpy())
            self.check["Predicted Value"].append(state_action_values[i].detach().cpu().numpy())
            self.check["Q_loss"].append(Q_loss.item())
            self.check["P_loss"].append(policy_loss.item())
            #print(expected_state_action_values[i].cpu().numpy(), state_action_values[i].detach().cpu().numpy())
            #print(Q_loss.item(), policy_loss.item())
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        return Q_loss, policy_loss

    def soft_update(self):
        actor_target_params = self.actor_target.state_dict()
        critic_target_params = self.critic_target.state_dict()

        actor_params = self.actor.state_dict()
        critic_params = self.critic.state_dict()

        for key in actor_params:
            actor_target_params[key] = actor_params[key]*self.TAU + actor_target_params[key]*(1-self.TAU)
        for key in critic_params:
            critic_target_params[key] = critic_params[key]*self.TAU + critic_target_params[key]*(1-self.TAU)

        self.actor_target.load_state_dict(actor_target_params)
        self.critic_target.load_state_dict(critic_target_params)

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.action_size)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(s_t)
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.noise.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def train(self, num_episodes, max_steps=500, w_path="./weights"):
        self.save_path = w_path
        max_ER, min_no_steps = -100000, max_steps
        results =  {"Episode": [], "No of steps": [], "Final position": [], "Reward Sum": [], "Q_loss": [], "Policy_loss": []}
        for i_episode in range(num_episodes):
            
            if i_episode >= 999 and np.all(np.array(self.episode_durations[-1000:]) < max_steps):
                break

            obs = self.env.reset()
            episode_reward = 0
            #previous_action = None
            state = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(
                0
            )

            self.episode = i_episode

            if self.eval_episodes is not None and (i_episode+1)%self.eval_episodes==0:
                self.evaluation(state)
                self.is_training = True
            else:
                for t in count():

                    if t <= self.warmup:
                        action = self.random_action()
                    else:
                        action = self.select_action(state)

                    observation, reward, terminated, truncated, _ = self.env.step(
                        action
                    )
                    # episode_reward += reward
                    # reward = torch.tensor([[reward]], device=device)
                    done = terminated or truncated

                    if terminated:
                        #reward = 0
                        next_state = None  # if terminated, the agent was successfull
                    else:
                        #reward = -1
                        next_state = torch.tensor(
                            observation, dtype=torch.float32, device=device
                        ).unsqueeze(0)

                    episode_reward += reward
                    reward = torch.tensor([[reward]], device=device)
                    action = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                    # Store the transition in memory
                    if t > self.warmup: #the first random actions should not be used to train the model
                        self.memory.push(state, action, next_state, reward)

                    # Move to the next state
                    state = next_state
                    #previous_action = action

                    # Perform one step of the optimization (on the policy network)
                    if len(self.memory) >= self.BATCH_SIZE:
                        Q_loss, policy_loss = self.optimize_model(terminated=False)

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    #self.soft_update()
                    soft_update(self.actor_target, self.actor, self.TAU)
                    soft_update(self.critic_target, self.critic, self.TAU)

                    # [optional] save intermideate model
                    #if (t + 1) == int(max_steps / 2):
                    #    self.save_model(w_path, t + 1, i_episode)

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

                        df_chek = pd.DataFrame(self.check)
                        df_chek.to_excel(w_path + "/check.xlsx", index=True)

                        if self.warmup_decay is not None:
                            self.set_warmup()
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

    def evaluation(self, state):
        self.is_training = False
        episode_reward = 0
        path = self.save_path + f"/eval/Episode{self.episode}"
        os.makedirs(path, exist_ok=True)
        for t in count():
            action = self.select_action(state, decay_epsilon=False)
            observation, reward, terminated, truncated, _ = self.env.step(
                action
            )

            done = terminated or truncated
            if t%50 == 0 or terminated:
                print(observation[0], action[0], reward)
                # Render the env
                env_screen = self.env.render()
                cv2.imwrite(path+f"/{t}.png", env_screen)
                #cv2.imshow(f"S: {state}, A: {action}, R: {reward}, S': {observation}", env_screen)
                #cv2.waitKey(1000) #& 0xFF

            episode_reward += reward
            if done:
                print(f"**** EVAL EPISODE: episode_reward={episode_reward}")
                cv2.destroyAllWindows()
                break
            else:
                state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
import numpy as np
import cv2
import os
import pandas as pd
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

from AC_stochastic_models import Actor, Critic

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def to_numpy(var):
    return var.detach().cpu().numpy() if USE_CUDA else var.data.numpy()

Transition = namedtuple('Transition',
                        ('state', 'action', 'log_prob'))

torch.autograd.set_detect_anomaly(True)

#https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
#https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea581467

class PPO():
    def __init__(self, env, LR=[10e-4, 10e-3], h1=400, h2=300, max_timesteps_episode=999, 
                timesteps_per_batch=2997, gamma=0.99, epochs=100, clip=0.2, c_s=0.3, c_s_min=None, n_steps_annealing=250, save_path="./") -> None:
        self.env=env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.actor = Actor(
            self.obs_dim, hidden1=h1, hidden2=h2
        ).to(device)
        self.critic = Critic(
            self.obs_dim, hidden1=h1, hidden2=h2
        ).to(device)

        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=LR[0], amsgrad=True) #CHECK these parameters !!!!!!!!!!!!!!!
        self.critic_optim = optim.AdamW(
            self.critic.parameters(), lr=LR[1], amsgrad=True, weight_decay=0.01
        )

        self.timesteps_per_batch=timesteps_per_batch
        self.max_timesteps_episode=max_timesteps_episode
        self.gamma=gamma
        self.epochs=epochs
        self.clip=clip 
        self.criterion=nn.SmoothL1Loss()

        if c_s_min is not None:
            self.m = -float(c_s - c_s_min) / float(n_steps_annealing) #n_steps_annealing is the number of steps it takes to get to the c_s_min value
            self.c_s = c_s #weight for entropy/exploration maximization in loss
                    #the less c_s, the less the entropy has impact in the optimization process
            self.c_max = c_s
            self.c_s_min = c_s_min
        else:
            self.m = 0.
            self.c_s = c_s
            self.c_max = c_s
            self.sigma_min = c_s
        self.n_steps = 0
        
        self.iteration = 0
        self.save_path = save_path
        self.w_path = os.path.join(self.save_path, "weights")
        self.r_path = os.path.join(self.save_path, "results")
        self.results = {"Iter": [], "Duration": [], "Reward": []}

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(np.array(batch_rtgs), dtype=torch.float, device=device).unsqueeze(-1)
        return batch_rtgs


    def rollout(self):
        # Batch data
        #batch_obs = []             # batch observations
        #batch_acts = []            # batch actions
        #batch_log_probs = []       # log probs of each action
        samples = [] #Transition Tuple (s, a, log_prob)
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0 
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            obs = self.env.reset()
            obs = obs[0]
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            for ep_t in range(self.max_timesteps_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                #Sample actions and log_probs from pi_{theta_k}=behavior policy
                action, log_prob = self.actor(state) # Return the sampled action and the log prob of that action
                # Note that I'm calling detach() since the action and log_prob  
                # are tensors with computation graphs, so I want to get rid
                # of the graph and just convert the action to numpy array.
                # log prob as tensor is fine. Our computation graph will
                # start later down the line.
                obs, reward, terminated, truncated, _ = self.env.step(
                         to_numpy(action).squeeze(0)
                    )

                done = terminated or truncated
                # Collect reward, action, and log prob
                ep_rews.append(reward)
                #batch_log_probs.append(log_prob.detach()) 
                if log_prob.isnan():
                    print(action)
                samples.append(Transition(state, action.detach(), log_prob.detach()))

                state = torch.tensor(
                    obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                if done:
                    break
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews) 

        # Reshape data as tensors in the shape specified before returning
        batch = Transition(
            *zip(*samples)
        )
        batch_obs = torch.cat(batch.state) #all the observations per timestep for this batch
        batch_acts = torch.cat(batch.action) #shape: timesteps x action_dim
        batch_log_probs = torch.cat(batch.log_prob) #shape: timesteps x 1
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews) #list of all rewards per timesteps and per episode (list of lists)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def evaluate(self, batch_obs, batch_acts):
        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        V = self.critic(batch_obs)

        mus, sigmas = self.actor.get_dist(batch_obs)
        probs = self.actor.norm_pdf(batch_acts, mus, sigmas)

        if (probs < 0).any():
            print(probs)

        #entropies = self.actor.norm_pdf_entropy(sigmas)
        #entropies = (mus.var() + sigmas.var())/2
        #if entropies > 100000:
        #    print("wtf")
        #entropy = -torch.sum(probs*torch.log(probs))
        #WHY log probs? https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
        #https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
        return V, torch.log(probs) #, entropies


    def get_current_cs(self):
        self.c_s = max(self.c_s_min, self.m * float(self.n_steps) + self.c_max)
        self.n_steps += 1
        
    def learn(self, total_timesteps, eps=1e-10):
        t_so_far=0
        while t_so_far < total_timesteps:
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens) #sums the duration of each episode inside the batch

            if torch.isnan(batch_log_probs).any():
                print(batch_log_probs)

            #Calculate V_{phi_k}
            V,_ = self.evaluate(batch_obs, batch_acts)

            #Compute Advantages (per episode)
            Ak = batch_rtgs - V.detach() #difference between the actual state_value and the predict value 
                                        #by the critic, for each timestep in eachepisode in the batch
            #Normalize advantages
            Ak = (Ak - Ak.mean()) / (Ak.std() + eps)

            for epoch in range(self.epochs):
                #Calculate V_phi and log_prob(pi_theta(at | st))
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                #S=: the pi_theta (curr_log_probs) is updated at every iteration (the most recent policy NN)
                #the pi_{theta_k} (batch_log_probs) is the behavior policy, used to generate/sample data
                #at the first epoch, these two are the same NN (with the same w). But, along the epochs, the pi_theta is updated
                #the pi_{theta_k} at the next iteration/batch of episodes will be equal to the final 
                #pi_theta updated in this iteration (at lhe last epoch)
                ratios = torch.exp(curr_log_probs - batch_log_probs) #importance sample ratio
                surr1 = ratios * Ak
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*Ak
                #Losses
                #self.get_current_cs()
                actor_loss = (-torch.min(surr1, surr2)).mean() #- self.c_s*entropies#.mean()
                if actor_loss.isnan():
                    print(surr1, surr2)
                critic_loss = self.criterion(V, batch_rtgs)

                # Critic and Actor update
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optim.step()
                
                #print((-torch.min(surr1, surr2)).mean() , self.c_s, self.c_s*entropies.mean())
                print(f"Iter{self.iteration}, Epoch{epoch}, Actor_loss={actor_loss}, Critic_loss={critic_loss}") #C_s={self.c_s}
            
            self.save_model()
            self.eval_policy()
            self.iteration+=1

        df = pd.DataFrame(self.results)
        df.to_excel(self.r_path + "/eval_results.xlsx", index=True)


    def save_model(self):
        os.makedirs(self.w_path, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            "{}/actor_iter{}.pkl".format(self.w_path, self.iteration),
        )
        torch.save(
            self.critic.state_dict(),
            "{}/critic_iter{}.pkl".format(self.w_path, self.iteration),
        )

    def eval_policy(self):
        episode_reward = 0
        path = self.r_path + f"/Performance_iter{self.iteration}"
        os.makedirs(path, exist_ok=True)

        obs = self.env.reset()
        state = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            act, _ = self.actor(state) 
            action = to_numpy(act).squeeze(0)
            observation, reward, terminated, truncated, _ = self.env.step(
                action
            )

            done = terminated or truncated
            if t%50 == 0 or terminated:
                #print(observation[0], action[0], reward)
                # Render the env
                env_screen = self.env.render()
                cv2.imwrite(path+f"/{t}.png", env_screen)
                #cv2.imshow(f"S: {state}, A: {action}, R: {reward}, S': {observation}", env_screen)
                #cv2.waitKey(1000) #& 0xFF

            episode_reward += reward
            if done:
                print(f"**** EVAL EPISODE: episode_reward={episode_reward}")
                self.results["Iter"].append(self.iteration)
                self.results["Duration"].append(t+1)
                self.results["Reward"].append(episode_reward)
                cv2.destroyAllWindows()
                break
            else:
                state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)


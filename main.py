import gymnasium as gym
import os
import torch

from train import Train
from AC_ContAct_train import Train as Train_AC_ContAct

dir_path = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try_DPG = True
try_SPG = False #stochastic policy gradient
warmup_steps=100
if __name__ == "__main__":

    env = gym.make(
        "MountainCarContinuous-v0", render_mode="rgb_array"
    )
    # Observation and action space
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    if try_DPG:
        # LR = [LR_actor, LR_critic]
        train_agent = Train(
            n_obs,
            n_actions,
            env,
            LR=[0.0001, 0.001],
            memory_length=6000000, #1000000,
            h1=400,
            h2=300,
            init_w=3e-3,
            BATCH_SIZE=64,
            GAMMA=0.99,
            epsilon = 50000, #'linear decay of exploration policy'
            warmup_steps=warmup_steps, #'time without training but only filling the replay memory'
            warmup_decay=None,
            noise_theta=0.15,
            noise_sigma=1.5, #0.5, 0.2,
            sigma_min=0.5, 
            n_steps_annealing=1000,
            eval_episodes=150, #10,
            TAU=0.001,
        )

        w_path = dir_path + "/weights_annealing_check_3"
        train_agent.train(num_episodes=300000000000, max_steps=999+warmup_steps, w_path=w_path) #

    elif try_SPG:
        # LR = [LR_actor, LR_critic]
        train_agent = Train_AC_ContAct(
            n_obs,
            n_actions,
            env,
            LR=[0.0001, 0.001],
            memory_length=64,
            h1=300,
            h2=400,
            BATCH_SIZE=64,
            GAMMA=0.99,
            TAU=0.0001,
        )

        w_path = dir_path + "/weights_SPG"
        train_agent.train(num_episodes=300, max_steps=999, w_path=w_path) #
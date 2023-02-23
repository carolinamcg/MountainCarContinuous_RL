import gymnasium as gym
import os
import torch

from PPO import PPO

dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    env = gym.make(
        "MountainCarContinuous-v0", render_mode="rgb_array"
    )

    w_path = dir_path + "/PPO_test_6"

    # LR = [LR_actor, LR_critic]
    train_agent = PPO(
        env,
        LR=[0.0001, 0.001],
        h1=400,
        h2=300,
        max_timesteps_episode=999, 
        timesteps_per_batch=999*5, #3*episode
        gamma=0.99, 
        epochs=100, 
        clip=0.2, 
        c_s=5,
        c_s_min=0.05, 
        n_steps_annealing=100*10,
        save_path=w_path
    )

    train_agent.learn(total_timesteps=999*5*200)

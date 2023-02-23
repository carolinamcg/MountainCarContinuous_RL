#https://blog.paperspace.com/getting-started-with-openai-gym/
import gymnasium as gym
import matplotlib.pyplot as plt 
import time 
import cv2

# import ale_py

# env = gym.make("MsPacman-v4")
env = gym.make("MountainCarContinuous-v0", max_episode_steps=500, render_mode="rgb_array")
# env = gym.make("CartPole-v1")

# Observation and action space
obs_space = (
    env.observation_space
)  # size=2 -> position and velocity. Box returns to arrays to the minimum and maximum values, respectively, of each state variabloe
action_space = env.action_space #directional force, between [-1,1]
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

#env_screen = env.render(mode = 'rgb_array')
#env.close()

#plt.imshow(env_screen)

# Number of steps you run the agent for 
num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()
    
    # apply the action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step%50 == 0:
        print(obs, action, reward)
        # Render the env
        env_screen = env.render()
        cv2.imshow("State", env_screen)
        cv2.waitKey(1000) #& 0xFF

    # Wait a bit before the next frame unless you want to see a crazy fast video
    #time.sleep(0.005)
    
    # If the epsiode is up, then start another one
    if terminated or truncated:
        env.reset()
        
cv2.destroyAllWindows()
# Close the env
env.close()

del env

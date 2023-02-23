import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits import mplot3d
import numpy as np
import torch

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
USE_CUDA = torch.cuda.is_available()

#matplotlib.use('TkAgg')  # Or any other X11 back-end GTK3Agg

plt.ion() #Enable interactive mode, which shows / updates the figure after every plotting command, so that calling show() is not necessary.


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            plt.gcf() #Get the current figure.
            plt.clf() #Clear figure ???????????????
        else:
            plt.gcf()

def plot_TDE(state_batch, action_batch, TDE, episode, save_path=None):
    states, actions, TDE = to_numpy(state_batch), to_numpy(action_batch), to_numpy(TDE)
    TDE=np.abs(TDE)
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    stateQ = ax.plot3D(states[:, 0], states[:,1], TDE[:,0])
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("TDE")

    ax = fig.add_subplot(1, 2, 2)
    actQ = ax.plot(actions[:,0], TDE[:,0])
    ax.set_xlabel("Action")
    ax.set_ylabel("TDE")

    fig.tight_layout()
    #plt.show()
    #plt.pause(0.001)
    if save_path is not None:
        plt.savefig(save_path + f"/Ep{episode}_check.png")

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

    '''
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    self.target_net.load_state_dict(target_net_state_dict)
    '''

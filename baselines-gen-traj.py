import gym

from mycart import MyCartPoleEnv
from mycartCont import MyCartContEnv

from stable_baselines.gail import generate_expert_traj
import numpy as np

env = MyCartContEnv()

# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller
def dummy_expert(_obs):
    x, x_dot, theta, theta_dot = _obs
    #print(obs)
    K1 = -50
    K2 = -5
    K3 = -4
    K4 = -2
    
    action = [-K1*theta - K2*theta_dot - K3*(x-env.xref) - K4*x_dot]
    return action
# Data will be saved in a numpy archive named `expert_cartpole.npz`
# when using something different than an RL expert,
# you must pass the environment object explicitely
generate_expert_traj(dummy_expert, 'dummy_expert_cartpole', env, n_episodes=100)

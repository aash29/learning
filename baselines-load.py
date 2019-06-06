import os

import gym
import numpy as np
import matplotlib.pyplot as plt

#from stable_baselines.ddpg.policies import MlpPolicy
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG, PPO1, SAC
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec

from mycart import MyCartPoleEnv
from mycartCont import MyCartContEnv

best_mean_reward, n_steps = -np.inf, 0
log_dir = "./baseline-logs/"
os.makedirs(log_dir, exist_ok=True)

env = MyCartContEnv()

# Create and wrap the environment
model = SAC(MlpPolicy, env)
model.load('./baseline-logs/best_model.pkl')

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

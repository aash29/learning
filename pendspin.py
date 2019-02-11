from spinup import ppo,trpo
import tensorflow as tf
import gym
from mycart import MyCartPoleEnv

#env_fn = lambda : gym.make('LunarLander-v2')
env_fn = lambda : MyCartPoleEnv()

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='./logsDDPG', exp_name='pend')

trpo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)
import os
import gym
import numpy as np
import matplotlib.pyplot as plt


import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1,A2C, DDPG, SAC

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor


from stable_baselines.ddpg import AdaptiveParamNoiseSpec


from stable_baselines.gail import ExpertDataset



best_mean_reward, n_steps = -np.inf, 0

#env = gym.make('CartPole-v1')

from mycart import MyCartPoleEnv
from mycartCont import MyCartContEnv


log_dir = "./baseline-logs/"
os.makedirs(log_dir, exist_ok=True)

env = MyCartContEnv()
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env]) # The algorithms require a vectorized environment to run


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')





def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')

        
    
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        #yn = np.array([],dtype=np.float32)
        #for y1 in y:
        #    yn = np.append(yn,float(y1[1:-1]))
        #y = yn
        if len(x) > 0:
            mean_reward = np.mean(np.asfarray(y[-100:]))
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True

param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)


# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
dataset = ExpertDataset(expert_path='dummy_expert_cartpole.npz',
                        traj_limitation=1, batch_size=128)


#model = DDPG(stable_baselines.ddpg.MlpPolicy, env, param_noise=param_noise, verbose=1)
#model = PPO1(MlpPolicy, env, verbose=1)
model = SAC(stable_baselines.sac.MlpPolicy, env, verbose=1)


# Pretrain the  model
model.pretrain(dataset, n_epochs=10000)


model.learn(total_timesteps=10000000,callback=callback,log_interval=10)
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs,deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()

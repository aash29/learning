#!/usr/bin/python

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from copy import deepcopy

ENV_NAME = 'city'

distToWork = 1;
distToShop = 1;
workDifficulty = 1;
workTemperature = 1;
workStart = 0;
workDuration = 24;
timeLimit = 90;


class eat():
    duration = 1
    name = 'eat'

    # @staticmethod
    def pre(state):
        # result = super(eat,eat).pre();
        result = (state[10] < timeLimit) & (state[5] > 0) & (state[0] > 0) & (state[9] == 0) & (state[4] > 0)
        return result

    # @staticmethod
    def eff(state):
        # global utime
        # super(eat,eat).eff();

        s1 = deepcopy(state)

        s1[10] += eat.duration
        s1[0] = 32
        s1[5] -= eat.duration
        s1[5] -= 1

        return s1



actions = [eat]

class city(gym.Env):
    def __init__(self):
        self.distToWork = 1;
        self.distToShop = 1;
        self.workDifficulty = 1;
        self.workTemperature = 1;
        self.workStart = 0;
        self.workDuration = 24;
        self.timeLimit = 90;

        #0: sat = 32;
        #1: heatHome = 0;
        #2: inv = # foodstamp;
        #3: invHomeWood = 0;
        #4: invHomeFood = 0;
        #5: energy = 256;
        #6: absent = 0;
        #7: workDays = 0;
        #8: isHomeHeated = 0;
        #9: loc  # atHome = 0 , atWork = 1
        #10: utime = 0;  # universal time


        self.action_space = gym.spaces.Discrete(1);
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                                high=np.array([32, 10, 10, 10, 10, 256, 3, 5, 1, 2, 96]));

        self.isd = np.array([20,
                             0,
                             2,
                             1,
                             1,
                             256,
                             0,
                             0,
                             0,
                             0,
                             0])
        self.s = deepcopy(self.isd)
        self.observation = deepcopy(self.isd)



    def step(self, a):
        action = actions[a]
        s1 = deepcopy(self.s)
        if (action.pre(self.s)):
            s1 = action.eff(self.s)
        r = 0
        if (s1[10] > 60) & (s1[10] < 100) & (s1[5] > 0) & (s1[0] > 0) & (s1[7] > 1):
            r = 100
        self.s = s1;
        return s1, r, False, {}

    def reset(self):
        self.s = deepcopy(self.isd)
        return self.s

    def render(self, mode='human', close=False):
        print(self.s);





# Get the environment and extract the number of actions.
#env = gym.make(ENV_NAME)
env = city();
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(11))
model.add(Activation('relu'))
model.add(Dense(11))
model.add(Activation('relu'))
model.add(Dense(11))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)

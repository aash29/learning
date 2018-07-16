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

from keras import backend as K

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


        return s1

class wait():
    name = 'wait'
    duration = 10
    def __str__(self):
        return 'wait'
    def pre(state):
        return (state[10]<timeLimit)&(state[5]>0)&(state[0]>0)
    def eff(state):
        s1 = deepcopy(state);
        s1[0] -= wait.duration
        s1[5] -= wait.duration
        s1[10] += wait.duration

        
        #print('Waiting');
        return s1

class gotoWork():
    name = 'go to work'
    duration = 2*distToWork+1
    def pre(state):
        return (state[10]<timeLimit) & (state[5] > 0) & (state[0] > 0) & (state[9] == 0)
    def eff(state):
        s1 = deepcopy(state)

        s1[9] = 1

        s1[0] -= 1

        s1[5] -= gotoWork.duration

        s1[10] += gotoWork.duration

        return s1

class work():
    name = 'work'
    duration = 20
    def pre(state):
        return (state[10] < timeLimit) & (state[5] > 0) & (state[0] > 0) & (state[9] == 1)
    def eff(state):
        s1 = deepcopy(state)

        s1[5] -= work.duration

        s1[0] -= 1

        s1[7] += 1
        s1[10] += work.duration

        return s1

actions = [eat, wait, gotoWork, work]

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


        self.action_space = gym.spaces.Discrete(4);
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
        self.nActions = 0
        self.observation = deepcopy(self.isd)



    def step(self, a):
        action = actions[a]
        s1 = deepcopy(self.s)
        r = 0
        if (action.pre(self.s)):
            s1 = action.eff(self.s)
        else:
            r -= 1000

        if (s1[10] > 80) & (s1[10] < 100) & (s1[5] > 0) & (s1[0] > 0) & (s1[7] > 1):
            r += 1000


        self.s = s1
        self.nActions += 1

        #r += s1[10]
        
        if self.nActions > 100:
            reset = True
        else:
            reset = False
        return s1, r, reset, {}

    def reset(self):
        self.s = deepcopy(self.isd)
        self.nActions = 0
        return self.s

    def render(self, mode='human', close=False):
        print(self.s)





# Get the environment and extract the number of actions.
#env = gym.make(ENV_NAME)
env = city();
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
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
dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_weights.h5f', overwrite=True)


# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=5, visualize=True)

kvar = K.variable(env.isd.reshape((1,) + env.observation_space.shape))
y = dqn.model(kvar)
a0 = np.argmax(y)

for i in range(1, 10):
    print(a0)
    s1 = env.step(a0)
    kvar = K.variable(s1[0].reshape((1,) + env.observation_space.shape))
    y = dqn.model(kvar)
    yv = K.eval(y)
    print(yv)
    a0 = np.argmax(yv)

print(s1[0])






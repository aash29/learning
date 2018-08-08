#!/usr/bin/python

#import numpy as np
import array
import gym
import matplotlib.pyplot as plt
import numpy as np

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

        s1 = list(deepcopy(state))

        s1[10] += eat.duration
        s1[0] = 32
        s1[4] -= 1
        s1[5] -= eat.duration

        return tuple(s1)

class wait():
    name = 'wait'
    duration = 5
    def __str__(self):
        return 'wait'
    def pre(state):
        return (state[10] < timeLimit) & (state[5] > 0) & (state[0] > 0)
    def eff(state):
        s1 = list(deepcopy(state))
        s1[0] -= wait.duration
        s1[5] -= wait.duration
        s1[10] += wait.duration

        
        #print('Waiting');
        return tuple(s1)

class gotoWork():
    name = 'go to work'
    duration = 2*distToWork+1
    def pre(state):
        return (state[10]<timeLimit) & (state[5] > 0) & (state[0] > 0) & (state[9] != 1)
    def eff(state):
        s1 = list(deepcopy(state))

        s1[9] = 1

        s1[0] -= 1

        s1[5] -= gotoWork.duration

        s1[10] += gotoWork.duration

        return tuple(s1)


class goToShop():
    name = 'go to shop'
    duration = 2 * distToShop + 1

    def pre(state):
        return (state[10] < timeLimit) & (state[5] > 0) & (state[0] > 0) & (state[9] != 2)

    def eff(state):
        s1 = list(deepcopy(state))

        s1[9] = 2

        s1[0] -= 1

        s1[5] -= goToShop.duration

        s1[10] += goToShop.duration

        return tuple(s1)


class buyFood():
    name = 'buy food'
    duration = 1
    def pre(state):
        return (state[10] < timeLimit) & (state[5] > 0) & (state[0] > 0) & (state[9] == 2) & (state[2] > 0)

    def eff(state):
        s1 = list(deepcopy(state))
        s1[2] -= 1
        s1[4] += 1
        return tuple(s1)



class goHome():
    name = 'go to work'
    duration = 2 * distToShop + 1
    def pre(state):
        return (state[10] < timeLimit) & (state[5] > 0) & (state[0] > 0) & (state[9] != 0)
    def eff(state):
        s1 = list(deepcopy(state))

        s1[9] = 0

        s1[0] -= 1

        s1[5] -= gotoWork.duration

        s1[10] += gotoWork.duration

        return tuple(s1)



class work():
    name = 'work'
    duration = 20
    def pre(state):
        return (state[10] < timeLimit) & (state[5] > 0) & (state[0] > 0) & (state[9] == 1)
    def eff(state):
        s1 = list(deepcopy(state))

        s1[5] -= work.duration

        s1[0] -= 1

        s1[7] += 1
        s1[10] += work.duration

        return tuple(s1)

actions = [eat, wait, gotoWork, work, goHome, goToShop, buyFood]

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
        #9: loc  # atHome = 0, atWork = 1, atShop = 2
        #10: utime = 0;  # universal time


        #self.action_space = gym.spaces.Discrete(4);
        #self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),high=np.array([32, 10, 10, 10, 10, 256, 3, 5, 1, 2, 96]));

        self.isd = (20,
                             0,
                             2,
                             1,
                             1,
                             256,
                             0,
                             0,
                             0,
                             0,
                             0)
        self.s = deepcopy(self.isd)
        self.nActions = 0
        self.observation = deepcopy(self.isd)



    def step(self, a):
        action = a
        r = 0
        s1 = action.eff(self.s)

        #if (action.pre(self.s)):
        #    s1 = action.eff(self.s)
        #else:
        #    r -= 100


        if (s1[10] > 60) & (s1[10] < 100) & (s1[5] > 0) & (s1[0] > 0) & (s1[7] > 1):
            r += 100
            r += s1[0]*10 #sat
            r += s1[4]*10  #food
            if (s1[9] == 0):
                r += 100
            r += s1[7]*10 #work done

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



def getLegalActions(s):
    legalActions = []
    for action in actions:
        if (action.pre(s)):
            legalActions.append(action)
    return legalActions
    
from qlearning import QLearningAgent
agent = QLearningAgent(alpha=0.5, epsilon=0.5, discount=0.99,
                       get_legal_actions = getLegalActions)

def play_and_train(env,agent,t_max=10**4):
    """
    This function should 
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s = env.reset()
    
    
    for t in range(t_max):
        # get agent to pick action given state s.
        a = agent.get_action(s)
        print(a)
        if a == None:
            break
        
        next_s, r, done, _ = env.step(a)
        #print("step")
        
        # train (update) agent for state s
        agent.update(s,a, r,next_s)
        
        s = next_s
        total_reward +=r
        if done: break
        
    return total_reward
    
        
    


# Get the environment and extract the number of actions.
#env = gym.make(ENV_NAME)
env = city();
np.random.seed(123)
#env.seed(123)
#nb_actions = env.action_space.n



rewards = []
for i in range(10000):
    rewards.append(play_and_train(env, agent))
    agent.epsilon *= 0.999
    
    if i %1000 ==0:
        #clear_output(True)
        print('eps =', agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
        plt.plot(rewards)
        plt.show()


s0 = env.isd;
while (s0[10]<timeLimit):
    a0 = agent.get_action(s0)
    s0 = a0.eff(s0)
    print(a0)
    print(s0)




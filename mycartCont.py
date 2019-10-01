import gym
import numpy as np
import math

from gym import spaces, logger
from gym.utils import seeding


class MyCartContEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value between ±0.05
    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.01  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.xref = 0

        # Angle at which to fail the episode
        self.theta_threshold_radians =  60 * math.pi / 360
        self.x_threshold = 2.4
        self.t = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.ndarray.flatten(np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max]))
        highF = 100

        self.action_space =  spaces.Box(-highF, highF, dtype=np.float32, shape=(1,))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print(action)
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot= state

        self.t =  self.t+1
        #print(action)
        force = action

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
     
        reward = 0
        done = x < -self.x_threshold \
               or x >  self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians \
               or self.t > 5000

        


        done = bool(done)

     
        if not done:
            #reward = max(1/(abs(x-1)), 100) + max(1/(theta), 100)
            #reward = -10*(abs(x)) -10*abs(theta)

            #r1 = abs(self.theta_threshold_radians - abs(theta))/self.theta_threshold_radians
            #r2 = abs(self.x_threshold- abs(x))/self.x_threshold
            #reward =  r1 + r2**2 

            #xref = -1.0
            yref = 1.0
            

            xp = x + math.sin(theta)
            yp = math.cos(theta)

            errorSq = (xp - self.xref)**2 + (yp - yref)**2

            #reward = (1 - error)/abs(1 - error)*(1 - error)**2
            #reward = (2 - error)

            #reward = math.exp(-10*errorSq) - 0.1

            reward = math.exp(-2*errorSq) - 0.1  - abs(theta_dot)

            #reward = -error

            #print(x)

            #if (self.t%50 == 0):
            #    print (r1, r2)

            #if (action==2):
            #    reward = 1

            #r1 = (self.x_threshold - abs(x))/self.x_threshold - 0.8
            #r2 = (self.theta_threshold_radians - abs(theta))/self.theta_threshold_radians - 0.5
            #reward = r1 + r2

        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        stateArray = np.array(self.state,dtype=np.float32)
        stateArray.shape = (4,)

        return stateArray, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=[-2,-0.5,-0.1, -0.1], high=[2,0.5,0.1, 0.1], size=(4,))
        self.steps_beyond_done = None
        self.t = 0
        self.xref = self.np_random.uniform(low = -2, high = 2)
        #self.xref = 2.3
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self.ref = rendering.Line((0, 0), (0, 30))
            self.refTrans = rendering.Transform()
            self.ref.add_attr(self.refTrans)
            self.ref.set_color(0, 0, 0)
            self.viewer.add_geom(self.ref)


            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        self.refTrans.set_translation(self.xref * scale + screen_width / 2.0,0)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


import gym
from gym import spaces
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import os

class CatAndRatEnvFixedRat(gym.Env):
    metadata = {'render.modes': ['console']}
    #ACTION CONSTANTS
    LEFT = 8
    RIGHT = 1
    UP = 2
    DOWN = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    DO_NOTHING = 0



    def __init__(self):
        super(CatAndRatEnvFixedRat, self).__init__()


        self.maxY = 800 #replace with screenHeight from Game
        self.maxX = 1500 #replace with screenWidth from Game
        self.maxTheta = 2 * math.pi
        self.minX = 0
        self.minY = 0
        self.minTheta = 0
        self.randomX = self.maxX * random.random()
        self.randomY = self.maxY * random.random()
        self.totalFrames = 3600
        global cat
        global rat
        cat = DynamicsSimulator(m = 1, positions = [self.randomX, self.randomY], damping = 1, dt = .01666667, max_force = 11)
        rat = DynamicsSimulator(m = 1, positions = [self.randomX, self.randomY], damping = 1, dt = .01666667, max_force = 11)

        high = np.array([self.maxX,
                         self.maxY,
                         self.maxX,
                         self.maxY,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        low = np.array([self.minX,
                        self.minY,
                        self.minX,
                        self.minY,
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).min,
                        np.finfo(np.float32).min],
                       dtype=np.float32)

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        xSep1 = cat.x - rat.x
        ySep1 = cat.y - rat.y

        if xSep1 > 0.5 * self.maxX:
            xSep1 = self.maxX - xSep1
        if ySep1 > 0.5 * self.maxY:
            ySep1 = self.maxY - ySep1

        distance1 = math.sqrt((xSep1)**2 + (ySep1)**2)


        if action == self.LEFT:
            cat.strat_force_x = -cat.max_force
            cat.strat_force_y = 0
        elif action == self.RIGHT:
            cat.strat_force_x = cat.max_force
            cat.strat_force_y = 0
        elif action == self.UP:
            cat.strat_force_y = cat.max_force
            cat.strat_force_x = 0
        elif action == self.DOWN:
            cat.strat_force_y = -cat.max_force
            cat.strat_force_x = 0
        elif action == self.UP_LEFT:
            cat.strat_force_x = -cat.max_force * math.cos(math.pi * 0.25)
            cat.strat_force_y = cat.max_force * math.sin(math.pi * 0.25) #feed 0 back to the JS simulator, equate this with left key down?
        elif action == self.UP_RIGHT:
            cat.strat_force_x = cat.max_force * math.cos(math.pi * 0.25)
            cat.strat_force_y = cat.max_force * math.sin(math.pi * 0.25)
        elif action == self.DOWN_LEFT:
            cat.strat_force_x = -cat.max_force * math.cos(math.pi * 0.25)
            cat.strat_force_y = -cat.max_force * math.sin(math.pi * 0.25)
        elif action == self.DOWN_RIGHT:
            cat.strat_force_x = cat.max_force * math.cos(math.pi * 0.25)
            cat.strat_force_y = -cat.max_force * math.sin(math.pi * 0.25)
        elif action == self.DO_NOTHING:
            cat.strat_force_x = 0
            cat.strat_force_y = 0
        else:
            raise ValueError("Recived invalid action!")
        cat.rk4_step()
        #Enforces toroidal space on cat
        if cat.x <= 0:
            cat.x = self.maxX + cat.x
        if cat.y <= 0:
            cat.y = self.maxY + cat.y
            
        if cat.x >= self.maxX:
            cat.x = cat.x - self.maxX
        if cat.y >= self.maxY:
            cat.y = cat.y - self.maxY
        #rat decides if it should turn or not

        #counter = 1
        #if counter == 1:
        #    counter += 1
        #    if random.random() < 0.33:
        #        rat.theta += 0.1
        #    elif random.random() > 0.66:
        #        rat.theta -= 0.1
        #else:
        #    counter -= 1

        #rat.strat_force_x = rat.max_force * math.cos(rat.theta)
        #rat.strat_force_y = rat.max_force * math.sin(rat.theta)

        #rat.rk4_step()
        #Enforces toroidal space on rat
        #if rat.x <= 0:
        #    rat.x = self.maxX + rat.x
        #if rat.y <= 0:
            #rat.y = self.maxY + rat.y

        xSep2 = cat.x - rat.x
        ySep2 = cat.y - rat.y

        if xSep2 > 0.5 * self.maxX:
            xSep2 = self.maxX - xSep2
        if ySep2 > 0.5 * self.maxY:
            ySep2 = self.maxY - ySep2

        distance2 = math.sqrt((xSep2)**2 + (ySep2)**2)

        self.totalFrames -= 1
        checkIfCatch = bool(
            abs(cat.x - rat.x) < 11.5 and abs(cat.y - rat.y) < 11.5
        )
        timesUp = bool(self.totalFrames == 0)
        done = bool(
            checkIfCatch == True
            or timesUp == True
        )
        reward = 0
        if not checkIfCatch and not timesUp:
            if distance1 - distance2 > 0:
                reward = 1
            elif distance1 - distance2 < 0:
                reward = -1
            else:
                reward = 0
        elif checkIfCatch and not timesUp:
            reward = 5000 - (3600 - self.totalFrames)
        elif timesUp and not timesUp:
            reward = -1
        else:
            reward = 0

        if reward > 5000:
            reward = 5000

            #theta = x; maybe include theta later?
        self.state = (cat.x, cat.y, rat.x, rat.y, cat.v_x, cat.v_y, rat.v_x, rat.v_y)
        info = {}
        return np.array(self.state), reward, done, info

    def reset(self):
        self.totalFrames = 3600
        cat.x = self.maxX * random.random()
        cat.y = self.maxY * random.random()
        rat.x = self.maxX * random.random()
        rat.y = self.maxY * random.random()
        cat.v_x, cat.v_y = 0, 0
        rat.v_x, rat.v_y = 0, 0
        cat.strat_force_x, cat.strat_force_y = 0, 0
        rat.strat_force_x, rat.strat_force_y = 0, 0
        self.state = (cat.x, cat.y, rat.x, rat.y, cat.v_x, cat.v_y, rat.v_x, rat.v_y)
        return np.array(self.state)

    def render(self, mode='human'):
        screenWidth = 1500
        screenHeight = 800
        world_width = 1500
        scale = screenWidth/world_width
        caty = 16
        raty = 16
        catx = 16
        ratx = 16

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screenWidth, screenHeight)
            catL, catR, catT, catB = -catx/2, catx/2, caty/2, -caty/2
            ratL, ratR, ratT, ratB = -ratx/2, ratx/2, raty/2, -raty/2

            catPic = rendering.FilledPolygon([(catL, catB), (catL, catT), (catR, catT), (catR, catB)])
            catPic.set_color(255, 0, 0)
            self.cattrans = rendering.Transform()

            ratPic = rendering.FilledPolygon([(ratL, ratB), (ratL, ratT), (ratR, ratT), (ratR, ratB)])
            ratPic.set_color(0, 191, 255)
            self.rattrans = rendering.Transform()


            catPic.add_attr(self.cattrans)
            ratPic.add_attr(self.rattrans)

            self.viewer.add_geom(catPic)
            self.viewer.add_geom(ratPic)

        if self.state is None:
            return None
    
        self.cattrans.set_translation((cat.x)/2, (cat.y)/2)
        
        self.rattrans.set_translation((rat.x)/2, (rat.y)/2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class DynamicsSimulator:
    def __init__(self, m, positions, velocities = [0, 0], damping = 0, dt = .01666667, max_force = 5):
        # set attributes like mass, x_0, v_0, t_0, damping from friction
        self.m = m
        # self.k_x, self.k_y = k[0], k[1]
        self.x, self.y = positions[0], positions[1]
        self.v_x, self.v_y = velocities[0], velocities[1]
        self.strat_force_x, self.strat_force_y = 0, 0
        self.strategy_label = 'Start!'
        self.t = 0
        self.dt = dt
        self.damping = damping
        self.max_force = max_force
        self.theta = 0
        # self.environment = environment


    def vel_func(self, positions, velocities, time):
        velocities = np.array([self.v_x, self.v_y])
        return velocities

    def accel_func(self, positions, velocities, time):
        # potential_forces = self.environment.get_env_forces(positions)
        strategy_forces = np.array([self.strat_force_x, self.strat_force_y])
        damping_forces = self.damping * self.vel_func(positions, velocities, time)
        total_forces = strategy_forces + damping_forces
        magnitude = math.sqrt(total_forces[0]**2 + total_forces[1]**2)
        if magnitude > self.max_force: # normalize to max_force if exceeded
            fractional_x = total_forces[0] / magnitude
            fractional_y = total_forces[1] / magnitude
            total_forces[0] = fractional_x * self.max_force
            total_forces[1] = fractional_y * self.max_force
        accelerations = total_forces / self.m
        return accelerations

    def rk4_step(self):
        current_positions = np.array([self.x, self.y])
        current_velocities = np.array([self.v_x, self.v_y])
        # numerics to calculate coefficients:
        pos_k1 = self.dt * self.vel_func(current_positions, current_velocities, self.t)
        vel_k1 = self.dt * self.accel_func(current_positions, current_velocities, self.t)
        pos_k2 = self.dt * self.vel_func(current_positions + (pos_k1 / 2), current_velocities + (vel_k1 / 2 ), self.t + (self.dt / 2))
        vel_k2 = self.dt * self.accel_func(current_positions + (pos_k1 / 2), current_velocities + (vel_k1 / 2), self.t + (self.dt / 2))
        pos_k3 = self.dt * self.vel_func(current_positions + (pos_k2 / 2), current_velocities + (vel_k2 / 2), self.t + (self.dt / 2))
        vel_k3 = self.dt * self.accel_func(current_positions + (pos_k2 / 2), current_velocities + (vel_k2 / 2), self.t + (self.dt / 2))
        pos_k4 = self.dt * self.vel_func(current_positions + pos_k3, current_velocities + vel_k3, self.t + self.dt)
        vel_k4 = self.dt * self.accel_func(current_positions + pos_k3, current_velocities + vel_k3, self.t + self.dt)
        # update according to rk4 formula:
        new_positions = current_positions + ((1/6) * (pos_k1 + 2*pos_k2 + 2*pos_k3 + pos_k4))
        new_velocites = current_velocities + ((1/6) * (vel_k1 + 2*vel_k2 + 2*vel_k3 + vel_k4))
        # load new values into instance attributes:
        self.x, self.y = new_positions[0], new_positions[1]
        self.v_x, self.v_y = new_velocites[0], new_velocites[1]
        self.t += self.dt

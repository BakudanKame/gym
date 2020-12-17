import gym
from gym import spaces
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import os


class RatEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(RatEnvironment, self).__init__()


        self.maxY = 600 #replace with screenHeight from Game
        self.maxX = 800 #replace with screenWidth from Game
        self.maxTheta = 2 * math.pi
        self.minX = 0
        self.minY = 0
        self.minTheta = 0
        self.randomX = self.maxX * random.random()
        self.randomY = self.maxY * random.random()
        self.totalFrames = 3600
        global cat
        global rat
        cat = DynamicsSimulator(m = 1, positions = [self.randomX, self.randomY], damping = -2, dt = (1/60), max_force = 800)
        rat = DynamicsSimulator(m = 1, positions = [self.randomX, self.randomY], damping = -5, dt = (1/60), max_force = 1100)


        self.action_space = spaces.Box(np.array([0]), np.array([math.pi * 2]), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([0]), np.array([500]))
        self.state = None
        self.viewer = None




    def step(self, action):

        x = cat.x - rat.x
        y = cat.y - rat.y

        SCREEN_WIDTH = self.maxX
        SCREEN_HEIGHT = self.maxY

        if x > SCREEN_WIDTH/2:
            x = x - SCREEN_WIDTH

        elif x < -SCREEN_WIDTH/2:
            x = x + SCREEN_WIDTH

        if y > SCREEN_HEIGHT/2:
            y = y - SCREEN_HEIGHT
        elif y < -SCREEN_HEIGHT/2:
            y = y + SCREEN_HEIGHT

        angle = math.atan2(y, x) + math.pi

        distanceBetween = math.sqrt(x**2+y**2)


        xEval = angle + action

        xEval = xEval % (2 * math.pi)

        if xEval < 0:
            xEval = xEval + 2 * math.pi


        rat.strat_force_x = 1100 * math.cos(xEval)
        rat.strat_force_y = 1100 * math.sin(xEval)

        rat.rk4_step()

        if rat.x <= 0:
            rat.x = self.maxX + rat.x
        if rat.y <= 0:
            rat.y = self.maxY + rat.y

        if rat.x >= self.maxX:
            rat.x = rat.x - self.maxX
        if rat.y >= self.maxY:
            rat.y = rat.y - self.maxY


        #TEST CAT
        angleToTarget = math.atan2(cat.y - rat.y, cat.x - rat.x)
        cat.strat_force_x = cat.max_force * math.cos(angleToTarget + math.pi)
        cat.strat_force_y = cat.max_force * math.sin(angleToTarget + math.pi)
        cat.rk4_step()
        if cat.x <= 0:
            cat.x = self.maxX + cat.x
        if cat.y <= 0:
            cat.y = self.maxY + cat.y

        if cat.x >= self.maxX:
            cat.x = cat.x - self.maxX
        if cat.y >= self.maxY:
            cat.y = cat.y - self.maxY

        #END TEST CAT


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
        if checkIfCatch == False:
            reward = 1
        else:
            reward = 0

        self.state = distance2
        info = {}
        return np.array([self.state]), reward, done, info


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
        xSep3 = cat.x - rat.x
        ySep3 = cat.y - rat.y

        if xSep3 > 0.5 * self.maxX:
            xSep3 = self.maxX - xSep3
        if ySep3 > 0.5 * self.maxY:
            ySep3 = self.maxY - ySep3

        distance3 = math.sqrt((xSep3)**2 + (ySep3)**2)


        self.state = distance3
        return np.array([self.state])




    def render(self, mode='human'):
        screenWidth = 800
        screenHeight = 600
        world_width = 800
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

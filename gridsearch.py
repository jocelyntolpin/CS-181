# Look at final.py for comments if you are having trouble understanding the code in 
# the learner class. They are esentially the same, but the plot_scores function has changed.

# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import random
import matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey

import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# global variables that we are going to change later
epochs = 300
eta=1
gamma=1
eps=0.001

class Learner(object):
    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.disc = [12, 8, 5, 2] 
        self.bounds = [(-150,500), (-200,350),(-45,45)]
        Q = np.zeros((self.disc + [2]))
        self.Q = Q
        self.times = np.zeros((self.disc + [2]))
        self.eta = eta
        self.gamma = gamma
        self.eps = eps
        self.epochs = 0
        self.new_game = True
        self.prev_vel = None
        self.grav = None

        self.score = None
        self.scores = []

    def reset(self):
        self.epochs += 1
        self.new_game = True
        self.prev_vel = None
        self.scores.append(self.score)


    def __discretize(self, index, val):
        l,u = self.bounds[index]
        width = u - l
        step = width / self.disc[index]
        return np.floor((val - l) / step)

    def __discretize_array(self, values):
        indices = []
        for i in range(len(values)):
            indices.append(int(self.__discretize(i,values[i])))

        return indices

    def action_callback(self, state):
        if self.new_game and self.last_reward != None:
            first = self.Q[tuple(self.last_state+[self.grav])][self.last_action]*(1-self.eta)
            second = self.eta*(self.last_reward) 
            self.Q[tuple(self.last_state+[self.grav])][self.last_action] = first+second
            self.grav = None
            self.last_state  = None
            self.last_action  = None
            self.last_reward = None

        values = [state['tree'][i] for i in ['dist','top','bot']] + [state['monkey'][i] for i in ['vel','top']]
        values = [values[0], values[1] - values[-1], values[-2]]

        self.score = state['score']
        indices = self.__discretize_array(values)

        if self.prev_vel != None and self.grav == None:
            dif = np.abs(values[2] - self.prev_vel)
            if dif == 4:
                self.grav = 1
            elif dif == 1:
                self.grav = 0
            else:
                raise 1 

        new_action = 0
        grav_val = 0
        if self.grav != None:
            grav_val = self.grav

        rewards = self.Q[tuple(indices+[grav_val])]

        if self.last_state != None and not self.new_game:  

            epsilon = self.eps
            max_action = np.argmax(rewards)
            max_reward = np.max(rewards)

            if self.times[tuple(indices+[grav_val])][max_action] > 0:
                epsilon /= self.times[tuple(indices+[grav_val])][max_action]

            random = npr.rand() < epsilon
            if random:
                choice = npr.rand() < 0.5
                if choice:
                    new_action = 0
                else:
                    new_action = 1
            else:
                new_action = max_action

            eta = self.eta
            if self.times[tuple(self.last_state+[grav_val])][self.last_action] > 0:
                eta /= self.times[tuple(self.last_state+[grav_val])][self.last_action]

            self.Q[tuple(self.last_state+[grav_val])][self.last_action] += eta * (self.last_reward + self.gamma * max_reward - self.Q[tuple(self.last_state+[grav_val])][self.last_action])


        self.times[tuple(indices+[grav_val])][new_action] += 1
        self.last_action = new_action
        self.last_state  = self.__discretize_array(values)
        self.prev_vel = values[2]

        if self.new_game:
            self.new_game = False
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

    def plot_scores(self):

        print('eta: {}, gamma: {}, epsilon: {}.'.format(self.eta, self.gamma, self.eps))
        print(self.scores)

        print('Average score over all epochs: {}'.format(np.average(self.scores)))
        print('Max score over all epochs: {}'.format(np.max(self.scores)))

        print("\n")


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        #coordinates.append(swing.coordinates)

        # Reset the state of the learner.
        learner.reset()

    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []
    # Emtply list to save coordinates
    coordinates = []

    gammas = [0.6,0.7,0.8,0.9,1] # gamma values to test
    epsilons = [.001, .01, 0.1] # epsilon values to test
    etas = [0.9, 0.8, 0.7, 0.6] # eta values to test
    
    for et in etas:
        for g in gammas:
            for e in epsilons:
                eps = e
                gamma = g
                eta = et
                agent = Learner()
                run_games(agent, hist, epochs, 10)
                agent.plot_scores()

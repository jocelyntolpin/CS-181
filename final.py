# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt
import os

from SwingyMonkey import SwingyMonkey

# don't display the output for faster running. Comment out to see your monkey friend.
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# number of training iterations
epochs = 300

class Learner(object):
    def __init__(self):
        # The monkey does not start with a prvious state, action, nor reward.
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # discretization and bounds
        self.disc = [12, 8, 5, 2] # num horizontal states, num vertical states, velocity states, gravity states
        self.bounds = [(-150,500), (-200,350), (-45,45)] # bounds of first three states. These values are fed into our helper functions.
        Q = np.zeros((self.disc + [2])) # create the Q Matrix with 2 actions for each state.
        self.Q = Q
        self.times = np.zeros((self.disc + [2])) # keep track of how many times each state-action pair is hit
        
        # intialize parameters. These were our optimal params.
        self.eta = 1
        self.gamma = 1
        self.eps = 0.001
        self.epochs = 0
        self.new_game = True
        
        # initialize variables representing current 
        # gravity state and previous velocity
        self.prev_vel = None
        self.grav = None

        # let's keep track of the scores, too
        self.score = None
        self.scores = []

    def reset(self):
        # add to number of epochs for training after each game finishes
        self.epochs += 1
        # boolean flag to keep track of restarts
        self.new_game = True
        # reinitilize previous velocity value
        self.prev_vel = None
        # append score to list of scores
        self.scores.append(self.score)


    def __discretize(self, index, val):
        # Helper func: input is the dimension of interest and value to course-grain
        l,u = self.bounds[index]
        width = u - l
        step = width / self.disc[index]
        # returns the bin to put the value in
        return np.floor((val - l) / step)

    def __discretize_array(self, values):
        # Helper func: returns the indices of each value in the Q matrix
        # by finding which bin it falls in
        indices = []
        for i in range(len(values)):
            indices.append(int(self.__discretize(i,values[i])))

        return indices

    def action_callback(self, state):
        # if in a new game and there is a previous reward (so not very first epoch)
        # aka, we're in a "death state," or we just died
        if self.new_game and self.last_reward != None:
            # first component of Q update via stochastic gradient descent
            first = self.Q[tuple(self.last_state+[self.grav])][self.last_action]*(1-self.eta)
            # second term of Q update in stochastic gradient descent. We are letting the 
            # Q value of a death-state be 0
            second = self.eta*(self.last_reward + 0) 
            # update Q
            self.Q[tuple(self.last_state+[self.grav])][self.last_action] = first+second
            # reset values of gravity, last state, last action, and last reward for next game
            # since we're in a death state and we need to start over
            self.grav = None
            self.last_state  = None
            self.last_action  = None
            self.last_reward = None

        # define list that describes current state
        values = [state['tree'][i] for i in ['dist','top','bot']] + [state['monkey'][i] for i in ['vel','top']]
        # compress current state space to be distance from tree, distance from 
        # top of tree to top of monkey, and velocity of monkey
        values = [values[0], values[1] - values[-1], values[-2]]

        # define the current score
        self.score = state['score']
        # find the indices of the state space in the Q matrix, except we are still missing gravity
        indices = self.__discretize_array(values)

        # define the value of gravity in this game if we have not already defined it
        if self.prev_vel != None and self.grav == None:
            # using the difference between current velocity and previous velocity,
            # we can determine how much gravity is in this game
            dif = np.abs(values[2] - self.prev_vel)
            if dif == 4:
                self.grav = 1
            elif dif == 1:
                self.grav = 0
            else:
                # if the gravity is not 1 or 4, something went wrong. Our code ensures that we never jump
                # for the very first state in an epoch
                raise 1 

        # initialize variables
        new_action = 0
        grav_val = 0

        if self.grav != None:
            grav_val = self.grav # this only updates once per game

        # define the possible rewards form this state by finding the value of the indices in the Q matrix
        rewards = self.Q[tuple(indices+[grav_val])]

        # if in the middle of a game...
        if self.last_state != None and not self.new_game:  
            epsilon = self.eps
            # defining the action that maximizes the reward, from the indexing of the Q matrix above
            max_action = np.argmax(rewards)
            # defining the maximum reward
            max_reward = np.max(rewards)

            # update epsilon to decrease with the amount of times that this state-action pair has been reached
            # where we are considering the max action
            if self.times[tuple(indices+[grav_val])][max_action] > 0:
                epsilon /= self.times[tuple(indices+[grav_val])][max_action]

            # implement e-greedy:

            # if a random float is less than epsilon
            random = npr.rand() < epsilon
            if random:
                # jump with 50% probability and swing with 50% probability
                choice = npr.rand() < 0.5
                if choice:
                    new_action = 0
                else:
                    new_action = 1
            # if greater than epsilon, choose the action that maximizes expected future reward
            else:
                new_action = max_action

            # update eta to decrease with the amount of times that this state-action pair has been reached
            # where we are now considering the last action
            eta = self.eta
            if self.times[tuple(self.last_state+[grav_val])][self.last_action] > 0:
                eta /= self.times[tuple(self.last_state+[grav_val])][self.last_action]

            # update Q according to stochastic grad descent algorithm for middle of game
            self.Q[tuple(self.last_state+[grav_val])][self.last_action] += eta * (self.last_reward + self.gamma * max_reward - self.Q[tuple(self.last_state+[grav_val])][self.last_action])

        # update the amount of times we've seen this state-action pair
        self.times[tuple(indices+[grav_val])][new_action] += 1
        # update the last action to be the current action
        self.last_action = new_action
        # update the last state to be the indices of the current state
        self.last_state  = self.__discretize_array(values)
        # update the previous velocity
        self.prev_vel = values[2]

        # update that we are not in a new game if this flag is set to true
        if self.new_game:
            self.new_game = False

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

    # function that plots the scores during your training
    def plot_scores(self):
        print('Average score over all epochs: {}'.format(np.average(self.scores)))
        print('Max score over all epochs: {}'.format(np.max(self.scores)))

        # comment these lines out if you'd like to do less than 100 epochs
        print('Average score after first 100 epochs: {}'.format(np.average(self.scores[99:])))
        print('Max score after first 100 epochs: {}'.format(np.max(self.scores[99:])))

        plt.figure()
        plt.plot(np.arange(len(self.scores)), self.scores, 'o')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Scores with eta: {}, gamma: {}, epsilon: {}. Zero Initialization.'.format(self.eta, self.gamma, self.eps))
        plt.show()

# run the game
def run_games(learner, hist, coordinates, iters = 100, t_len = 100):
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

        # Save the location of the monkey at times of death
        coordinates.append(swing.coordinates) # swing.coordinates was added in our version of Swinging Monkey

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

    # Run games. 
    run_games(agent, hist, coordinates, epochs, 10)

    # view the scores
    agent.plot_scores()

    # plot the coordinates of the monkey at times of death
    plt.figure()
    plt.scatter(np.arange(len(coordinates)), coordinates)
    plt.ylim([-50,450])
    plt.xlabel('Epochs')
    plt.ylabel('Height at Death')
    plt.show()

    # Save history. 
    np.save('hist',np.array(hist))
    # Save the coordinates of the monkey at times of death
    np.save('coordinates',np.array(coordinates))
    


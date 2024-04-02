import math
import random

import numpy as np


class q_learning_cart:
    mdp=None
    epsilon=0.1
    alpha=0.2
    q_func = {}
    n=10
    intervals_delta = []
    min_max_values = []
    actions = []
    discount_factor = 0.9
    intervals = []

    def __init__(self, mdp, intervals_delta, min_max_values, actions, n=10, epsilon=0.1, alpha=0.2, discount_factor=0.9):
        self.mdp = mdp
        self.intervals_delta = intervals_delta
        self.min_max_values = min_max_values
        self.q_func = self.create_q_func(self.intervals_delta, self.min_max_values, actions)
        self.actions = actions
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount_factor = discount_factor


    def create_q_func(self, intervals_delta, min_max_values, actions):
        self.intervals = []
        for i in range(len(min_max_values)):
            self.intervals.append(np.floor(np.arange(min_max_values[i][0], min_max_values[i][1], intervals_delta[i])*100)/100)
        states = []
        for i in range(len(self.intervals[0])):
            for j in range(len(self.intervals[1])):
                for k in range(len(self.intervals[2])):
                    for l in range(len(self.intervals[3])):
                        states.append((self.intervals[0][i], self.intervals[1][j], self.intervals[2][k], self.intervals[3][l]))
        q_func = {(s,a): random.random() for a in actions for s in states}
        return q_func

    def greedy(self,s):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            self.getState(s)
            return max(self.actions, key=lambda a: self.q_func[(s,a)])

    def getState(self, s):
        int1 = math.floor(s[0] * 10) / 10
        int2 = math.floor(s[1] * 10) / 10
        int3 = math.floor(s[2] * 100) / 100
        int4 = math.floor(s[3] * 10) / 10
        # verify that we are in between the min and max values
        if int1 < self.intervals[0][0]:
            int1 = self.intervals[0][0]
        if int1 > self.intervals[0][-1]:
            int1 = self.intervals[0][-1]
        if int2 < self.intervals[1][0]:
            int2 = self.intervals[1][0]
        if int2 > self.intervals[1][-1]:
            int2 = self.intervals[1][-1]
        if int3 < self.intervals[2][0]:
            int3 = self.intervals[2][0]
        if int3 > self.intervals[2][-1]:
            int3 = self.intervals[2][-1]
        if int4 < self.intervals[3][0]:
            int4 = self.intervals[3][0]
        if int4 > self.intervals[3][-1]:
            int4 = self.intervals[3][-1]
        s2 = (int1, int2, int3, int4)
        return s2

    def solve(self):
        i=0
        episode_length = []
        while i<self.n:
            print("épisode :", i)
            state = self.getState(tuple(list(self.mdp.reset()[0])))
            initialstate = state
            done = False
            j=0
            while not done:
                action = self.greedy(state)
                next_state_con, reward, done, _, _ = self.mdp.step(action)
                next_state = self.getState(next_state_con)
                alpha_delta = self.alpha * self.get_delta(reward, self.q_func[(state, action)], state, next_state)
                self.q_func[(state, action)] += alpha_delta
                state = next_state
                j+=1
            print("fin de l'épisode en", j, "steps")
            print("initial state", self.q_func[(initialstate,0)], self.q_func[(initialstate,1)])
            self.alpha = self.alpha - self.alpha/self.n
            episode_length.append(j)
            i+=1
        return self.q_func, episode_length

    def get_delta(self, reward, q_value, state, next_state):
        return reward + self.discount_factor * max(self.q_func[(next_state, a)] for a in self.actions) - q_value
import random
from matplotlib import pyplot as plt

class q_agent:

    mdp=None
    epsilon=0.4
    alpha=0.2
    q_func = {}
    n=10

    def __init__(self, mdp, n=10):# and here...
        self.mdp = mdp
        self.q_func = {(s,a): 0.0 for a in self.mdp.get_actions() for s in self.mdp.get_states()}
        self.n= n

    def greedy(self,s):
        if random.random() < self.epsilon:
            return random.choice(self.mdp.get_actions(s))
        else:
            return max(self.mdp.get_actions(s), key=lambda a: self.q_func[(s,a)])

    def solve(self):
        i=0
        while i<self.n:
            state = self.mdp.get_initial_state()
            while not self.mdp.is_terminal(state):
                action = self.greedy(state)
                next_state, reward = self.mdp.execute(state, action)
                alpha_delta = self.alpha * self.get_delta(reward, self.q_func[(state, action)], state, next_state)
                self.q_func[(state, action)] += alpha_delta
                state = next_state
            i+=1
        return self.q_func
    def get_delta(self, reward, q_value, state, next_state):
        return reward + self.mdp.get_discount_factor() * max(self.q_func[(next_state, a)] for a in self.mdp.get_actions(next_state)) - q_value


    def state_value(self, state):
        #Get the value of a state
        return 0.0

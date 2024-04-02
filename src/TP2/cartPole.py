import gymnasium as gym
import numpy as np
import random


env = gym.make('CartPole-v1',render_mode="None")

print("action space:", env.action_space) #Discrete(n) : l'ensemble des actions est fini, et il y a n actions : 0,...,n-1
print("observation space:", env.observation_space) #Box(4,) : l'ensemble des observations est continu, et il y a 4 observations

def custom_policy(state):
    tab=[0,1]
    return random.choice(tab)

initial_state=env.reset()
# done=False
# new_state=initial_state
# while not done:
#     policy = custom_policy(new_state)
#     new_state,reward,done,_,_ = env.step(policy)
#     print("policy,new_state,reward,done",policy,new_state,reward,done)

import q_learning_cart as qlc

intervals_delta = [0.1, 0.1, 0.01, 0.1]
min_max_values = [(-2.4, 2.4), (-3, 3), (-0.21, 0.21), (-3, 3)]
actions = [0, 1]
n=40000
epsilon=0.05
alpha=0.5
discount_factor=0.99

q_agent = qlc.q_learning_cart(env, intervals_delta, min_max_values, actions, n, epsilon, alpha, discount_factor)
q_func, episode_length = q_agent.solve()

env.close()

import matplotlib.pyplot as plt
import numpy as np

# make a bar chart for episode length (count the number of occurences)
plt.hist(episode_length, bins=max(episode_length))
plt.xlabel('Episode length')
plt.ylabel('Count')
plt.title('Episode length distribution')
plt.show()

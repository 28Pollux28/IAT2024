from gridworld import *

mdp = GridWorld(noise=0.1)

print(" states :", mdp.get_states())
print(" terminal states :", mdp.get_goal_states())
print(" actions :", mdp.get_actions())
print(mdp.get_transitions(mdp.get_initial_state(), mdp.UP))

class policy:
    q_func = {}
    actions = mdp.get_actions()

    def __init__(self, q_func):
        self.q_func = q_func

    def select_action(self, state):
        best_action = self.actions[0]
        best_value = -float("inf")
        for action in self.actions:
            value = q_func[(state, action)]
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

class q_function:
    def __init__(self, q_func):
        self.q_func = q_func

    def get_q_value(self, s, a):
        return self.q_func[(s, a)]


def policy_custom(state, q_func, mdp):
    actions = mdp.get_actions()
    best_action = actions[0]
    best_value = -float("inf")
    for action in actions:
        value = q_func[(state, action)]
        if value > best_value:
            best_value = value
            best_action = action
    return best_action



import q_learning as ql


for n in [10, 100, 1000, 10000, 100000]:
    q_agent = ql.q_agent(mdp,n)

    q_func = q_agent.solve()
    print("Q function:")
    states = mdp.get_states()
    actions = mdp.get_actions()
    for s in states:
        for a in actions:
            print(f"Q({s},{a}) = {q_func[(s,a)]}")
    mdp.visualise_q_function(q_function(q_func))
    policy_custom = policy(q_func)

    mdp.visualise_policy(policy_custom, q_func)



# while (1):
#     state = mdp.get_initial_state()
#     new_state, _ = mdp.execute(state, policy_custom(state, v, mdp))
#     mdp.initial_state = new_state
#     mdp.visualise()
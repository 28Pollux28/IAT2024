from gridworld import *

mdp = GridWorld(noise=0.3, width=50,height=50,blocked_states=[(1,1),(5,3)], goals=[((50 - 1, 50 - 1), 1), ((50 - 1, 50 - 2), -1), ((6,2),-2)])

print(" states :", mdp.get_states())
print(" terminal states :", mdp.get_goal_states())
print(" actions :", mdp.get_actions())
print(mdp.get_transitions(mdp.get_initial_state(), mdp.UP))

class value_function:
    def __init__(self, v):
        self.v = v

    def get_value(self, s):
        return self.v[s]


class policy:
    v = {}
    actions = mdp.get_actions()

    def __init__(self, v):
        self.v = v

    def select_action(self, state):
        best_action = self.actions[0]
        best_value = -float("inf")
        for action in self.actions:
            value = sum(
                p * mdp.get_reward(state, action, next_state) + mdp.get_discount_factor() * self.v[next_state] * p for next_state, p in mdp.get_transitions(state, action))
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

def policy_custom(state, v, mdp):
    actions = mdp.get_actions()
    best_action = actions[0]
    best_value = -float("inf")
    for action in actions:
        value = sum(p * mdp.get_reward(state, action, next_state) + mdp.get_discount_factor() * v[next_state] * p for next_state, p in mdp.get_transitions(state, action))
        if value > best_value:
            best_value = value
            best_action = action
    return best_action



import dynamic_programming as dp

dp_agent = dp.dp_agent(mdp)

v = dp_agent.solve()
print("Value function:")
print(v)
states = mdp.get_states()
# for s in states:
#     print(f"v({s}) = {v[s]}")

mdp.visualise_value_function(value_function(v))
mdp.visualise_policy(policy(v))
# while (1):
#     state = mdp.get_initial_state()
#     new_state, _ = mdp.execute(state, policy_custom(state, v, mdp))
#     mdp.initial_state = new_state
#     mdp.visualise()

# mdp.execute_policy(policy_custom,10)
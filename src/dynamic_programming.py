from mdp import *


class dp_agent():
    mdp=None
    ''' add attributes here! '''
    
    def __init__(self,mdp): #and here...
        self.mdp=mdp

    def get_value(self,s,v):
        #return the value of a specific state s according to value function v
        return v[s]
        
    def get_width(self,v,v_bis):
        return max(abs(v[i]-v_bis[i]) for i in self.mdp.get_states()[1:])

    def solve(self):
        v = {s:0 for s in self.mdp.get_states()}
        v_bis = {s:5 for s in self.mdp.get_states()}

        while self.get_width(v,v_bis) > 1e-5:
            v_bis = v.copy()
            v_func = lambda s: max(sum(p*self.mdp.get_reward(s,a,s_bis) for s_bis, p in self.mdp.get_transitions(s,a)) +sum(self.mdp.get_discount_factor() * v_bis[s_bis] *p for (s_bis,p) in self.mdp.get_transitions(s,a)) for a in self.mdp.get_actions())
            for s in self.mdp.get_states():
                v[s] = v_func(s)
            print(v, v_bis)
        return v
    def update(self,s):
        pass
        #updates the value of a specific state s




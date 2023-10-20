import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np
import pulp

class compute_V:
    def __init__(self, mdp_path, algo) -> None:
        self.mdp = mdp_path
        self.algo = algo
        self.data = {}
    def reading(self):
        with open(self.mdp, 'r') as file:
            for line in file:
                line  = line.strip()
                if not line:
                    continue
                parts = line.split()
                key = parts[0]
                if key == 'numStates':
                    self.S = int(parts[1])
                elif key == 'numActions':
                    self.A = int(parts[1])
                elif key == 'end':
                    self.data[key] = int(parts[1])
                elif  key == 'transition':
                    s = int(parts[1])
                    a = int(parts[2])
                    s_dash = int(parts[3])
                    r = float(parts[4])
                    p = float(parts[5])
                    if key not in self.data:
                        self.data[key] = []
                    self.data[key].append([s,a,s_dash,r,p])
                elif  key == 'mdptype':
                    self.data[key] = parts[1]
                elif key =='discount':
                    self.gamma = float(parts[1])
        self.trans_probs = np.zeros((self.S, self.A, self.S))
        self.rewards = np.zeros((self.S, self.A, self.S))
        for transition in self.data['transition']:
            s = transition[0] ; a = transition[1] ; s_next = transition[2]
            self.trans_probs[s,a,s_next] = transition[4]
            self.rewards[s,a,s_next] = transition[3]
        if self.algo == 'vi':
            self.valueIteration()
        elif self.algo == 'lp':
            self.linearProgram()
        elif self.algo == 'hpi':
            self.howardPolicy()
# ---------------Value Iteration
    def bellman_vi(self, V):
        V_next = np.zeros_like(V)
        pi = np.zeros(self.S, dtype=int)
        for s in range(self.S):
            value = float('-inf')
            for a in range(self.A):
                val = np.sum( self.trans_probs[s,a] * (self.rewards[s,a] + self.gamma*V) )
                value = max(val, value)         
                if abs(value-val) < 1e-2:
                    pi[s] = a   
            V_next[s] = value
        return V_next, pi
    def valueIteration(self):
        prev=  np.ones(self.S)
        while True:
            V , pi= self.bellman_vi(prev)
            if np.all( np.abs(prev - V) < 1e-7):
                break
            prev[:] = V[:]
        for s in range(self.S):
            print(V[s], '', pi[s], '\n')

# -----------Horward Policy Iteration  
    def value_for_a_policy(self, policy):
        V = np.zeros(self.S)
        while True:
            V_next = np.zeros(self.S)
            for s in range(self.S):
                action = policy[s]
                V_next[s] = np.sum( self.trans_probs[s,action] * (self.rewards[s,action] + self.gamma*V) )
            if np.all(np.abs(V_next -V) < 1e-7):
                break
            else:
                V[:] = V_next[:]
        return V
    def Q_value(self, s,a, V): # equivalue to Q^pi(s,a) in slides
        return np.sum( self.trans_probs[s,a] * (self.rewards[s,a] + self.gamma*V) )
    def howardPolicy(self):     
        policy_prev = np.zeros(self.S, dtype=int)
        while True:
            policy = np.copy(policy_prev)
            V = self.value_for_a_policy(policy)
            for s in range(self.S):
                for action in [a for a in range(self.A) if a!=policy[s]]:
                    if self.Q_value(s, action, V) > V[s]:
                        policy[s] = action
            if np.all(policy_prev == policy):
                break
            policy_prev[:] = policy[:]
        with open('output_test_1.txt', 'w') as file:
            for s in range(self.S):
                print(round(self.value_for_a_policy(policy)[s], 6), '', policy[s], '\n' )
                file.write(f'{round(self.value_for_a_policy(policy)[s], 6)} {policy[s]}\n')

# --------------Linear Programming               
    def linearProgram(self):
        lp_problem = pulp.LpProblem("MDP_Optimization", pulp.LpMaximize)
        V = {}
        for s in range(self.S):
            V[s] = pulp.LpVariable(f"V_{s}", lowBound=None)

        lp_problem += -1*pulp.lpSum([V[s] for s in range(self.S)])
        # Define the constraints (Bellman equations)
        for s in range(self.S):
            for a in range(self.A):
                lp_problem += V[s] >= pulp.lpSum(self.trans_probs[s][a][s_next] * (self.rewards[s][a][s_next] + self.gamma * V[s_next]) for s_next in range(self.S))
        pulp.LpSolverDefault.msg = 0
        lp_problem.solve()

        if lp_problem.status == pulp.LpStatusOptimal:
            optimal_values = {s: V[s].varValue for s in range(self.S)}
            for s, value in optimal_values.items():
                policy = np.argmax( np.array([ np.sum([ self.trans_probs[s][a][s_next] * (self.rewards[s][a][s_next] + self.gamma * V[s_next].varValue) for s_next in range(self.S)]) for a in range(self.A)]))
                print(f"{value}",'', f"{policy}", '\n' )   
        else:
            print("LP problem did not reach an optimal solution.")

if __name__ == "__main__":
    parser.add_argument("--mdp",type=str,default='./data/mdp/episodic-mdp-2-2.txt')
    parser.add_argument("--algorithm",type=str,default='hpi')
    # parser.add_argument("--file", type=str, default='print')
    args = parser.parse_args()

    if not (args.algorithm in ['vi', 'hpi', 'lp']):
        print("Algorithm should be one of vi, hpi, lp")
        sys.exit(0)
    algo = compute_V(args.mdp, args.algorithm)
    algo.reading()
    # if args.file == 'print':
    #     algo.reading()
    # else:
    #     original_stdout = sys.stdout
    #     with open( args.file, "w") as file:
    #         sys.stdout = file

    #     algo.reading()
    #     sys.stdout = original_stdout
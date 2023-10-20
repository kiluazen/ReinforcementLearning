import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np

class PlannerEncoder:
    def __init__(self, opponent, p,q) -> None:
        self.p = p; self.q = q

        self.idx_to_states = {}
        self.opp_action_probs = {}
        with open(opponent,'r') as file:
            i = 0
            for line in file:
                parts = line.split()
                if parts[0] == 'state':
                    continue
                if len(parts[0]) == 7:
                    self.idx_to_states[i] = parts[0]
                    self.opp_action_probs[parts[0]] = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                i+=1
        self.idx_to_states[i] = 'lost'   # both of these are terminal states
        self.idx_to_states[i+1] = 'goal'
        self.states_to_idx = {}
        for i in self.idx_to_states:
            self.states_to_idx[self.idx_to_states[i]] = i
        self.S = len(self.idx_to_states)
        self.A = 10
        # Next step is to calculate probs based on different situations
    def player_pos(self, player, action):
            new = None
            if action ==0:
                new = player -1
                if (new-1)//4 == (player-1)//4 and new > 0 and new < 17:
                    player = new
            elif action == 1:
                new = player +1
                if (new-1)//4 == (player-1)//4 and new > 0 and new < 17:
                    player = new
            elif action ==2:
                new = player - 4
                if new > 0 and new < 17:
                    player = new
            elif action ==3:
                new = player + 4
                if new > 0 and new < 17:
                    player = new
            return player
    def state_after_action(self, curr_state, a):
        b1_int = int(curr_state[:2])
        b2_int = int(curr_state[2:4])
        r_int = int(curr_state[4:6])
        ball_int = int(curr_state[-1])      
        if a <4:
            b1_int = self.player_pos(b1_int, a)
        elif a < 8:
            b2_int = self.player_pos(b2_int, a - 4)
        elif a == 8:
            if ball_int ==1:
                ball_int = 2
            elif ball_int ==2:
                ball_int = 1
        elif a == 9:
            return 'goal'
        
        b1_str = str(b1_int) ; b2_str = str(b2_int)
        r_str = str(r_int)
        ball_str = str(ball_int)
        if len(b1_str)==1:
            b1_str = '0' + b1_str
        if len(b2_str)==1:
            b2_str = '0' + b2_str
        if len(r_str)==1:
            r_str = '0' + r_str
        new_state = b1_str + b2_str + r_str + ball_str
        return new_state
    
    def cordinates(self, state):
        b1 = int(state[:2]); b2 = int(state[2:4]); r = int(state[-3:-1])
        b1_cor = ( (b1 -1)//4 , (b1-1)%4 )
        b2_cor = ( (b2 -1)//4 , (b2-1)%4 )
        r_cor = ( (r -1)//4 , (r-1)%4 )
        return [b1_cor, b2_cor, r_cor]
    def transition_function(self, current_s, next_s, action):
        ball_pos = int(current_s[-1])
        if action <4:
            if ball_pos == 1:
                b1_old = current_s[:2] ; r_old = current_s[-3:-1]
                b1_new = next_s[:2] ; r_new = next_s[-3:-1]
                if b1_new == r_new:
                    return (0.5 - self.p, 0.5 + self.p)
                elif b1_old == r_new and b1_new == r_old:
                    return (0.5 - self.p, 0.5 + self.p)
                else:
                    return (1- self.p, self.p)
            elif ball_pos == 2:
                return (1- self.p, self.p)
        elif action <8:
            if ball_pos == 1:
                return (1-self.p, self.p)
            elif ball_pos == 2:
                b2_old = current_s[2:4] ; r_old = current_s[-3:-1]
                b2_new = next_s[2:4] ; r_new = next_s[-3:-1]
                if b2_new == r_new:
                    return (0.5 - self.p, 0.5 + self.p)
                elif b2_old == r_new and b2_new == r_old:
                    return (0.5 - self.p, 0.5 + self.p)
                else:
                    return (1- self.p, self.p)
        if action ==8:
            b1_cor, b2_cor, r_cor = self.cordinates(next_s)
            val = self.q - 0.1*max( abs(b1_cor[0] - b2_cor[0]), abs(b1_cor[1] - b2_cor[1]))
            if b1_cor[0] == r_cor[0] and b2_cor[0] == r_cor[0]:
                return (0.5*val, 1 - 0.5*val)
            elif b1_cor == r_cor or b2_cor == r_cor:
                return (0.5*val, 1 - 0.5*val)
            elif ((b1_cor[1]- r_cor[1])/(b1_cor[0] - r_cor[0] + 1e-3)) == ((r_cor[1] - b2_cor[1])/(r_cor[0]- b2_cor[0] + 1e-3)):
                return (0.5*val, 1 - 0.5*val)
            else:
                return (val, 1- val)
        if action ==9:
            b1_cor, b2_cor, r_cor = self.cordinates(next_s)
            ball_pos = int(current_s[-1])
            if ball_pos ==1:
                val = self.q - 0.2*(3 - b1_cor[1]) # NOTE my x,y are reverse to the one used in the assgn description
                # I use like the matrix 0,1 axis
                if r_cor[0]>0 and r_cor[0]<3 and r_cor[1]>1:
                    return (0.5*val, 1- 0.5*val)
                else:
                    return( val, 1-val)
            elif ball_pos ==2:
                val = self.q - 0.2*(3 - b2_cor[1]) # NOTE 
                if r_cor[0]>0 and r_cor[0]<3 and r_cor[1]>1:
                    return (0.5*val, 1- 0.5*val)
                else:
                    return( val, 1-val)

    def calculate_trans_probs(self):
        self.trans_probs = np.zeros((self.S, self.A, self.S))
        for s in range(self.S - 2): # we don't start from lost and goal state
            current_s = self.idx_to_states[s]
            for a in range(self.A):
                if a <9:
                    new_state = self.state_after_action(current_s, a)
                    if new_state != current_s:
                        r_int = int(current_s[-3:-1])
                        for i, prob_opp in enumerate(self.opp_action_probs[current_s]):
                            # now for the current_s you will get a reaction from the opponent
                            if prob_opp !=0:
                                r_str = str(self.player_pos(r_int, i)) # 'i' would give the action for R                            
                                if len(r_str)==1:
                                    r_str = '0' + r_str # NOTE: I hope the prob's are zero when the R is at the edge
                                next_s = new_state[:4] + r_str + new_state[-1]
                                # Now let's call a helper function to give prob. # It looks if there is tackling or intergecting etc... 
                                # it's inputs would be current_s and next_s and the action taking place.
                                prob_s, prob_f = self.transition_function(current_s, next_s, a)
                                self.trans_probs[self.states_to_idx[current_s], a, self.states_to_idx[next_s]] = prob_opp*prob_s
                                self.trans_probs[self.states_to_idx[current_s], a, self.states_to_idx['lost']] = prob_opp*prob_f
                    else:
                        self.trans_probs[self.states_to_idx[current_s], a, self.states_to_idx['lost']] = 1 # regardless of what R does if you take a non feasible action then lossing is 1

                elif a ==9: # this has to be separate because state_after_action function gives 'goal' for this so u can't slice like before.
                    new_state = current_s[:]
                    for i, prob_opp in enumerate(self.opp_action_probs[current_s]):
                            if prob_opp != 0:
                                r_int = int(current_s[-3:-1])
                                r_str = str(self.player_pos(r_int, i)) # 'i' would give the action for R                            
                                if len(r_str)==1:
                                    r_str = '0' + r_str # NOTE: I hope the prob's are zero when the R is at the edge
                                next_s = new_state[:4] + r_str + new_state[-1]
                                
                                prob_s, prob_f = self.transition_function(current_s, next_s, a)
                                self.trans_probs[self.states_to_idx[current_s], a, self.states_to_idx['goal']] = prob_opp*prob_s
                                self.trans_probs[self.states_to_idx[current_s], a, self.states_to_idx['lost']] = prob_opp*prob_f
        self.rewards = np.zeros((self.S, self.A, self.S))
        self.rewards[:,:,8192] = -1
        self.rewards[:,:,8193] = 1

    def save_transition_probabilities_and_rewards(self, filename):
        self.calculate_trans_probs()
        trans_probs = self.trans_probs
        rewards = self.rewards
        num_states, num_actions, _ = trans_probs.shape

        with open(filename, 'w') as file:
            file.write(f"numStates {num_states}\n")
            file.write(f"numActions {num_actions}\n")
            file.write("end 8192 8193\n")

            for s in range(num_states - 2):  # Exclude terminal states 'lost' and 'goal'
                for a in range(num_actions):
                    for s_prime in range(num_states):
                        prob = trans_probs[s, a, s_prime]
                        reward = rewards[s, a, s_prime]

                        if prob != 0 or reward != 0:
                            file.write(f"transition {s} {a} {s_prime} {prob} {reward}\n")

            file.write("mdptype episodic\n")
            file.write("discount 0.9\n")

            # Example usage:


if __name__ == "__main__":
    parser.add_argument("--opponent",type=str,default='./data/football/test-1.txt')
    parser.add_argument("--p", type=float)
    parser.add_argument("--q", type=float)
    
    args = parser.parse_args()

    if not (args.p <=1.0 and args.p >=0.0):
        print("p is a probability, should be btw 0,1")
        sys.exit(0)
    if not (args.q<=1.0 and args.q >=0.0):
        print("q is a probability, should be btw 0,1")
        sys.exit(0)

    enco = PlannerEncoder(args.opponent, args.p, args.q)
    enco.save_transition_probabilities_and_rewards('t-2.txt')
    
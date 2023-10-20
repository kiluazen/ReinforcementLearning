"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need
def kl(p,q):
    return p*math.log(p/q + 1e-6) + (1-p)*math.log((1-p)/(1-q))
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        # END EDITING HERE
        self.counts = np.zeros(num_arms); self.values = np.zeros(num_arms)
        self.ucb_values = np.zeros(num_arms)
    
    def give_pull(self):
        # START EDITING HERE
        # raise NotImplementedError
        total_samples = np.sum(self.counts)
        if total_samples < 1:
            return np.random.randint(self.num_arms)
        else:
            for bandit in range(self.num_arms):
                self.ucb_values[bandit] = self.values[bandit] + np.sqrt(2*math.log(total_samples)/(self.counts[bandit] + 1e-6 ) )
            return np.argmax(self.ucb_values)
        # END EDITING HERE  

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # raise NotImplementedError
        # assert arm_index == self.give_pull(), 'Not giving the correct arm to pull!'
        self.counts[arm_index] +=1 
        n = self.counts[arm_index] ; prev_value = self.values[arm_index]
        new_value = ((n-1)*prev_value + reward)/n
        self.values[arm_index] = new_value
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.kl_ucb = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # raise NotImplementedError
        t = np.sum(self.counts) ; c =3
        if t<4:
            return np.random.randint(self.num_arms)
        else:
            for bandit in range(self.num_arms):
                # Have to find q, using binary search; q belong to [p,1]
                if self.counts[bandit] <1: # Making sure counts are atleast 1
                    return bandit 
                rhs_value = ( math.log(t) + c*math.log(math.log(t)) )/(self.counts[bandit])
                lower_q = self.values[bandit] ; upper_q = 1.0
                num_iterations = 0
                while upper_q - lower_q > 1e-4 and num_iterations < 1000:
                    mid = (lower_q + upper_q)/2
                    if kl(self.values[bandit], mid) > rhs_value:
                        upper_q = mid
                    else:
                        lower_q = mid
                    num_iterations +=1
                # final q after binary search
                q = (upper_q + lower_q)/2
                self.kl_ucb[bandit] = q
            return np.argmax(self.kl_ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # raise NotImplementedError
        self.counts[arm_index] +=1 ; n = self.counts[arm_index]
        prev_value = self.values[arm_index]
        new_value = ((n-1)*prev_value + reward) /n
        self.values[arm_index] = new_value
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.thompson_sampling = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # raise NotImplementedError
        for bandit in range(self.num_arms):
            s = self.counts[bandit] * self.values[bandit]
            f = self.counts[bandit] - s
            sample = np.random.beta(s+1, f+1)
            self.thompson_sampling[bandit] = sample
        return np.argmax(self.thompson_sampling)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # raise NotImplementedError
        self.counts[arm_index] +=1 ; n = self.counts[arm_index]
        prev_value = self.values[arm_index]
        new_value = ((n-1)*prev_value + reward) /n
        self.values[arm_index] = new_value
        # END EDITING HERE

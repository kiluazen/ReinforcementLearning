"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
import math
# END EDITING HERE


class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        # START EDITING HERE
        self.counts = np.zeros((2, num_arms))
        self.values = np.zeros((2,num_arms))
        self.thompson_sampling = np.zeros((2,num_arms))
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # raise NotImplementedError
        i = np.random.binomial(1, 0.5)
        for bandit in range(self.num_arms):
            s = self.counts[i][bandit] * self.values[i][bandit]
            f = self.counts[i][bandit] - s
            sample = np.random.beta(s+1, f+1)
            self.thompson_sampling[i][bandit] = sample
        return np.argmax(self.thompson_sampling[i])

        # END EDITING HERE
    
    def get_reward(self, arm_index, set_pulled, reward):
        # START EDITING HERE
        # raise NotImplementedError
        self.counts[set_pulled][arm_index] +=1 
        n = self.counts[set_pulled][arm_index] ; prev_value = self.values[set_pulled][arm_index]
        new_value = ((n-1)*prev_value + reward)/n
        self.values[set_pulled][arm_index] = new_value
        # END EDITING HERE


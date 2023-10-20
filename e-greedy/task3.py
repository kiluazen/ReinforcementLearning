"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np
# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault # probability that the bandit returns a faulty pull
        # START EDITING HERE
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)
        self.thompson_sampling = np.zeros(self.num_arms)
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
        # self.counts[arm_index] += 1
        # n = self.counts[arm_index]
        # prev_value = self.values[arm_index]
        # # When the reward is faulty, we give the reward as it's current so far bandit value.
        # reward_with_fault = (1 - self.fault)* reward + self.fault*prev_value
        # new_value = ((n - 1) / n) * prev_value + (1 / n) * reward_with_fault
        # self.values[arm_index] = new_value
        fault_or_not = np.random.binomial(1, 1- self.fault)
        if fault_or_not == 1:
            self.counts[arm_index] +=1 ; n = self.counts[arm_index]
            prev_value = self.values[arm_index]
            new_value = ((n-1)*prev_value + reward) /n
            self.values[arm_index] = new_value
        else:
            self.counts[arm_index] +=1
            # We don't change the value because it's a faulty reward.
        #END EDITING HERE


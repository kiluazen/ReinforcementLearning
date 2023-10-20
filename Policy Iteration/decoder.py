import argparse,sys
parser = argparse.ArgumentParser()
import numpy as np

class Decode:
    def __init__(self, value_file, opponent_file) -> None:
        self.idx_to_states = {}
        with open(opponent_file,'r') as file:
            i = 0
            for line in file:
                parts = line.split()
                if parts[0] == 'state':
                    continue
                if len(parts[0]) == 7:
                    self.idx_to_states[i] = parts[0]
            i+=1
        self.action_value = {}
        with open(value_file, 'r') as file:
            i= 0
            for line in file:
                parts = line.split()
                value = float(parts[0])
                action = int(parts[1])
                self.action_value[i] = [action, value]
            i+=1
        with open('policyfile.txt', 'w') as file:
            for s in self.action_value:
                file.write(f'{self.idx_to_states[s]} {self.action_value[0]} {self.action_value[1]}\n')

if __name__ == "__main__":
    parser.add_argument("--value-policy",type=str,default='./t-1.txt')
    parser.add_argument("--opponent",type=str,default='./data/football/test-1.txt')
    
    args = parser.parse_args()
    
    deco = Decode(args.value-policy, args.opponent)
    
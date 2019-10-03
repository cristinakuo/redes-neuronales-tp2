from perceptron import SimplePerceptron
from utils import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():
    
    N = 2
    P = 4*N

    capacities = []

    #for p in range(2,P):
    for p in range(1,P):
        patterns = get_random_patterns(p,N)
        desired_y_permutations = get_all_permutations(p)
        num_learned_patterns = 0
        
        for y_d in desired_y_permutations:
            myPercep = SimplePerceptron(patterns,y_d)
            learned = myPercep.train()
            if learned:
                num_learned_patterns += 1
                
        capacities.append(num_learned_patterns/(len(desired_y_permutations)))

    print("Las capacidades: {}".format(capacities))    




if __name__ == '__main__':
    main()
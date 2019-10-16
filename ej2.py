from SimplePerceptron import SimplePerceptron
from utils import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

def main():
    
    N = 3   # Number of inputs
    P = 4*N # Number of patterns

    capacities_over_total_perms = []

    # Determine Capacity/(2^P) for a range of number of patterns
    log.info("Obtaining capacities for a range of {} patterns...".format(P))
    for p in tqdm(range(1,P+1)):
        patterns = get_random_patterns(p,N)
        desired_y_permutations = get_all_permutations(p)
        
        num_learned_patterns = 0
        num_permutations = len(desired_y_permutations)
        
        # Using the same patterns, train with all possible desired outputs
        for y_d in desired_y_permutations:
            myPercep = SimplePerceptron(N,1)
            myPercep.setPatterns(patterns,y_d)
            learned = myPercep.train()
            if learned:
                num_learned_patterns += 1

        capacity_over_total_perms = np.true_divide(num_learned_patterns,num_permutations)    
        capacities_over_total_perms.append(capacity_over_total_perms)

    
    log.info("Capacities obtained: {}".format(capacities_over_total_perms))
    capacities_over_total_perms = np.array(capacities_over_total_perms)

    plt.plot(np.array(range(1,P+1))/N,capacities_over_total_perms)
    plt.xlabel("P/N")
    plt.ylabel("C(P,N)/2^P")    
    plt.show()

if __name__ == '__main__':
    main()
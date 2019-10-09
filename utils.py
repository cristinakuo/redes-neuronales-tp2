import numpy as np
from itertools import product
import sys
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

def sgn(n,ref=0):
		return 1 if n>=ref else -1

def get_random_patterns(num_of_patterns, num_of_inputs):
    res = np.random.uniform(-1,1, (num_of_inputs, num_of_patterns))
    if type(res) == 'int':
        res = np.array([res])
    return res
    
def get_all_permutations(num_of_patterns):
    all_perms = []
    permutations_tuples = list(product([1,-1],repeat=num_of_patterns))
    
    for perm in permutations_tuples:
        all_perms.append(np.array(perm))

    if len(all_perms) != np.power(2,num_of_patterns):
         log.error("Permutations not doing right. Bye.")
         sys.exit(1)

    return all_perms
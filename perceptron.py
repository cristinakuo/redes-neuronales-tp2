import numpy as np
import logging
import matplotlib.pyplot as plt
from utils import *
from enum import Enum

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

MAX_ITERATIONS = 100 

class SimplePerceptron():
    def __init__(self, x, y, etha=0.1):
        self.input_dim,_ = x.shape
        self.num_patterns = y.size
        self.x = x
        self.y_desired = y
        self.y_trained = np.zeros(self.num_patterns) 
        self.W = np.random.rand(self.input_dim)
        self.etha = etha # Learning rate

    def train(self):
        iter_counter = 0
        train_success = False
        timed_out = False

        # Train until trained output is equal to desired output or until maximum number of iterations is reached
        while True:
            # Pick random sample at a time
            iter_counter += 1
            for sample_i in np.random.permutation(self.num_patterns):  
                self.y_trained[sample_i] = sgn(np.dot(self.W,self.x[:,sample_i]),0)
                delta_W = self.etha * self.x[:,sample_i] * (self.y_desired[sample_i]-self.y_trained[sample_i])
                
                self.W = self.W + delta_W
                
            
            if np.array_equal(self.y_trained, self.y_desired):
                train_success = True
                break
            elif iter_counter > MAX_ITERATIONS:
                timed_out = True
                break
            
        return train_success    
            

if __name__ == '__main__':
    
    # AND
    x = np.array([[-1,-1, 1, 1],[-1,-1,-1,1],[1,1,1,1]])
    y = np.array([-1,-1,-1,1]) 

    percep = SimplePerceptron(x,y)
    percep.train()
    print(percep.W)
    print("trained y is: {}".format(percep.y_trained))

    plt.plot()




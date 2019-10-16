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
        
        self.W = np.zeros(self.input_dim)
        self.etha = etha # Learning rate

    def train(self):
        iter_counter = 0
        train_success = False

        # Initialize weight vector in random numbers
        self.W = np.random.rand(self.input_dim) 
        y_trained = np.zeros(self.num_patterns) 
        # Train until trained output is equal to desired output or until maximum number of iterations is reached
        while True:
            # Pick random pattern at a time
            iter_counter += 1
            for p in np.random.permutation(self.num_patterns):  
                y_trained[p] = sgn(np.dot(self.W,self.x[:,p]),0)
                delta_W = self.etha * self.x[:,p] * (self.y_desired[p]-y_trained[p])
                
                self.W = self.W + delta_W
                
            
            if np.array_equal(y_trained, self.y_desired):
                train_success = True
                break
            elif iter_counter > MAX_ITERATIONS:
                break
            
        return train_success    
    
    
    def get_output(self, x_input): # Expects a list or an array
        # Convert to numpy array
        x_input = np.array(x_input)

        # Validate input dimension is correct
        input_rows = x_input.shape[0] 
        if input_rows != len(self.W):
            log.error("Input dimension {} doesnt match weight matrix dimension {}.".format(x_input.shape,self.W.shape))
            sys.exit(1)
        
        result = np.dot(self.W, x_input)
        result = np.vectorize(sgn)(result)
        return result           

if __name__ == '__main__':
    
    # AND
    x = np.array([[-1,-1, 1, 1],[-1,-1,-1,1],[1,1,1,1]])
    y = np.array([-1,-1,-1,1]) 

    percep = SimplePerceptron(x,y)
    percep.train()
    log.info("Output obteined from trained perceptron: {}".format( percep.get_output(x) ) )

    plt.plot()




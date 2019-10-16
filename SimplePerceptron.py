import numpy as np
import logging
import matplotlib.pyplot as plt
from utils import *
from enum import Enum

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

MAX_ITERATIONS = 100 
DEFAULT_LEARNING_RATE = 0.1

class SimplePerceptron():
    def __init__(self, input_dimension, output_dimension, etha = DEFAULT_LEARNING_RATE):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.learning_rate = etha

    def setPatterns(self,input_patterns, desired_outputs):
        rows,cols = input_patterns.shape
        if self.input_dimension != rows:
            raise Exception("Patterns input dimension {} does not match Perceptron input dimensions {}.".format(rows,self.input_dimension))
        
        self.patterns_input = input_patterns
        self.patterns_output = desired_outputs
        self.weight_matrix = np.zeros(self.input_dimension)
        self.num_patterns = cols

#    def init(self, x, y):
#        self.input_dim,_ = x.shape
#        self.num_patterns = y.size
#        self.x = x
#        self.y_desired = y
#        
#        self.W = np.zeros(self.input_dim)
#        self.etha = etha # Learning rate

    def train(self):
        iter_counter = 0
        train_success = False

        # Initialize weight vector in random numbers
        self.W = np.random.rand(self.input_dimension)
        
        # Train until trained output is equal to desired output or until maximum number of iterations is reached
        while True:
            # Pick random pattern at a time
            iter_counter += 1
            
            for p in np.random.permutation(self.num_patterns):  
                input_pattern = self.patterns_input[:,p]
                output_pattern = self.patterns_output[p]
        
                actual_output = sgn(np.dot(self.weight_matrix,self.patterns_input[:,p]),0)

                delta_W = self.learning_rate * input_pattern * (output_pattern-actual_output)
                
                self.W = self.W + delta_W

            expectation = self.patterns_output    
            reality = self.get_output(self.patterns_input)

            if np.array_equal(expectation,reality):
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

    perceptron = SimplePerceptron(3,1)
    perceptron.setPatterns(x,y)

    isSuccessful = perceptron.train()

    if isSuccessful:
        print("TRAIN SUCCESS")
    else:
        print("TRAIN FAILURE")
    log.info("Output obteined from trained perceptron: {}".format( perceptron.get_output(x) ) )



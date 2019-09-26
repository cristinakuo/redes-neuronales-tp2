import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

def sgn(n,ref=0):
		return 1 if n>=ref else -1

class SimplePerceptron():
    def __init__(self, x, y, etha=0.1):
        self.input_dim,_ = x.shape
        self.num_samples = y.size
        self.x = np.vstack([x, [ np.ones(self.num_samples)]] ) # Add row of ones
        self.y_desired = y
        self.y_trained = np.zeros(self.num_samples) 
        self.W = np.random.rand(self.input_dim+1)
        self.etha = etha # Learning rate

    def train(self):
        iter_counter = 0
        while not np.array_equal(self.y_trained,self.y_desired):
            # Pick random sample at a time
            
            for sample_i in np.random.permutation(self.num_samples):  
                self.y_trained[sample_i] = np.vectorize(sgn)(np.dot(self.W,self.x[:,sample_i]),0)
                self.W = self.W + self.etha * self.x[:,sample_i] * (self.y_desired[sample_i]-self.y_trained[sample_i])
                # TODO: no guardarlo de una
                # DEBUG
                print("W is {}".format(self.W))
                print("x[:,i] is {}".format(x[:,sample_i]))
                print("y_trained is {}".format(self.y_trained[sample_i]))
                
            iter_counter += 1

        log.info("Total iterations: {}".format(iter_counter))    
            

if __name__ == '__main__':
    
    x = np.array([[-1,-1, 1, 1],[-1,-1,-1,1]])
    y = np.array([-1,-1,-1,1])
    percep = SimplePerceptron(x,y)
    percep.train()
    print(percep.W)
    print("trained y is: {}".format(percep.y_trained))

    plt.plot()




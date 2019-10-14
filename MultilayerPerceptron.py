import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

class MultiLayerPerceptron():   
    def __init__(self, x, y, layers, layers_dim):
        self.x = x
        self.y = y
        self.input_dim,_ = x.shape
        self.output_dim = 1 # TODO: this is hardcoded
        self.num_patterns = len(y)
        self.layers = [] # Includes input layer
        self.layers.append(np.zeros(self.input_dim))
        self.etha = 0.95 # Learning rate
        
        for i in range(layers-1):
            self.layers.append(np.zeros(layers_dim[i]))

        self.layers.append(np.zeros(self.output_dim))
        print(self.layers)

        self.weights = list()
        self.init_weight(layers_dim)

    def init_weight(self,layers_dim):
        n = 0 # index to get layers dim

        for layer in self.layers[:-1]:
            self.weights.append( np.random.rand( layers_dim[n],len(layer)) )
        
            n = n + 1
        
        # DEBUG
        print("Initial weight matrices are: {}".format(self.weights))

    def g(self,h):
        return np.tanh(h)

    def g_deriv(self,h):
        return (1-np.power(self.g(h),2))

    def train(self):
        # For each layer get the output
        # This output has to be fed into next layer
        
        # Getting the output is getting h for each node
        # Get random pattern
        for p in np.random.permutation(self.num_patterns):
            x_input = self.x[:,p]

            # ########## Forward
            i=0
            h_list = []
            x_inputs = []
            
            for weight_matrix in self.weights:

                h = np.dot(weight_matrix,x_input)
                x_input = self.g(h)

                i = i+1
                h_list.append(h)
                x_inputs.append(x_input)

            h_array = np.array(h_list)
            x_inputs = np.array(x_inputs)
            
            # ######### Backward
            it_list = [1]
            m=1
            deltas_m = self.g_deriv(h_array[m])*(self.y[p]-x_input)
            dW = self.etha * x_inputs[m-1] * deltas_m
            self.weights[m] += dW

            current_weight = self.weights[m]
            temp = current_weight*deltas_m
            deltas_m = np.multiply(np.vectorize(self.g_deriv)(h_array[m-1]),temp)
            dW = self.etha * np.array([np.multiply(a,b) for a,b in zip(x_input,deltas_m)])
            self.weights[m-1] += dW
            break
        

if __name__ == '__main__':
    XOR_input = np.array([[-1,1,-1,1],[-1,-1,1,1],[1,1,1,1]])
    XOR_output = np.array([-1,1,1,-1])
    layers_dim = np.array([2,1]) # includes output dim
    perceptron = MultiLayerPerceptron(XOR_input,XOR_output,2,layers_dim)  
    perceptron.train()  



    

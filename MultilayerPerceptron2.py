import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

MAX_ITER = 10
# Only for two layers
class MultiLayerPerceptron(): 
    def __init__(self,input_neurons,hidden_neurons,output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.layers = 2

    def setPatterns(self,input_patterns,desired_output):
        rows,cols = input_patterns.shape
        if self.input_neurons != rows:
            raise Exception("Patterns input dimension does not match Perceptron input dimensions {}.".format(rows,self.input_neurons))
        # TODO: check output neurons
        self.patterns_inputs = input_patterns
        self.patterns_outputs = desired_output   
        self.weights = np.array(2)
        self.initWeights()
        self.num_patterns = cols

    def initWeights(self):
        a = np.random.rand(self.hidden_neurons,self.input_neurons)
        b = np.random.rand(self.output_neurons,self.hidden_neurons)
        
        c = []
        c.append(a)
        c.append(b)

        self.weights = np.array(c)

    def g(self,h):
        return np.tanh(h)

    def g_deriv(self,h):
        return (1-np.power(self.g(h),2))

    def forwardPropagation(self,V_input):
        V_list = []
        h_list = []
        V_list.append(V_input)

        for m in range(self.layers):
            weighted_sum = np.dot(self.weights[m],V_input)
            V_new = np.vectorize(self.g)(weighted_sum)
            
            V_list.append(V_new)
            h_list.append(weighted_sum)

            V_input = V_new

        return V_list,h_list
    
    def getDeltaWeight(self, delta, V):
        dW = np.outer(delta,V)
        return dW

    def getDeltas(self,desired_y,V_list,h_list):
        delta_M = np.vectorize(self.g_deriv)(h_list[-1])*(desired_y-V_list[-1])

        temp = self.weights[1]*delta_M
        delta_new = np.vectorize(self.g_deriv)(h_list[0])
        
        deltas = []
        deltas.append(delta_new)
        deltas.append(delta_M)

        return deltas

    def backPropagation(self,V_list,h_list,desired_y):
        deltas = self.getDeltas(desired_y, V_list, h_list)

        # Uptdate weights
        dW = self.getDeltaWeight(deltas[0],V_list[0])
        self.weights[0] += dW
        dW = self.getDeltaWeight(deltas[1],V_list[1])
        self.weights[1] += dW

    def evaluateInputPatterns(self):
        output = np.zeros(self.num_patterns)
        for i in range(self.num_patterns):
            V_list,_ = self.forwardPropagation(self.patterns_inputs[:,i])
            output[i] = V_list[-1]
        return output

    def trainingIteration(self):
        for p in np.random.permutation(self.num_patterns):
            V_input = self.patterns_inputs[:,p]
            desired_y = self.patterns_inputs[0][p] # TODO: corregir por si la salida es de mas dimensiones
        
            V_list, h_list = self.forwardPropagation(V_input)
            self.backPropagation(V_list,h_list,desired_y)

    def train(self):    
        it = 0
        while True:
            it += 1
            self.trainingIteration()

            if ( np.array_equal(self.evaluateInputPatterns(), self.patterns_outputs)):
                log.info("TRAIN SUCCESS")
                break
            if (it > MAX_ITER):
                log.error("TRAINING FAILED")
                break

            print(self.weights)


        

if __name__ == '__main__':
    XOR_input = np.array([[-1,1,-1,1],[-1,-1,1,1],[1,1,1,1]])
    XOR_output = np.array([-1,1,1,-1])
    perceptron = MultiLayerPerceptron(3,2,1)
    perceptron.setPatterns(XOR_input,XOR_output)
    perceptron.train()
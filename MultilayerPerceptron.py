import numpy as np
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

MAX_ITER = 2000
TOL = 1e-1
# Actually, only two layers :-)
class MultiLayerPerceptron(): 
    def __init__(self,input_neurons,hidden_neurons,output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.layers = 2
        self.g = np.tanh

    def setPatterns(self,input_patterns,desired_output):
        rows,cols = input_patterns.shape
        if self.input_neurons != rows:
            raise Exception("Patterns input dimension {} does not match Perceptron input dimensions {}.".format(rows,self.input_neurons))
        # TODO: check output neurons
        self.patterns_inputs = input_patterns
        self.patterns_outputs = desired_output   
        self.weights = np.array(2)
        self.initWeightsRandomly()
        self.num_patterns = cols

    def initWeightsRandomly(self):
        
        a = np.random.rand(self.hidden_neurons,self.input_neurons)
        b = np.random.rand(self.output_neurons,self.hidden_neurons)
        c = [a,b]
        self.weights = np.array(c)

    def getInputOutputOfPattern(self,p):
        return (self.patterns_inputs[:,p], self.patterns_outputs[p])

    def g_deriv(self,h):
        return (1-np.power(self.g(h),2))

    def forwardPropagation(self,V_input):
        V_list = []
        h_list = []
        V_list.append(V_input)

        Vm_prev = V_input
        for m in range(self.layers):
            weighted_sum = np.dot(self.weights[m],Vm_prev)
            V_m = np.vectorize(self.g)(weighted_sum)
            
            V_list.append(V_m)
            h_list.append(weighted_sum)

            Vm_prev = V_m

        return V_list,h_list
    
    def getDeltaWeight(self, delta, V):
        dW = 0.1*np.outer(delta,V)
        return dW

    def computeLastDelta(self,last_h,desired_output,actual_output):
        delta_M = (self.g_deriv)(last_h)*(desired_output-actual_output) #TODO: vectorize
        return delta_M

    def backPropagation(self,h_list, delta_M):
        deltas = []
        deltas.append(delta_M)
        # TODO: make loop
        delta_m = delta_M
        delta_m_prev = np.vectorize(self.g_deriv)(h_list[0]) * (self.weights[1]*delta_m)
        deltas.append(delta_m_prev)
        deltas = list(reversed(deltas))
        return deltas
        

    def evaluateInputPatterns(self):
        output = np.zeros(self.num_patterns)
        for i in range(self.num_patterns):
            V_list,_ = self.forwardPropagation(self.patterns_inputs[:,i])
            output[i] = V_list[-1]
        return output

    def updateWeights(self, deltas, V_list):
        for m in range(self.layers):
            dW = self.getDeltaWeight(deltas[m],V_list[m])
            self.weights[m] += dW

    def trainingIteration(self):
        for p in np.random.permutation(self.num_patterns):
            V_input,desired_output = self.getInputOutputOfPattern(p)    
            
            V_list, h_list = self.forwardPropagation(V_input)
            delta_M = self.computeLastDelta(h_list[-1],desired_output,V_list[-1])    
            delta_list = self.backPropagation(h_list,delta_M)
            
            self.updateWeights(delta_list,V_list)
           
    def train(self):    
        it = 0
        errors = []
        real_outputs = []
        while True:
            it += 1
            self.trainingIteration()
            
            expected = self.patterns_outputs
            real = self.evaluateInputPatterns()
            
            errors.append(((expected - real)**2).sum())
            real_outputs.append(real[2])
            if ( np.allclose(expected, real, atol=TOL)):
                log.info("TRAIN SUCCESS")
                break
            if (it > MAX_ITER):
                log.error("TRAINING FAILED")
                break
        
        # DEBUG
        #plt.plot(range(it),np.array(errors),'xb')
        #plt.xlabel("Iteraciones")
        #plt.ylabel("Error=sum((expected-real)**2)")
        #plt.show()

        #plt.plot(range(it),np.array(real_outputs),'xr')
        #plt.xlabel("Iteraciones")
        #plt.show()

if __name__ == '__main__':
    XOR_input = np.array([[-1,1,-1,1],[-1,-1,1,1],[1,1,1,1]])
    XOR_output = np.array([-1,1,1,-1])

    perceptron = MultiLayerPerceptron(3,2,1)
    perceptron.setPatterns(XOR_input,XOR_output)
    perceptron.train()
    print("Actual output is:", perceptron.evaluateInputPatterns())
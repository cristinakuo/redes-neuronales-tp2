# Energia, es el error, lo quiere minimizar
import numpy as np
import matplotlib.pyplot as plt

STD_DEV = 4
ITERATIONS_PER_TEMPERATURE = 4
TEMPERATURE_DECREASING_FACTOR = 0.99
class SimulatedAnnealing():
    def __init__(self,input_neurons,hidden_neurons,output_neurons, init_temp):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.layers = 2
        self.g = np.tanh
        self.temperature = init_temp
        self.iterations_per_temp_count = 0
        self.currentEnergy = 0
        self.num_patterns = 0

    def setPatterns(self,input_patterns,desired_output):
        rows,cols = input_patterns.shape
        if self.input_neurons != rows:
            raise Exception("Patterns input dimension {} does not match Perceptron input dimensions {}.".format(rows,self.input_neurons))
        # TODO: check output neurons dimensions
        self.patterns_inputs = input_patterns
        self.patterns_outputs = desired_output
        self.num_patterns = cols   
        self.initWeightsRandomly()
        self.initEnergy()
        

    def initWeightsRandomly(self):   
        a = np.random.rand(self.hidden_neurons,self.input_neurons)
        b = np.random.rand(self.output_neurons,self.hidden_neurons)
        c = [a,b]
        self.weights = np.array(c)

    def initEnergy(self):
        # Calculate initial energy
        actual_output = np.zeros(self.num_patterns)
        for p in np.random.permutation(self.num_patterns):
            V_input,desired_output = self.getInputOutputOfPattern(p)    
            actual_output[p] = self.forwardPropagation(V_input, self.weights)
        return (self.getErrorEnergy(desired_output, actual_output))

    def getInputOutputOfPattern(self,p):
        return (self.patterns_inputs[:,p], self.patterns_outputs[p])

    def forwardPropagation(self,V_input, weights):
        Vm_prev = V_input
        for m in range(self.layers):
            weighted_sum = np.dot(weights[m],Vm_prev)
            V_m = np.vectorize(self.g)(weighted_sum)

            Vm_prev = V_m
        return V_m

    def getErrorEnergy(self, desired_output, actual_output):
        return (0.5*np.mean(np.square(desired_output-actual_output)))
    
    def getWeightsWithNormal(self):
        delta_W = np.random.normal(0,STD_DEV, size=(self.hidden_neurons,self.input_neurons))
        new_weight_1 = self.weights[0] + delta_W
        delta_W = np.random.normal(0,STD_DEV, size=(self.output_neurons,self.hidden_neurons))
        new_weight_2 = self.weights[1] + delta_W        
        return np.array([new_weight_1, new_weight_2])

    def acceptWithProb(self, dE):
        if (np.random.rand() <= np.exp(-dE/self.temperature) ): # random() returns value between 0 and 1
            return True
        else:
            return False

    def updateWeights(self, delta_energy, new_weights):
        if delta_energy < 0:
            self.weights = new_weights
            self.currentEnergy += delta_energy    
        elif self.acceptWithProb(delta_energy): 
            self.weights = new_weights
            self.currentEnergy += delta_energy

    def updateTemperature(self):
        if self.iterations_per_temp_count == ITERATIONS_PER_TEMPERATURE:
            self.temperature *= TEMPERATURE_DECREASING_FACTOR
            self.iterations_per_temp_count = 1
        else:
            self.iterations_per_temp_count += 1

    def trainingIteration(self):
        actual_output = np.zeros(self.num_patterns)
        
        # Forward propagation of input patterns with normally generated weights
        new_weights = self.getWeightsWithNormal()
        for p in np.random.permutation(self.num_patterns):
            V_input = self.patterns_inputs[:,p]   
            actual_output[p] = self.forwardPropagation(V_input, new_weights)
            
        new_energy = self.getErrorEnergy(self.patterns_outputs, actual_output)
        dE =  new_energy - self.currentEnergy
        self.updateWeights(dE, new_weights)
        self.updateTemperature() 

    def train(self):
        plt.ion()
        it = 0
        _,axs=plt.subplots(nrows=2,ncols=1)
        
        while self.temperature > 0.1:
            self.trainingIteration()
            it += 1
            axs[0].plot(self.temperature,self.currentEnergy,"*r") # TODO: error vs temp
        input("Press to continue...")

if __name__=='__main__':
    XOR_input = np.array([[-1,1,-1,1],[-1,-1,1,1],[1,1,1,1]])
    XOR_output = np.array([-1,1,1,-1])
    perceptron = SimulatedAnnealing(3,2,1,init_temp=5)
    perceptron.setPatterns(XOR_input,XOR_output)
    perceptron.train()
    print(perceptron.forwardPropagation(XOR_input,perceptron.weights))

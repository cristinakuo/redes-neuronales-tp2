from MultilayerPerceptron import MultiLayerPerceptron
import numpy as np

XOR_input = np.array([[-1,1,-1,1],[-1,-1,1,1],[1,1,1,1]])
XOR_output = np.array([-1,1,1,-1])
perceptron = MultiLayerPerceptron(3,2,1)
perceptron.setPatterns(XOR_input,XOR_output)
perceptron.train()
print("Actual output is:", perceptron.evaluateInputPatterns())
from MultilayerPerceptron import MultiLayerPerceptron
import numpy as np
import matplotlib.pyplot as plt

POINTS = 100
x = np.linspace(0, 2*np.pi, POINTS)
y = np.linspace(0, 2*np.pi, POINTS)
z = np.linspace(-1, 1, POINTS)
bias = np.ones(POINTS)

function_input = np.array([x,y,z,bias])
function_output = np.sin(x)+ np.cos(y) + z

perceptron = MultiLayerPerceptron(4,4,1)
perceptron.setPatterns(function_input,function_output)
perceptron.train()
print("Calculated output is: ", z)
print("Perceptron output is:", perceptron.evaluateInputPatterns())
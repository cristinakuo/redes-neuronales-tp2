from perceptron import SimplePerceptron
import matplotlib.pyplot as plt
import numpy as np

def main():
    # AND
    x = np.array([[-1,-1, 1, 1],[-1,-1,-1,1],[1,1,1,1]])
    y = np.array([-1,-1,-1,1])

    percep = SimplePerceptron(x,y)
    train_success = percep.train()
    print(percep.W)
    if train_success:
        print("TRAIN SUCCESS")
    else:
        print("TRAIN FAILURE")
    print("trained y is: {}".format(percep.y_trained))
    plt.plot()

if __name__ == '__main__':
    main()

from SimplePerceptron import SimplePerceptron
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

def or_2dim():
    # AND
    x = np.array([[-1, 1,-1, 1],
                  [-1,-1, 1, 1],
                  [ 1, 1, 1, 1]])
    y = np.array([-1, 1, 1,1])

    percep = SimplePerceptron(3,1)
    percep.setPatterns(x,y)
    train_success = percep.train()

    if train_success:
        log.info("TRAIN SUCCESS")
    else:
        log.error("TRAIN FAILURE")
        exit()

    log.info("Output obteined from trained perceptron: {}".format(percep.evaluate(x)))
    plt.plot(x[0,:],x[1,:],'or')
    x_line = np.linspace(-5,5,10)
    recta_1 = -percep.weight_matrix[0]*x_line-percep.weight_matrix[2]
    recta_1 = recta_1/percep.weight_matrix[1]
    plt.plot(x_line,recta_1)
    plt.grid()
    plt.xlabel("E1")
    plt.ylabel("E2")

    plt.show()

def and_2dim():
    # AND
    x = np.array([[-1, 1,-1, 1],
                  [-1,-1, 1, 1],
                  [ 1, 1, 1, 1]])
    y = np.array([-1,-1,-1,1])

    percep = SimplePerceptron(3,1)
    percep.setPatterns(x,y)
    train_success = percep.train()

    if train_success:
        log.info("TRAIN SUCCESS")
    else:
        log.error("TRAIN FAILURE")
        exit()

    log.info("Output obteined from trained perceptron: {}".format(percep.evaluate(x)))
    plt.plot(x[0,:],x[1,:],'or')
    x_line = np.linspace(-5,5,10)
    recta_1 = -percep.weight_matrix[0]*x_line-percep.weight_matrix[2]
    recta_1 = recta_1/percep.weight_matrix[1]
    plt.plot(x_line,recta_1)
    plt.grid()
    plt.xlabel("E1")
    plt.ylabel("E2")

    plt.show()


def and_4dim():
    x = np.array([[-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1],
                  [-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1],
                  [-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1],
                  [-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    y = np.array( [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1])

    percep = SimplePerceptron(5,1)
    percep.setPatterns(x,y)
    train_success = percep.train()

    if train_success:
        log.info("TRAIN SUCCESS")
    else:
        log.error("TRAIN FAILURE")
        exit()

    log.info("Output obteined from trained perceptron: {}".format(percep.evaluate(x)))


def main():
    or_2dim()

if __name__ == '__main__':
    main()

from SimplePerceptron import SimplePerceptron
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 


def and_2dim():
    # AND
    x = np.array([[-1,-1, 1, 1],[-1,-1,-1,1],[1,1,1,1]])
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
    plt.plot()

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
    plt.plot()


def main():
    and_4dim()

if __name__ == '__main__':
    main()

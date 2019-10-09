from perceptron import SimplePerceptron
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 


def main():
    # AND
    x = np.array([[-1,-1, 1, 1],[-1,-1,-1,1],[1,1,1,1]])
    y = np.array([-1,-1,-1,1])

    percep = SimplePerceptron(x,y)
    train_success = percep.train()
    print(percep.W)

    if train_success:
        log.info("TRAIN SUCCESS")
    else:
        log.error("TRAIN FAILURE")
        exit()

    log.info("Output obteined from trained perceptron: {}".format(percep.get_output(x)))
    plt.plot()

if __name__ == '__main__':
    main()

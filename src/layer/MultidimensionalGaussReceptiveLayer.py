import numpy as np

class Multidimensional_Gauss_Receptive_Layer:
    def __init__(self, numNeurons, numDimensions, VThreshold=1, VRest=0):
        """
        Integrate and Fire SNN layer with gaussian receptive fields for multidimensional input.

        :param numNeurons: Number of neurons (int) >=2
        :param numDimensions: Number of input dimensions (int)
        :param VThreshold: Threshold potential
        :param VRest: Resting potential
        """
        self.numNeurons = numNeurons
        self.numDimensions = numDimensions
        self.VT = VThreshold
        self.VR = VRest
        self.V = np.ones((numNeurons * numDimensions, 1)) * self.VR
        self.S = np.ones((numNeurons * numDimensions, 1)) * -1

    def defineGausses(self, valMin, valMax, amplMax):
        """
        Defining Gaussian receptive fields for each neuron of each dimension.

        :param valMin: Array of minimal values of each dimension.
        :param valMax: Array of maximal values of each dimension.
        :param amplMax: Scaling value of all the gaussian receptive functions.
        :return:
        """
        self.amplMax = amplMax
        self.M = []
        self.sigma = []
        for i in range(self.numDimensions):
            M, sigma = np.linspace(valMin[i], valMax[i], self.numNeurons, retstep=True)
            self.M.append(M)
            self.sigma += [sigma, ] * self.numNeurons
        self.M = np.array(self.M).reshape(-1, 1)
        self.sigma = np.array(self.sigma).reshape(-1, 1)

    def gauss(self, x):
        """
        Calculate values of the receptive functions for every neuron of every dimension.

        :param x: Input vector
        :return:
        """
        x = np.repeat(x, self.numNeurons).reshape(-1, 1)
        return np.exp(-1 / 2 * ((x - self.M) / self.sigma) ** 2) * self.amplMax

    def reset(self):
        """
        Reset all potentials and spiketimes related to the layer.

        :return:
        """
        self.V = np.ones((self.numNeurons * self.numDimensions, 1)) * self.VR
        self.S = np.ones((self.numNeurons * self.numDimensions, 1)) * -1

    def forward(self, t, x):
        """
        Calculate a forward step of the layer with the given inputs.

        :param t: Simulation time
        :param x: Input vector
        :return:
        """

        g = self.gauss(x)
        self.V += g
        self.S[np.bitwise_and(self.S == -1, self.V >= self.VT)] = t

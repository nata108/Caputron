import numpy as np

class IF_Layer:
    def __init__(self, numNeurons, VThreshold=1, VRest=0):
        """
        Basic Integrate and Fire SNN layer.

        :param numNeurons: Number of neurons in the layer (int)
        :param VThreshold: Threshold potential
        :param VRest: Resting potential
        """
        self.numNeurons = numNeurons
        self.VT = VThreshold
        self.VR = VRest
        self.V = np.ones((numNeurons, 1)) * self.VR
        self.S = np.ones((numNeurons, 1)) * -1

    def forward(self, t, invec):
        """
        Calculates a forward step of the layer based on the given inputs.

        :param t: Simulation time
        :param invec: Current input vector
        :return:
        """
        self.V += invec.reshape(-1, 1)
        self.S[np.bitwise_and(self.S == -1, self.V >= self.VT)] = t

    def reset(self):
        """
        Reset potentials and spiketimes related to this layer.

        :return:
        """
        self.V = np.ones((self.numNeurons, 1)) * self.VR
        self.S = np.ones((self.numNeurons, 1)) * -1

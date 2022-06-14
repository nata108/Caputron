import numpy as np


class LIF_K_Layer:
    def __init__(self, numNeurons, tM=15, tS=5, VThreshold=1, VRest=0, V0=1, preNeurons=None):
        """
        Initalizer of a Leaky Integrate and Fire SNN Layer with a Tempotron-like K kernel function which is a
        solution of the differential equation derived from neural dynamics.

        :param numNeurons: Number of neurons in the layer (int)
        :param tM: Membrane time constant of Tempotron-like K kernel
        :param tS: Synapse time constant of Tempotron-like K kernel
        :param VThreshold: Threshold potential
        :param VRest: Resting potential
        :param V0: Potential multiplier of Tempotron-like K kernel
        """
        self.numNeurons = numNeurons
        self.Kshape = numNeurons
        if preNeurons is not None:
            self.Kshape = preNeurons
        self.tM = tM
        self.tS = tS
        self.VT = VThreshold
        self.VR = VRest
        self.V0 = V0
        self.V = np.ones((numNeurons, 1)) * self.VR
        self.S = np.ones((numNeurons, 1)) * -1
        self.K = np.zeros((self.Kshape, 1))
        self.maxK = np.zeros((self.Kshape, numNeurons))
        self.maxt = np.ones((numNeurons, 1)) * -1
        self.maxV = np.ones((numNeurons, 1)) * self.VR

    def logMaxValues(self, t):
        """
        Updates saved values wich refer to the step with maximal membrane potential.

        :param t: Simulation time
        :return:
        """
        affectedNeurons = (self.V > self.maxV).flatten()
        self.maxK[:, affectedNeurons] = self.K.copy()
        self.maxt[affectedNeurons] = t
        self.maxV[affectedNeurons] = self.V[affectedNeurons].copy()

    def forward(self, t, spikes, w):
        """
        Calculates a forward step of the layer with the given inputs.

        :param t: Simulation time
        :param spikes: Input spikes
        :param w: The corresponding weight matrix before the layer
        :return:
        """
        self.calcK(t, spikes)
        self.V = w @ self.K
        self.S[np.bitwise_and(self.S == -1, self.V >= self.VT)] = t
        self.logMaxValues(t)

    def calcK(self, t, spikes):
        """
        This function calculates the value of the Tempotron-like kernel.

        :param t: Simulation time
        :param spikes: Input spikes
        :return:
        """
        self.K = np.zeros((self.Kshape, 1))
        self.K[spikes > -1] = self.V0 * (
                np.exp(-(t - spikes[spikes > -1]) / self.tM) - np.exp(-(t - spikes[spikes > -1]) / self.tS))

    def reset(self):
        """
        Reset all layer related potential, spike and kernel values.

        :return:
        """
        self.V = np.ones((self.numNeurons, 1)) * self.VR
        self.S = np.ones((self.numNeurons, 1)) * -1
        self.K = np.zeros((self.Kshape, 1))
        self.maxK = np.zeros((self.Kshape, self.numNeurons))
        self.maxt = np.ones((self.numNeurons, 1)) * -1
        self.maxV = np.ones((self.numNeurons, 1)) * self.VR

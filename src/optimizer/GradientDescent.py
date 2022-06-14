import numpy as np

class GradientDescent:
    def __init__(self, lr):
        """
        Classic gradient descent based optimizer with first order derivative, and Tempotron-like loss.

        :param lr:
        """
        self.lr = lr

    def calcDW(self, w, K, labels, spikes):
        """
        Calculate weight changes.

        :param w: Original weight matrix
        :param K: Kernel function value vector of the postsynaptic layer
        :param labels: Desired output in the form of required or undesired spike activity (bool vector)
        :param spikes: Spikes of the postsynaptic layer
        :return:
        """

        sign = np.zeros_like(spikes)
        sign += -1 * labels * (spikes < 0)
        sign += -1 * (labels - 1) * (spikes > -1)

        return K.T * self.lr * -1 * sign

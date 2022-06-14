import numpy as np
from scipy.special import gamma

class Caputron:
    def __init__(self, alpha, lr):
        """
        Caputo optimizer with Tempotron-like loss function.

        :param alpha: Order of the Caputo derivative
        :param lr: Learning rate
        """
        self.alpha = alpha
        self.lr = lr
        self.gamma = gamma(alpha)

    def integrativePart(self, c, w):
        """
        Calculate the coefficient of the Caputron loss derived from the solution of the integral.

        :param c: Vector of minimas of the weight values for each neuron
        :param w: Weight matrix
        :return:
        """
        return (w - c) ** (1 - self.alpha) / (1 - self.alpha)

    def calcDW(self, w, K, labels, spikes):
        """
        Calculating weight changes.

        :param w: Original weight matrix
        :param K: Kernel function value vector of the postsynaptic layer
        :param labels: Desired output in the form of required or undesired spike activity (bool vector)
        :param spikes: Spikes of the postsynaptic layer
        :return:
        """
        # Tempotron modell alapjan elojel meghatarozasa
        # W matrix (post,pre) alaku
        sign = np.zeros_like(spikes)  # nulla alap esetben
        sign += -1 * labels * (spikes < 0)  # -1 ha a label 1, de nincs spike
        sign += -1 * (labels - 1) * (spikes > -1)  # 1 ha a label 0, de van spike (labels-1 negativ lesz)

        c = np.min(w, axis=1).reshape(-1, 1)  # soronként
        return K.T / self.gamma * self.integrativePart(c, w) * self.lr * -1 * sign

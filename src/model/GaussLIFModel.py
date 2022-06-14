import numpy as np
import time

from layer.MultidimensionalGaussReceptiveLayer import Multidimensional_Gauss_Receptive_Layer
from layer.LIFKLayer import LIF_K_Layer
from optimizer.GradientDescent import GradientDescent
from optimizer.Caputron import Caputron


class GaussLIFModel:
    def __init__(self, preDim, preSize, preMin, preMax, postSize, logger=None):
        """
        Two-layer SNN model with a Gaussian receptive field IF layer on the presynaptic side, and a LIF layer on the
        postsynaptic side.

        :param preDim: Dimension of the presynaptic layer
        :param preSize: Neuron number of the presynaptic layer
        :param preMin: Vector of minimal values of the presynaptic layer
        :param preMax: Vector of maximal values of the presynaptic layer
        :param postSize: Neuron number of the postsynaptic layer
        :param logger: (Optional) Logger object to capture training data
        """
        self.preDim = preDim
        self.preSize = preSize
        self.preMin = preMin
        self.preMax = preMax
        self.postSize = postSize
        self.optimizer = None
        self.logger = logger
        self.W = np.random.rand(self.postSize, self.preSize * self.preDim)

        # Gauss reteg def tobb dimenziora
        self.pre = Multidimensional_Gauss_Receptive_Layer(preSize, preDim)
        self.pre.defineGausses(self.preMin, self.preMax, 0.3)

        # LIF_K reteg def
        print(preSize * preDim)
        self.post = LIF_K_Layer(self.postSize, preNeurons=preSize * preDim)

    def setLogger(self, logger, alphaidx, lridx, runidx):
        """
        Set logger with the given alpha, lr, and run number values acquired from outside of the model.

        :param logger: Logger object
        :param alphaidx: Index of alpha value
        :param lridx: Index of learning rate value
        :param runidx: Index of run number value
        :return:
        """
        self.logger = logger
        self.alphaidx = alphaidx
        self.lridx = lridx
        self.runidx = runidx

    def createOptimizer(self, alpha, lr, tarray):
        """
        Create optimizer with the given values. Creates Caputron optimizer for fractional order derivatives, and
        classic GradientDescent optimizer if alpha is exactly 1.

        :param alpha: Order of derivative to use
        :param lr: Learning rate
        :param tarray: Array of timesteps used for simulation
        :return:
        """
        self.alpha = alpha
        self.lr = lr
        self.tarray = tarray
        if alpha == 1:
            self.optimizer = GradientDescent(self.lr)
        else:
            self.optimizer = Caputron(self.alpha, self.lr)

    def resetWeights(self):
        """
        Randomize weight values from 0 to 1 uniformly.

        :return:
        """
        self.W = np.random.rand(self.postSize, self.preSize * self.preDim)

    def resetWeightsGauss(self):
        """
        Randomize weight values drawing from a normal distribution with std 0.05 and mean 0.5.

        :return:
        """
        self.W = np.random.normal(0.5, 0.05, (self.postSize, self.preSize * self.preDim))

    def step(self, inData, outData=None, training=False):
        """
        Perform a training step.

        :param inData: Input vector of data
        :param outData: (Optional) Desired output
        :param training: Training indicator, if set weights are modified according to the optimizer (bool)
        :return:
        """
        if self.optimizer is None:
            raise Exception("No optimizer defined")
        if training and outData is None:
            raise Exception("No labels are defined for a supervised step.")

        for t in self.tarray:  # Minden timestepre
            # Lefuttatni a lepeseket
            self.pre.forward(t, inData)
            # pre vektorok egyesitese
            self.post.forward(t, self.pre.S, self.W)
            if np.sum(self.post.S > 0) > 0:
                break

        if training:
            # Utolso lepes adataival (tmax) suly update
            DW = self.optimizer.calcDW(self.W, self.post.maxK, outData.reshape(-1, 1), self.post.S)
            self.W = self.W + DW

    def reset(self):
        """
        Reset the whole model to its initial state.

        :return:
        """
        # Reteg potencialok es spikeok resetje
        self.pre.reset()
        self.post.reset()

    def train(self, epochnum, trainData, trainLabels, valData, valLabels, retBestEpochAcc=False, verbose=True):
        """
        Perform a whole training process.

        :param epochnum: Number of epochs
        :param trainData: Training data
        :param trainLabels: Training labels
        :param valData: Validation data
        :param valLabels: Validation labels
        :param retBestEpochAcc: Best epoch validation accuracy is returned when set, None is returned otherwise (bool)
        :param verbose: Displays training status messages if set (bool)

        :return: Validation accuracy of the best epoch during training, or None
        """
        if verbose:
            print("\n\n***Training...***")
        traindatanum = trainData.shape[0]
        valdatanum = valData.shape[0]
        epochAccs = []
        for i in range(epochnum):  # Minden epochra
            stime = time.time()
            # Az adatok beadasanak uj random sorrendet valasztunk
            r = np.random.permutation(traindatanum)  # np.arange(traindatanum)#np.random.permutation(traindatanum)
            trainAcc = 0
            trainLoss = 0
            for j in range(traindatanum):  # Minden adatpontra
                self.reset()

                # Adatok kivalasztasa
                inData = trainData[r[j]].copy()
                outData = trainLabels[r[j]].copy()

                self.step(inData, outData, training=True)
                trainAcc += int(self.checkIfCorrect(outData.reshape(-1, 1)))
                trainLoss += self.loss(outData.reshape(-1, 1))

            trainAcc = trainAcc / traindatanum
            trainLoss = trainLoss / traindatanum

            valAcc = 0
            valLoss = 0
            r = np.random.permutation(valdatanum)
            for j in range(valdatanum):  # Minden adatpontra
                self.reset()

                # Adatok kivalasztasa
                inData = valData[r[j]].copy()
                outData = valLabels[r[j]].copy()

                self.step(inData, outData, training=False)
                valAcc += int(self.checkIfCorrect(outData.reshape(-1, 1)))
                valLoss += self.loss(outData.reshape(-1, 1))

            valAcc = valAcc / valdatanum
            valLoss = valLoss / valdatanum

            epochAccs.append(valAcc)

            if self.logger is not None:
                self.logger.log(self.alphaidx, self.lridx, i, self.runidx, trainLoss, trainAcc, valLoss, valAcc)

            if verbose:
                print(
                    "Time taken(s): {0:1.4f} TrainLoss: {1:1.4f} TrainAcc: {2:1.4f} ValLoss: {3:1.4f} ValAcc: {4:1.4f}" \
                        .format(time.time() - stime, trainLoss, trainAcc, valLoss, valAcc))
        if retBestEpochAcc:
            return max(epochAccs)
        else:
            return None

    def predict(self, data):
        """
        Makes a prediction of the outputs based on a single input data

        :param data: Single input
        :return: Spiketimes of the output layer
        """
        self.reset()

        # Adatok kivalasztasa
        inData = data.copy()

        self.step(inData, outData=None, training=False)
        return self.post.S.copy()

    def checkIfCorrect(self, outData):
        """
        Checks if the current state of the network is correct according to the output label.

        :param outData: Output label
        :return: Boolean value of the prediction result.
        """
        postActivity = np.where(self.post.S < -0.5, self.tarray[-1] + 1, self.post.S)
        return np.argmin(postActivity) == np.argmax(outData)

    def loss(self, outData):
        """
        Calculates the loss based on the current state of the network and the output label.

        :param outData: Output label
        :return: Tempotron-like loss value
        """
        sign = np.zeros_like(self.post.S)  # nulla alap esetben
        sign += -1 * outData * (self.post.S < 0)  # -1 ha a label 1, de nincs spike
        sign += -1 * (outData - 1) * (self.post.S > -1)  # 1 ha a label 0, de van spike (labels-1 negativ lesz)

        return np.average(np.abs((self.post.maxV - self.post.VT) * sign))
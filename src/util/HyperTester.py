import numpy as np
from util.Logger import Logger

class HyperTester:
    def __init__(self, alphas, lrs, model, logpath):
        """
        Object for testing hyperparameter settings with multiple possible runs.

        :param alphas: Array of derivative orders to test [0,1]
        :param lrs: Array of learning rates
        :param model: Model architecture to test with
        :param logpath: Path to generate log files
        """
        self.alphas = alphas
        self.lrs = lrs
        self.model = model
        self.logpath = logpath
        self.genSchedule()

    def genSchedule(self):
        """
        Generating meshgrid of the alpha and lr vectors for testing all possible pairs of values.

        :return:
        """
        self.paramSchedule = np.array(np.meshgrid(self.alphas, self.lrs)).reshape(2, -1).T
        self.indexSchedule = np.array(np.meshgrid(np.arange(len(self.alphas)), np.arange(len(self.lrs)))).reshape(2,
                                                                                                                  -1).T

    def run(self, runsPerPreset, epochnum, dataManager, tarray, verbose=True):
        """
        Runs a whole hyperparameter optimization process according to the previously generated schedule.

        :param runsPerPreset: Number of times to run a full training with a randomly initialized network for each parameter preset
        :param epochnum: Epochs to run each trial for
        :param dataManager: DataManager object which handles the data for training and validation
        :param tarray: Array of simulation timesteps
        :param verbose: Displays training status informations if set (bool)
        :return:
        """
        self.logger = Logger(self.logpath, len(self.alphas), len(self.lrs), epochnum, runsPerPreset)
        self.logger.logParameters(self.alphas, self.lrs)
        for i in range(self.paramSchedule.shape[0]):
            self.model.createOptimizer(self.paramSchedule[i, 0], self.paramSchedule[i, 1], tarray)
            trainData, trainLabels = dataManager.getTrainData()
            valData, valLabels = dataManager.getValData()

            for j in range(runsPerPreset):
                self.model.setLogger(self.logger, self.indexSchedule[i, 0], self.indexSchedule[i, 1], j)
                self.model.resetWeightsGauss()
                if verbose:
                    print("\nProgress: ",
                          int(100 * (i * runsPerPreset + j) / (self.paramSchedule.shape[0] * runsPerPreset)), "%")
                self.model.train(epochnum, trainData, trainLabels, valData, valLabels, verbose=verbose)
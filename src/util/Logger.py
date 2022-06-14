import numpy as np
import shutil
import time
import os
from datetime import datetime

class Logger:
    def __init__(self, path, dimAlpha, dimLr, dimEpoch, dimRun):
        """
        Object for handling the logging of optimization runs.

        :param path: Path to store logfiles at
        :param dimAlpha: Number of alpha values to test
        :param dimLr: Number of learning rate values to test
        :param dimEpoch: Number of epochs to run each optimization for
        :param dimRun: Number of runs for each alpha-learning rate pair
        """
        self.prepareLogPath(path)

        self.TLFile = self.logpath + "trainLoss.npy"
        self.TAFile = self.logpath + "trainAcc.npy"
        self.VLFile = self.logpath + "valLoss.npy"
        self.VAFile = self.logpath + "valAcc.npy"
        self.alphaFile = self.logpath + "alpha.npy"
        self.lrFile = self.logpath + "lr.npy"

        self.TLmat = -1 * np.ones((dimAlpha, dimLr, dimEpoch, dimRun))
        self.VLmat = -1 * np.ones((dimAlpha, dimLr, dimEpoch, dimRun))
        self.TAmat = np.zeros((dimAlpha, dimLr, dimEpoch, dimRun))
        self.VAmat = np.zeros((dimAlpha, dimLr, dimEpoch, dimRun))

    def logParameters(self, alphas, lrs):
        """
        Save alpha and learning rate parameters in the order of execution to a separate file.

        :param alphas: Vector of alpha values
        :param lrs: Vector of learning rate values
        :return:
        """
        np.save(self.alphaFile, np.array(alphas))
        np.save(self.lrFile, np.array(lrs))

    def prepareLogPath(self, path):
        """
        Open a new folder in the specified folder. The new folder for logging is based on a second resolution timestamp
        and a random id.

        :param path: Path of the root folder
        :return:
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S") + "_" + str(
            int(time.time_ns() / 100) % 1000) + "_" + "_randid_" + str(int(np.random.rand() * 1000))
        if path[-1] != "/":
            path += "/"
        self.logpath = path + timestamp + "/"
        if os.path.exists(self.logpath):
            shutil.rmtree(self.logpath)
            os.makedirs(self.logpath, exist_ok=True)
        else:
            os.makedirs(self.logpath, exist_ok=True)

    def log(self, alphaidx, lridx, epochnumidx, runnumidx, trainLoss, trainAcc, valLoss, valAcc):
        """
        Save the results of an epoch to the corresponding file.

        :param alphaidx: Index of the alpha value
        :param lridx: Index of the learning rate value
        :param epochnumidx: Index of the epoch number value
        :param runnumidx: Index of the run value
        :param trainLoss: Training loss for the given epoch
        :param trainAcc: Training accuracy for the given epoch
        :param valLoss: Validation loss for the given epoch
        :param valAcc: Validation accuracy for the given epoch
        :return:
        """
        self.TLmat[alphaidx, lridx, epochnumidx, runnumidx] = trainLoss
        np.save(self.TLFile, self.TLmat)
        self.TAmat[alphaidx, lridx, epochnumidx, runnumidx] = trainAcc
        np.save(self.TAFile, self.TAmat)
        self.VLmat[alphaidx, lridx, epochnumidx, runnumidx] = valLoss
        np.save(self.VLFile, self.VLmat)
        self.VAmat[alphaidx, lridx, epochnumidx, runnumidx] = valAcc
        np.save(self.VAFile, self.VAmat)

    def getAverageBestEpochVAcc(self):
        """
        Select the best epoch based on validation accuracy and average it by all the other dimensions.

        :return: Average validation accuracy of best epochs stored for each run
        """
        return np.average(np.max(self.VAmat, axis=2))

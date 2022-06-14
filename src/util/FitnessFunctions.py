import numpy as np
import time

from model.GaussLIFModel import GaussLIFModel
from model.IFLIFModel import IFLIFModel
from util.HyperTester import HyperTester

def fitness(ind):
    """
    Basic fitness function to use with the GaussLIF model, for training.

    :param ind: Individual and globalFitParams to evaluate
    :return: Average best epoch validation accuracy of all the runs
    """
    alpha = ind[0]
    datamanager = ind[1][0]
    logpath = ind[1][1]

    preDim, postDim = datamanager.getDataLabelDim()
    preMin, preMax = datamanager.getMinMax()
    preSize = 15

    model = GaussLIFModel(preDim, preSize, preMin, preMax, postDim, logger=None)
    tester = HyperTester(np.array([1]), np.array([0.05]), model, logpath)

    tarray = np.linspace(0, 15, 50)  # MinT, MaxT, Timestepek szama
    epochnum = 30
    repetitions = 10
    tester.alphas = np.array([alpha])
    tester.genSchedule()
    tester.run(repetitions, epochnum, datamanager, tarray, verbose=False)
    score = tester.logger.getAverageBestEpochVAcc()
    print("Alpha: ", alpha, " VAcc: ", score)
    return score


def IFfitness(ind):
    """
    Basic fitness function to use with the IFLIF model, for training.

    :param ind: Individual and globalFitParams to evaluate
    :return: Average best epoch validation accuracy of all the runs
    """
    alpha = ind[0]
    datamanager = ind[1][0]
    logpath = ind[1][1]


    preSize, postDim = datamanager.getDataLabelDim()
    preMin, preMax = datamanager.getMinMax()

    model = IFLIFModel(preSize, preMin, preMax, postDim, logger=None)
    tester = HyperTester(np.array([1]), np.array([0.01]), model, logpath)

    tarray = np.linspace(0, 15, 50)  # MinT, MaxT, Timestepek szama
    epochnum = 30
    repetitions = 5
    tester.alphas = np.array([alpha])
    tester.genSchedule()
    print("Start ",alpha,"T: ",time.time())
    tester.run(repetitions, epochnum, datamanager, tarray, verbose=False)
    score = tester.logger.getAverageBestEpochVAcc()
    print("End ", alpha,"T: ",time.time())
    print("Alpha: ", alpha, " VAcc: ", score)
    return score
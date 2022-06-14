#encoding:utf-8
"""
PSO hyperparameter optimization example.
In this file, an optimal alpha search of the Sonar dataset is performed using the GaussLIF model.
"""

from util.DataLoaderFunctions import loadSonarData
from util.DataManager import DataManager
from util.ParticleSwarmOptimizer import PSO
from util.FitnessFunctions import fitness


if __name__ == "__main__":
    data, categories = loadSonarData('./data/sonar.csv')

    popsize = 8
    dim = 1
    mins = [0.65]
    maxs = [1]
    randgbest = 0.2
    randpbest = 0.3

    sonar = DataManager(data, categories, 'sonar')
    sonar.preprocess(0.4)

    pso = PSO(popsize, dim, mins, maxs, randgbest, randpbest, fitness, globalFitParams=[sonar,"./logs/sonar"])
    pso.initPopEqually1D(0.05)
    pso.optimize(20)
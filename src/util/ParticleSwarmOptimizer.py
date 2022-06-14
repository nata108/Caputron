import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

class PSO:
    def __init__(self, popsize, dim, mins, maxs, randgbest, randpbest, fitness, globalFitParams=[]):
        """
        Base class for particle swarm optimization.

        :param popsize: Number of individuals in the swarm
        :param dim: Dimension of each individual
        :param mins: Array of lower bounds of the search space for each dimension
        :param maxs: Array of upper bounds of the search space for each dimension
        :param randgbest: Maximal random coefficient value for the gbest based acceleration component
        :param randpbest: Maximal random coefficient value for the pbest based acceleration component
        :param fitness: Fitness function
        :param globalFitParams: Array of parameters to pass for the fitness functions with the individual itself
        """
        self.popsize = popsize
        self.dim = dim
        self.mins = np.array(mins).reshape(1, dim)
        self.maxs = np.array(maxs).reshape(1, dim)
        self.pop = np.zeros((popsize, dim))
        self.vals = np.zeros((popsize, 1))
        self.pbests = np.zeros((popsize, dim))
        self.valPbests = np.zeros((popsize, 1))
        self.gbest = np.zeros((1, dim))
        self.valGbest = np.zeros((1, 1))
        self.fitness = fitness
        self.velocities = np.zeros((popsize, dim))
        self.randgbest = randgbest
        self.randpbest = randpbest
        self.globalFitParams = globalFitParams

    def initPopEqually1D(self,maxVRatio=1):
        """
        Distribute the swarm equally on a 1 dimensional line.

        :param maxVRatio: Velocities are drawn from the position distribution, then scaled with this coefficient
        :return:
        """
        self.pop = (self.maxs[0] - self.mins[0]) / (self.popsize - 1) * np.arange(self.popsize).reshape(self.popsize, 1) + \
                   self.mins[0]
        self.velocities = np.random.uniform(-(self.maxs-self.mins),(self.maxs-self.mins),self.velocities.shape)*maxVRatio
        self.evalMax(init=True)

    def step(self):
        """
        Perform a PSO step, by calculating speeds, making a step and bounding outlier individuals to the limits of the
        search space, then evaluating the swarm.

        :return:
        """

        self.calcV()
        self.pop += self.velocities
        self.bound()
        self.evalMax()

    def evalMax(self, init=False):
        """
        Evaluating the whole swarm in search of a maxima. This method uses 4 processes to evaluate 4 individuals in
        parallel.

        :param init: Set true, if this is the first evaluation and there is no global or personal bests set
        :return:
        """
        stime = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            values = [[self.pop[i, :], self.globalFitParams] for i in range(self.popsize)]
            results = executor.map(self.fitness, values)
        results = [result for result in results]
        for i in range(self.popsize):
            self.vals[i] = results[i].copy()
            if self.vals[i] > self.valPbests[i] or init:
                self.valPbests[i] = self.vals[i].copy()
                self.pbests[i, :] = self.pop[i, :].copy()

                if self.vals[i] > self.valGbest or (init and i == 0):
                    self.valGbest = self.vals[i].copy()
                    self.gbest = self.pop[i, :].copy()
        print("eval done in: ", time.time() - stime)

    def calcV(self):
        """
        Calculate the new velocities for each individual.

        :return:
        """
        self.velocities += (self.pbests - self.pop) * np.random.uniform(0, self.randpbest, size=(self.popsize, 1))
        self.velocities += (self.gbest - self.pop) * np.random.uniform(0, self.randgbest, size=(self.popsize, 1))

    def bound(self):
        """
        Bounding individual's positions for each dimensions by setting the corresponding limit value for every outlier.

        :return:
        """
        self.pop = np.where(self.pop > self.maxs, self.maxs, self.pop)
        self.pop = np.where(self.pop < self.mins, self.mins, self.pop)

    def optimize(self, iterations):
        """
        Run optimization for the given iteration count.

        :param iterations: Number of epochs to run for
        :return:
        """
        print(self.gbest,self.valGbest)
        for i in range(iterations):
            self.step()
            print(self.gbest, self.valGbest)
# Exploring the Effects of Caputo Fractional Derivative in Spiking Neural Network Training

This repository contains the supplementary code files of manuscript entitled "Exploring the Effects of Caputo Fractional Derivative in Spiking Neural Network Training".
<br />
The code found in the source folder contain Tempotron-like learning algorithms acompanied by Caputron, a fractional-order derivative-based optimizer. Utility functions for mass evaluations and hyperparameter optimization are also included.

## Implemented by<br>
Natababara Gyöngyössy <natabara@inf.elte.hu><br />
Gábor Erős <gaboreros96@gmail.com><br>
Dr. János Botzheim <botzheim@inf.elte.hu>

## Installation guide<br>
**1. step:** Clone this repository. <br>
**2. step:** Set up the environment. (Python 3.8 or later)<br>
Install the basic dependency packages<br>
```
pip install numpy
pip install scipy
```

## Manual<br>

### [Layers](/src/layer/)

Layers of spiking neurons come in 3 types in this project. All layers have a `reset()` and a `forward()` method for restoring the initial state of the neurons in the layer, and for passing information forward in the network respectively. All layers are designed to fire once per neuron, thus their refractory mechanism is simple blocking. Spiketimes of neurons are stored in the `.S` variable. Negative spiketime indicates the absence of spike in the past from the corresponding neuron.

**[Integrate-and-fire (IF) Layer](/src/layer/IFLayer.py)**<br>
IF neurons integrate the input without any loss over time. They fire above the threshold potential, and then do not change before a reset is called on them. This layer is used as an input layer for processing the MNIST dataset.

**[Multidimensional Gaussian Receptive Field Layer](/src/layer/MultidimensionalGaussReceptiveLayer.py)**<br>
Multidimensional Gaussian Receptive Field Layers define a set of Gaussian Receptive Field neurons for every input dimension. Each dimension must contain at least two neurons. After initialization the `defineGausses()` method has to be called in order to set up the receptive fields. The receptive fields are distributed equally with the minimal and maximal value being the center of the Gaussian of the first and last neurons. The STD of each Gaussian function equals to the distance between the maxima of receptive fields.

**[Leaky Integrate-and-fire (LIF) Layer](/src/layer/LIFKLayer.py)**<br>
LIF neurons integrate the output with a loss. This loss is controlled by the time constants of kernel function K. In order to calculate Tempotron loss, the maximal values of potential and kernel value are stored locally, they can be reset calling the corresponding method.

### [Models](/src/layer/)

Models are wrappers around two layers, with one trainable weight matrix between them. They all share the following methods: `predict()` performs forward steps without any optimization, `train()` performs a full training over the input data, `createOptimizer()` must be called before training the model in order to specify which optimizer to use. Winner-take-all mechanism is implemented in the `step()` method.

**[Gauss-LIF Model](/src/model/GaussLIFModel.py)**<br>
This two-layer model uses a Multidimensional Gaussian Receptive Field layer and a LIF layer in order to process multidimensional data. The model has two weight initialization functions and a built-in Tempotron-like loss function. This model was used for training on the datasets Iris, Liver and Sonar.

**[IF-LIF Model](/src/model/IFLIFModel.py)**<br>
This two-layer model uses an IF layer and a LIF layer in order to process multidimensional data. The model has two weight initialization functions and a built-in Tempotron-like loss function. This model was used for training on the MNIST dataset. The IF input layer functions as a few parameter alternative of the Gaussian Receptive Field layer, as it assigns only one neuron per input dimension.

### [Optimizers](/src/optimizer/)

Optimizers are used to adjust weight values based on the current weight values, labels and output spikes. They have a method `calcDW()` called from models when they perform a backward step. This function returns the matrix of weight value changes.

**[Gradient Descent](/src/optimizer/GradientDescent.py)**<br>
This optimizer works with the classic first order derivative of the Tempotron-loss in order to update weights. It has an adjustable learning rate (lr) parameter.

**[Caputron](/src/optimizer/Caputron.py)**<br>
This novel optimizer introduced in our paper works with Caputo fractional order derivatives from the range of ]0, 1[. It has an adjustable learning rate (lr) and an alpha parameter which refers to the derivative order to be used.

### [Utility](/src/util/)

**[Fitness Functions](/src/util/FitnessFunctions.py)**<br>
This file contains fitness functions for the PSO optimizer. Each fitness function facilitates training with a given model and DataManager for a fixed number of epochs, and repeats the process with randomly initialized weights for a given number of repetitions. It takes an iterable as input. The first element must contain the derivative order, while the second parameter is a list of globally used objects like a log directory path or the DataManager class. These are passed as globalFitParams from the PSO object.

**[Data Management](/src/util/DataManager.py)**<br>
DataManager objects parse the training data, slice them into training and validation sets and return handlers for these arrays when needed. Functions for data parsing are located in [DataLoaderFunctions.py](/src/util/DataLoaderFunctions.py). Frequently used data attributes like dimensionality and minimal, maximal values can also be queried here. 

**[Batch Testing](/src/util/HyperTester.py)**<br>
A HyperTester class is available for grid-testing with each available learning rate - alpha value pairs. This tester object initializes a Logger in order to save information from the training at the end of every epoch. Logged values are stored in a location under the log path. Each training creates a subfolder with the date and ms precise starting time converted to string as a folder name. Each subfolder contains a saved matrix of alpha and lr values, and the matrix of train and validation accuracy and loss values in a .npy format. These are readable with the `numpy.load()` function.

**[Particle Swarm Optimizer for Derivative Order](/src/util/ParticleSwarmOptimizer.py)**<br>
PSO is used for optimizing the derivative order as a hyperparameter. This class has an initializer function `initPopEqually1D()` which should be called before using `optimize()` this function spreads the individuals of the swarm equally with randomized velocities in the given range of parameter search. This range is bounded at the top and bottom, therefore any swarm members moving outside of the range is bound to the corresponding limit. A fitness function and globalFitParams can be provided to be used as the base of the optimization. As the PSO class uses parallel evaluation over 4 processes the fitness function must be in a different file separated from the main function, so the main process would not put a lock on it and deny the 4 worker processes from accessing the fitness functions.

### [Hyperparameter optimization over multiple trainings](/src/ParallelTraining.py)
This is an example of hyperparameter optimization of the derivative order alpha, with the Sonar dataset. A PSO optimizer is initialized and used to optimize with the above mentioned fitness function which facilitates batch trainings for each hyperparameter value examined.





import numpy as np

class DataManager:
    def __init__(self, data, categories, name):
        """
        Class to handle basic data preprocessing and management.

        :param data: Numpy array of the full dataset
        :param categories: Numpy array of the class labels (1D array!)
        :param name: Name for the dataset
        """
        self.data = data
        self.categories = categories
        self.name = name

    def preprocess(self, valsplit):
        """
        Basic data preprocessor, turns class labels into desired output vectors, and splits data.

        :param valsplit: Ratio of the validation dataset compared to the full dataset
        :return:
        """
        labels = []
        for label in self.categories:
            a = np.zeros(np.max(self.categories) + 1).reshape(-1, 1)
            a[label] = 1
            labels.append(a.copy())
        self.labels = np.array(labels).reshape(-1, np.max(self.categories) + 1)

        self.splitData(valsplit)

    def splitData(self, valsplit):
        """
        Splits data into two separate datasets, train and validation.

        :param valsplit:
        :return:
        """
        validx = int(self.data.shape[0] * valsplit)
        r = np.random.permutation(self.data.shape[0])
        self.data = self.data[r, ...]
        self.labels = self.labels[r, ...]

        self.valData = self.data[:validx, ...].copy()
        self.valLabels = self.labels[:validx, ...].copy()
        self.trainData = self.data[validx:, ...].copy()
        self.trainLabels = self.labels[validx:, ...].copy()

    def getTrainData(self):
        """
        Returns the train data and labels.

        :return: [trainData, trainLabels]
        """
        return self.trainData, self.trainLabels

    def getValData(self):
        """
        Returns the validation data and labels.

        :return: [validationData, validationLabels]
        """
        return self.valData, self.valLabels

    def getMinMax(self):
        """
        Returns the minimum and maximum of the dataset for each dimension of it, in this order.

        :return: [vectorOfMinimas, vectorOfMaximas]
        """
        return np.min(self.data, axis=0), np.max(self.data, axis=0)

    def getDataLabelDim(self):
        """
        Returns the data and preprocessed label dimensions.

        :return: [dataDimension, labelDimension]
        """
        return self.data.shape[1], self.labels.shape[1]
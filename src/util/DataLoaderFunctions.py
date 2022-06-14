import numpy as np

def loadIrisData(path):
    """
    Loading Iris dataset, reading csv file.

    :param path: Path at wich the dataset file is located
    :return: [Data, Labels]
    """
    data = []
    labeltexts = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        values = line.split(',')
        labeltexts.append(values[-1].replace('\n', ''))
        data.append(np.array([float(x) for x in values[:-1]]))

    uni = list(np.unique(labeltexts))
    labels = []
    for label in labeltexts:
        labels.append(uni.index(label))

    return np.array(data), np.array(labels)


def loadSonarData(path):
    """
    Loading Sonar dataset, reading csv file.

    :param path: Path at wich the dataset file is located
    :return: [Data, Labels]
    """
    data = []
    labeltexts = []
    with open(path, 'r') as f:
        lines = f.readlines()
        del lines[0]
    for line in lines:
        values = line.split(',')
        labeltexts.append(values[-1].replace('\n', ''))
        data.append(np.array([float(x) for x in values[:-1]]))

    uni = list(np.unique(labeltexts))
    labels = []
    for label in labeltexts:
        labels.append(uni.index(label))

    return np.array(data), np.array(labels)


def loadLiverData(path):
    """
    Loading Liver dataset, reading csv file.

    :param path: Path at wich the dataset file is located
    :return: [Data, Labels]
    """
    data = []
    labeltexts = []
    with open(path, 'r') as f:
        lines = f.readlines()
        del lines[0]
    for line in lines:
        values = line.split(',')
        labeltexts.append(values[-1].replace('\n', ''))
        data.append(np.array([float(x) for x in values[:-1]]))

    uni = list(np.unique(labeltexts))
    labels = []
    for label in labeltexts:
        labels.append(uni.index(label))

    return np.array(data), np.array(labels)

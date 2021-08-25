from keras.backend import expand_dims
from keras.datasets.mnist import load_data
from numpy.random import randint
from numpy.random import randn
from numpy import zeros
from numpy import ones
from tensorflow.keras.models import Sequential

class Data():
    def loadDataset(self):
        pass

    def generateRealTrainingSamples(self, dataset, samples):
        idx = randint(0, dataset.shape[0], samples)
        X = dataset[idx]
        y = ones((samples, 1))
        return X, y

    def generateFakeTrainingSamples(self, generator:Sequential, latentDim, samples):
        x = self.generateLatentPoints(latentDim, samples)
        X = generator.predict(x)
        y = zeros((samples, 1))
        return X, y

    def generateFakeTrainingGanSamples(self, generator:Sequential, latentDim, samples):
        x = self.generateLatentPoints(latentDim, samples)
        X = generator.predict(x)
        y = ones((samples, 1))
        return X, y

    def generateLatentPoints(self, latentDim, samples):
        x = randn(latentDim * samples)
        x = x.reshape((samples, latentDim))
        return x
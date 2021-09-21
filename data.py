from keras.backend import expand_dims
from keras.datasets.mnist import load_data
from numpy.random import randint
from numpy.random import randn
from numpy import zeros
from numpy import ones
from tensorflow.keras.models import Sequential

class Data():
    def loadDataset(self):
        (trainX, _), (_, _) = load_data()
        X = expand_dims(trainX, axis=-1)
        X = X.numpy().astype('float32')
        X = (X - 127.5) / 127.5 # scale from 0,255 to -1,1
        return X

    def generateRealTrainingSamples(self, dataset, samples):
        idx = randint(0, dataset.shape[0], samples)
        X = dataset[idx]
        y = ones((samples, 1))
        return X, y

    def generateRealTrainingWanSamples(self, dataset, samples):
        idx = randint(0, dataset.shape[0], samples)
        X = dataset[idx]
        y = -ones((samples, 1))
        return X, y

    def generateFakeTrainingSamples(self, generator:Sequential, latentDim, samples):
        x = self.generateLatentPoints(latentDim, samples)
        X = generator.predict(x)
        y = zeros((samples, 1))
        return X, y

    def generateFakeTrainingWanSamples(self, generator:Sequential, latentDim, samples):
        x = self.generateLatentPoints(latentDim, samples)
        X = generator.predict(x)
        y = ones((samples, 1))
        return X, y

    def generateFakeTrainingGanSamples(self, latentDim, samples):
        X = self.generateLatentPoints(latentDim, samples)
        y = ones((samples, 1))
        return X, y

    def generateFakeTrainingGanWanSamples(self, latentDim, samples):
        X = self.generateLatentPoints(latentDim, samples)
        y = -ones((samples, 1))
        return X, y

    def generateLatentPoints(self, latentDim, samples):
        x = randn(latentDim * samples)
        x = x.reshape((samples, latentDim))
        return x
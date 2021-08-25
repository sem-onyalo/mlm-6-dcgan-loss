import os
from data import Data
from generator import Generator
from discriminator import Discriminator
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Gan():
    def __init__(self, discriminator:Sequential, generator:Sequential):
        self.discriminator = discriminator
        self.generator = generator
        self.model = self.createModel()

    def createModel(self):
        discriminator.trainable = False

        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train(self, dataset, latentDim, epochs=100, batches=128):
        batchesPerEpoch = int(dataset.shape[0] / batches)
        halfBatch = int(batchesPerEpoch / 2)

        for i in range(epochs):
            for j in range(batchesPerEpoch):
                xReal, yReal = Data.generateRealTrainingSamples(dataset, halfBatch)
                dLossReal, dAccReal = self.discriminator.train_on_batch(xReal, yReal)

                xFake, yFake = Data.generateFakeTrainingSamples(latentDim, halfBatch)
                dLossFake, dAccFake = self.discriminator.train_on_batch(xFake, yFake)

                xGan, yGan = Data.generateFakeTrainingGanSamples(self.generator, batchesPerEpoch)
                gLoss = self.generator.train_on_batch(xGan, yGan)

if __name__ == '__main__':
    discriminator = Discriminator()
    generator = Generator()
    gan = Gan(discriminator.model, generator.model)
    gan.model.summary()
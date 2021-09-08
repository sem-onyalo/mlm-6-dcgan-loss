import os, time, datetime

from data import Data
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from util import Constants

class Trainer:
    def __init__(self, discriminator:Sequential, generator:Sequential, gan:Sequential, data:Data):
        self.discriminator = discriminator
        self.generator = generator
        self.gan = gan
        self.data = data

        self.dLossRealHistory = list()
        self.dAccRealHistory = list()
        self.dLossFakeHistory = list()
        self.dAccFakeHistory = list()
        self.gLossHistory = list()
        self.metricsHistory = list()

    def train(self, latentDim, epochs=100, batchSize=256, evalFreq=10):
        if not os.path.exists('eval'):
            os.makedirs('eval')

        print('\nTraining parameters:')
        print('\tLatent dim:', latentDim)
        print('\tEpochs:', epochs)
        print('\tBatch size:', batchSize)
        print('\tEvaluation frequency:', evalFreq)
        print('==================================================\n')

        dataset = self.data.loadDataset()
        batchesPerEpoch = int(dataset.shape[0] / batchSize)
        halfBatch = int(batchSize / 2)

        self.plotStartingImageSamples(latentDim)

        self.startTime = time.time()

        for i in range(epochs):
            for j in range(batchesPerEpoch):
                xReal, yReal = self.data.generateRealTrainingSamples(dataset, halfBatch)
                dLossReal, dAccReal = self.discriminator.train_on_batch(xReal, yReal)

                xFake, yFake = self.data.generateFakeTrainingSamples(self.generator, latentDim, halfBatch)
                dLossFake, dAccFake = self.discriminator.train_on_batch(xFake, yFake)

                xGan, yGan = self.data.generateFakeTrainingGanSamples(latentDim, batchSize)
                gLoss = self.gan.train_on_batch(xGan, yGan)

                metrics = ('>%d, %d/%d, dLossReal=%.3f, dLossFake=%.3f, gLoss=%.3f' % 
                    (i + 1, j + 1, batchesPerEpoch, dLossReal, dLossFake, gLoss))

                print(metrics)

                self.dLossRealHistory.append(dLossReal)
                self.dAccRealHistory.append(dAccReal)
                self.dLossFakeHistory.append(dLossFake)
                self.dAccFakeHistory.append(dAccFake)
                self.gLossHistory.append(gLoss)
                self.metricsHistory.append(metrics)

            if (i + 1) % evalFreq == 0:
                self.evaluate(i, dataset, latentDim)
                self.metricsHistory = list() # save mini-batch metrics separately

            self.printElapsedTime()

    def evaluate(self, epoch, dataset, latentDim, samples=150):
        xReal, yReal = self.data.generateRealTrainingSamples(dataset, samples)
        _, accReal = self.discriminator.evaluate(xReal, yReal)

        xFake, yFake = self.data.generateFakeTrainingSamples(self.generator, latentDim, samples)
        _, accFake = self.discriminator.evaluate(xFake, yFake)

        print(epoch, accReal, accFake)
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (accReal * 100, accFake * 100))

        filename = 'eval/generated_model_e%03d.h5' % (epoch + 1)
        self.generator.save(filename)

        metricsFilename = 'eval/metrics_e%03d.txt' % (epoch + 1)
        with open(metricsFilename, "w") as fd:
            for i in self.metricsHistory:
                fd.write(i + '\n')

        self.plotImageSamples(xFake, epoch)

        self.plotHistory(epoch)

    def plotStartingImageSamples(self, latentDim, samples=150):
        xFake, _ = self.data.generateFakeTrainingSamples(self.generator, latentDim, samples)
        self.plotImageSamples(xFake, -1)

    def plotImageSamples(self, samples, epoch, n=10):
        scaledSamples = (samples + 1) / 2.0 # scale from -1,1 to 0,1

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledSamples[i, :, :, 0], cmap='gray_r')

        filename = 'eval/generated_plot_e%03d.png' % (epoch + 1)
        pyplot.savefig(filename)
        pyplot.close()

    def plotHistory(self, epoch):
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.dLossRealHistory, label='dLossReal')
        pyplot.plot(self.dLossFakeHistory, label='dLossFake')
        pyplot.plot(self.gLossHistory, label='gLoss')
        pyplot.legend()

        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.dAccRealHistory, label='accReal')
        pyplot.plot(self.dAccFakeHistory, label='accFake')
        pyplot.legend()

        pyplot.savefig('eval/loss_acc_history_e%03d.png' % (epoch + 1))
        pyplot.close()

    def printElapsedTime(self):
        elapsedTime = time.time() - self.startTime
        print('Elapsed time:', str(datetime.timedelta(seconds=elapsedTime)))
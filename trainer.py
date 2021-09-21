import os, time, datetime

from data import Data
from matplotlib import pyplot
from numpy import mean
from tensorflow.keras.models import Sequential
from util import Constants

class Trainer:
    def __init__(self, discriminator:Sequential, generator:Sequential, gan:Sequential, data:Data, loss):
        self.discriminator = discriminator
        self.generator = generator
        self.gan = gan
        self.data = data
        self.loss = loss

        print('\nDiscriminator\n')
        self.discriminator.summary()
        
        print('\nGenerator\n')
        self.generator.summary()
        
        print('\nGAN\n')
        self.gan.summary()
        print()

        self.resetHistoryLists()

    def train(self, latentDim, epochs=100, batchSize=256, evalFreq=10, criticIterations=5):
        if not os.path.exists('eval'):
            os.makedirs('eval')

        dataset = self.data.loadDataset()
        halfBatch = int(batchSize / 2)
        batchesPerEpoch = int(dataset.shape[0] / batchSize)
        trainingIterations = epochs * batchesPerEpoch

        print('\nTraining parameters:')
        print('\tGAN loss:', self.loss)
        print('\tLatent dim:', latentDim)
        print('\tEpochs:', epochs)
        print('\tBatch size:', batchSize)
        print('\tBatches per epoch:', batchesPerEpoch)
        print('\tEpoch evaluation frequency:', evalFreq)
        print('\tTraining iterations:', trainingIterations)
        print('==================================================\n')

        if self.loss == Constants.WASSERSTEIN_LOSS:
            self.trainCriticArchitecture(latentDim, batchSize, (evalFreq * batchesPerEpoch), 
                dataset, trainingIterations, batchesPerEpoch, halfBatch, criticIterations)
        else:
            self.trainDiscriminatorArchitecture(latentDim, batchSize, (evalFreq * batchesPerEpoch), 
                dataset, trainingIterations, batchesPerEpoch, halfBatch)

    def trainDiscriminatorArchitecture(self, latentDim, batchSize, evalFreq, 
        dataset, trainingIterations, batchesPerEpoch, discriminatorBatchSize):

        self.plotStartingImageSamples(latentDim)

        self.startTime = time.time()

        for i in range(trainingIterations):
            xReal, yReal = self.data.generateRealTrainingSamples(dataset, discriminatorBatchSize)
            dLossReal, dAccReal = self.discriminator.train_on_batch(xReal, yReal)

            xFake, yFake = self.data.generateFakeTrainingSamples(self.generator, latentDim, discriminatorBatchSize)
            dLossFake, dAccFake = self.discriminator.train_on_batch(xFake, yFake)

            xGan, yGan = self.data.generateFakeTrainingGanSamples(latentDim, batchSize)
            gLoss = self.gan.train_on_batch(xGan, yGan)

            currentEpoch = int(i / batchesPerEpoch)
            metrics = ('>%d, %d/%d, dLossReal=%.3f, dLossFake=%.3f, gLoss=%.3f' % 
                (currentEpoch + 1, i + 1, trainingIterations, dLossReal, dLossFake, gLoss))
            print(metrics)

            self.dLossRealHistory.append(dLossReal)
            self.dAccRealHistory.append(dAccReal)
            self.dLossFakeHistory.append(dLossFake)
            self.dAccFakeHistory.append(dAccFake)
            self.gLossHistory.append(gLoss)
            self.metricsHistory.append(metrics)

            if (i + 1) % evalFreq == 0:
                self.evaluate(currentEpoch, dataset, latentDim)
                self.metricsHistory = list() # save mini-batch metrics separately
                self.printElapsedTime()

        self.printElapsedTime()

    def trainCriticArchitecture(self, latentDim, batchSize, evalFreq, 
        dataset, trainingIterations, batchesPerEpoch, criticBatchSize, criticIterations):

        self.plotStartingImageSamples(latentDim)

        self.startTime = time.time()

        for i in range(trainingIterations):
            dLossRealHistoryTemp, dAccRealHistoryTemp = list(), list()
            dLossFakeHistoryTemp, dAccFakeHistoryTemp = list(), list()
            for _ in range(criticIterations):
                xReal, yReal = self.data.generateRealTrainingWanSamples(dataset, criticBatchSize)
                dLossReal, dAccReal = self.discriminator.train_on_batch(xReal, yReal)

                xFake, yFake = self.data.generateFakeTrainingWanSamples(self.generator, latentDim, criticBatchSize)
                dLossFake, dAccFake = self.discriminator.train_on_batch(xFake, yFake)

                dLossRealHistoryTemp.append(dLossReal)
                dAccRealHistoryTemp.append(dAccReal)
                dLossFakeHistoryTemp.append(dLossFake)
                dAccFakeHistoryTemp.append(dAccFake)

            xGan, yGan = self.data.generateFakeTrainingGanWanSamples(latentDim, batchSize)
            gLoss = self.gan.train_on_batch(xGan, yGan)

            dLossRealMean = mean(dLossRealHistoryTemp)
            dAccRealMean = mean(dAccRealHistoryTemp)
            dLossFakeMean = mean(dLossFakeHistoryTemp)
            dAccFakeMean = mean(dAccFakeHistoryTemp)

            currentEpoch = int(i / batchesPerEpoch)
            metrics = ('>%d, %d/%d, dLossReal=%.3f, dLossFake=%.3f, gLoss=%.3f' % 
                (currentEpoch + 1, i + 1, trainingIterations, dLossRealMean, dLossFakeMean, gLoss))
            print(metrics)

            self.dLossRealHistory.append(dLossRealMean)
            self.dAccRealHistory.append(dAccRealMean)
            self.dLossFakeHistory.append(dLossFakeMean)
            self.dAccFakeHistory.append(dAccFakeMean)
            self.gLossHistory.append(gLoss)
            self.metricsHistory.append(metrics)

            if (i + 1) % evalFreq == 0:
                self.evaluate(currentEpoch, dataset, latentDim)
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

    def resetHistoryLists(self):
        self.dLossRealHistory = list()
        self.dAccRealHistory = list()
        self.dLossFakeHistory = list()
        self.dAccFakeHistory = list()
        self.gLossHistory = list()
        self.metricsHistory = list()
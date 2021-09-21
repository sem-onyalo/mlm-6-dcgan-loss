'''
DCGAN - MNIST Grayscale Handwritten Digits.

Ref: https://machinelearningmastery.com/generative_adversarial_networks/
'''

import argparse
from data import Data
from model import createGan
from trainer import Trainer
from util import Constants

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latentDim', '-d', type=int, default=100, help='Latent space dimension')
    parser.add_argument('--loss', '-l', type=str, default='ns', help='GAN loss type (ns, ls, ws)')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=128, help='The training batch size')
    parser.add_argument('--evalfreq', '-v', type=int, default=10, help='Frequency to run model evaluations')
    args = parser.parse_args()

    dInputShape = (28,28,1)
    dInputDim = 7

    ganLoss = Constants.ganLossType[args.loss]
    gan, discriminator, generator = createGan(dInputShape, dInputDim, args.latentDim, ganLoss)
    data = Data()

    trainer = Trainer(discriminator, generator, gan, data, ganLoss)
    trainer.train(args.latentDim, epochs=args.epochs, batchSize=args.batchsize, evalFreq=args.evalfreq)
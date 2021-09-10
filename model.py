from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, Dropout, Flatten, ReLU, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from util import Constants

def createDiscriminator(input, loss=Constants.NON_SATURATING_LOSS, batchNorm=True)->Sequential:
    model = Sequential()
    init = RandomNormal(stddev=0.02) #, kernel_initializer=init
    
    model.add(Conv2D(64, (4,4), (2,2), padding='same', kernel_initializer=init, input_shape=input)) # --> 14x14
    if batchNorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (4,4), (2,2), padding='same', kernel_initializer=init)) # --> 7x7
    if batchNorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    if loss == Constants.NON_SATURATING_LOSS:
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    elif loss == Constants.LEAST_SQUARES_LOSS:
        model.add(Flatten())
        model.add(Dense(1, activation='linear', kernel_initializer=init))
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    else:
        raise Exception('invalid GAN loss: ' + loss)

    return model

def createGenerator(input, inputDim, batchNorm=True, leakyRelu=False)->Sequential:
    model = Sequential()
    init = RandomNormal(stddev=0.02)

    inputFilters = 256
    inputNodes = inputDim * inputDim * inputFilters
    model.add(Dense(inputNodes, kernel_initializer=init, input_dim=input))
    if batchNorm:
        model.add(BatchNormalization())
    if leakyRelu:
        model.add(LeakyReLU(0.2))
    else:
        model.add(ReLU())
    model.add(Reshape((inputDim, inputDim, inputFilters)))

    model.add(Conv2DTranspose(128, (4,4), (2,2), padding='same', kernel_initializer=init)) # --> 14x14
    if batchNorm:
        model.add(BatchNormalization())
    if leakyRelu:
        model.add(LeakyReLU(0.2))
    else:
        model.add(ReLU())
    
    model.add(Conv2DTranspose(64, (4,4), (2,2), padding='same', kernel_initializer=init)) # --> 28x28
    if batchNorm:
        model.add(BatchNormalization())
    if leakyRelu:
        model.add(LeakyReLU(0.2))
    else:
        model.add(ReLU())
    
    model.add(Conv2D(1, (7,7), padding='same', activation='tanh', kernel_initializer=init)) # -1,1 activation function

    return model

def createGan(discriminator:Sequential, generator:Sequential, loss=Constants.NON_SATURATING_LOSS):
    # discriminator.trainable = False
    for layer in discriminator.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    if loss == Constants.NON_SATURATING_LOSS:
        model.compile(loss='binary_crossentropy', optimizer=opt)
    elif loss == Constants.LEAST_SQUARES_LOSS:
        model.compile(loss='mse', optimizer=opt)
    else:
        raise Exception('invalid GAN loss: ' + loss)

    return model

if __name__ == '__main__':
    discriminator = createDiscriminator((28,28,1))
    generator = createGenerator(100, 7)
    gan = createGan(discriminator, generator)

    print('\nDiscriminator\n')
    discriminator.summary()
    
    print('\nGenerator\n')
    generator.summary()
    
    print('\nGAN\n')
    gan.summary()
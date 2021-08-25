from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, LeakyReLU, Reshape
from tensorflow.keras.models import Sequential

class Generator:
    def __init__(self, input=100, inputDim=7, inputFilters=128, convFilters=128):
        self.model = self.createModel(input, inputDim, inputFilters, convFilters)

    def createModel(self, input, inputDim, inputFilters, convFilters):
        model = Sequential()

        inputNodes = inputDim * inputDim * inputFilters
        model.add(Dense(inputNodes, input_dim=input))
        model.add(LeakyReLU(0.2))
        model.add(Reshape((inputDim, inputDim, inputFilters)))

        model.add(Conv2DTranspose(convFilters, (4,4), (2,2), padding='same')) # --> 14x14
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2DTranspose(convFilters, (4,4), (2,2), padding='same')) # --> 28x28
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(1, (7,7), padding='same', activation='tanh'))
        return model

if __name__ == '__main__':
    generator = Generator()
    generator.model.summary()
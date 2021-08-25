from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Discriminator:
    def __init__(self, input=(28,28,1), convFilters=64):
        self.model = self.createModel(input, convFilters)

    def createModel(self, input, convFilters):
        model = Sequential()
        
        model.add(Conv2D(convFilters, (3,3), (2,2), padding='same', input_shape=input)) # --> 14x14
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(convFilters, (3,3), (2,2), padding='same', input_shape=input)) # --> 7x7
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

if __name__ == '__main__':
    discriminator = Discriminator()
    discriminator.model.summary()
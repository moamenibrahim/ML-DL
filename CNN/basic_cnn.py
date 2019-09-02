from keras.datasets import mnist
import numpy as np
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from keras.models import Sequential

# Images fed into this model are 512 x 512 pixels with 3 channels
img_shape = (28, 28, 1)

# Set up the model
model = Sequential()

# Add convolutional layer with 3, 3 by 3 filters and a stride size of 1
# Set padding so that input size equals output size
model.add(Conv2D(6, 2, input_shape=img_shape))

# Add relu activation to the layer
model.add(Activation('relu'))

# Pooling
model.add(MaxPool2D(2))

# Fully connected layers
# Use Flatten to convert 3D data to 1D
model.add(Flatten())

# Add dense layer with 10 neurons
model.add(Dense(10))

# we use the softmax activation function for our last layer
model.add(Activation('softmax'))

# give an overview of our model
model.summary

"""Before the training process,
we have to put together a learning process in a particular form.
It consists of 3 elements: an optimiser, a loss function and a metric."""

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

# dataset with handwritten digits to train the model on
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1)
# Train the model, iterating on the data in batches of 32 samples# for 10 epochs
x_test = np.expand_dims(x_test, -1)
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Train the model, iterating on the data in batches of 32 samples
# for 10 epochs
model.fit(x_train, y_train, batch_size=32, epochs=10,
          validation_data=(x_test, y_test))

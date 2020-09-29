import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import TensorBoard
import pickle
# import time

#train_name = "Emotions-happy-sad-{}".format(int(time.time()))

#PATH = os.path.join('UpdateLogs', train_name)
#tensor_board = TensorBoard(log_dir=PATH)

training_set = pickle.load(open("training_data.pickle", "rb"))
test_set = pickle.load(open("testing_data.pickle", "rb"))

training_set = training_set/255.0

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=training_set.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for layer in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for layers in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            # model.add(Flatten())
            # model.add(Dense(64))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

            model.fit(training_set, test_set, batch_size=10, epochs=8, validation_split=0.2)


# model.fit(training_set, test_set, batch_size=2, epochs=6,
# validation_split=0.5, callbacks=[tensor_board])

model.save('emotions-CNN.model')

#https://pythonprogramming.net/using-trained-model-deep-learning-python-tensorflow-keras/
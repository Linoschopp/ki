import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
import numpy as np
import requests
import datetime
import models

epochs = 14

#x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [0], [0], [1]])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

took = None

class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        self.start = None
        self.last_predicted_time = ()

    def on_train_begin(self, logs=None):
        self.start = datetime.datetime.now()
    def on_epoch_end(self, epoch, logs=None):
        """if (epoch + 1) % 50 == 0 and epoch < (epochs - 1):
            done = datetime.datetime.now() - self.start
            time_per_epoch = done / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            remaining_time = time_per_epoch * remaining_epochs
            requests.post("https://ntfy.sh/linoschopp",
                              data=f"Done at: {(datetime.datetime.now() + remaining_time).hour}:{(datetime.datetime.now() + remaining_time).minute}",
                              headers={"Title":f"Finished epoch {epoch + 1}"}
                         )"""
        if (epoch == -1):
            done = datetime.datetime.now() - self.start
            time_per_epoch = done / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            remaining_time = time_per_epoch * remaining_epochs
            finish_time = datetime.datetime.now() + remaining_time
            self.last_predicted_time = (finish_time.hour, finish_time.minute, finish_time.second)
            requests.post("https://ntfy.sh/linoschopp",
                              data=f"Done at: {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}",
                              headers={"Title":"Training started"}
                         )
        elif True:#(epoch + 1) % 50 == 0:
            done = datetime.datetime.now() - self.start
            time_per_epoch = done / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            remaining_time = time_per_epoch * remaining_epochs
            finish_time = datetime.datetime.now() + remaining_time
            if (finish_time.hour, finish_time.minute, finish_time.second) != self.last_predicted_time:
                self.last_predicted_time = (finish_time.hour, finish_time.minute, finish_time.second)
                requests.post("https://ntfy.sh/linoschopp",
                                  data=f"Done at: {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}",
                                  headers={"Title":f"Training: {((epoch+1)/epochs):%}"}
                             )
        
        
    def on_train_end(self, logs=None):
        global took
        took = datetime.datetime.now() - self.start


model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add( Conv2D(64, kernel_size=(3, 3), activation="relu" ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add( Conv2D(128, kernel_size=(3, 3), activation="relu" ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x=x_train, y=y_train, epochs=epochs)#, callbacks=[CustomCallback()], verbose=1)
loss, acc = model.evaluate(x=x_test, y=y_test, verbose=1)
#requests.post("https://ntfy.sh/linoschopp",
#                  data=f"Took: {took}\nAccuracy: {acc:%}",
#                  headers={"Title":f"Finished training"}
#             )
print(f"Model accuracy: {acc:%}")
models.ask_for_save(model)
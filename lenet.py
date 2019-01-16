from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation
from keras.datasets import mnist
from keras import backend as k
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

# Defining a LeNet Class
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        # Defining the model to be used
        model = Sequential()
        # Adding the layers
        model.add(Conv2D(20, kernel_size=5, activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(50, kernel_size=5, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dense(classes, activation='softmax'))
        return model
# Hyperparameter Definition
BATCH_SIZE=128
EPOCHS = 10
IP_SHAPE = (1, 28, 28)
k.set_image_dim_ordering('th')
# Loading and preprocessing dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
# Building the model
model = LeNet.build(input_shape=IP_SHAPE, classes=10)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, validation_split=0.25)
score=model.evaluate(X_test, y_test, verbose=True)
print("Test Score : ", score[0])
print("Test Accuracy : ", score[1])
print(history.history.keys())
# Accuracy Visualization
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("LeNet Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

import alex_hyperparameters as hp


class myModel(tf.keras.Model):
    def __init__(self):
        super(myModel, self).__init__()
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.learning_rate, momentum=hp.momentum)

        self.architecture = [
            # Block 1
            Conv2D(filters=32, kernel_size=(5, 5), padding='same', strides=(1, 1), activation='relu'),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),

            # Block 2
            Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),

            # Block 3
            Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
            Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
            MaxPool2D(pool_size=(4, 4)),
            Dropout(rate=0.3),

            # Block 4
            Flatten(),
            Dense(units=512, activation='relu'),

            # Block 5
            Dense(units=hp.num_classes, activation='softmax')
        ]

    def call(self, x):
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)

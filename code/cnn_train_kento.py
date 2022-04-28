from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D
import numpy as np
import matplotlib.pyplot as plt
import os
from get_data import get_data

class Model(tf.keras.Model):
    """CNN that is similar to VGG and has parameters to specify feature or texture selection.
    It is first trained as a classification model of different skin cancers.
    Then the CNN (withoout dense layers) is used for feature and texture selection."""
    def __init__(self):
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 9

        self.lr = 1E-3 # learning rate for optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.dropoutrate = 0.3
        self.epsilon = 1E-5
        self.epochs = 1

        # Call layers
        # Block 1
        self.block1_conv1 = Conv2D(32, 3, 1, padding="same", activation="relu", name="block1_conv1")
        self.block1_conv2 = Conv2D(32, 3, 1, padding="same", activation="relu", name="block1_conv2")
        # Block 2
        self.block2_conv1 = Conv2D(64, 3, 1, padding="same", activation="relu", name="block2_conv1")
        self.block2_conv2 = Conv2D(64, 3, 1, padding="same", activation="relu", name="block2_conv2")
        # Block 3
        self.block3_conv1 = Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv1")
        self.block3_conv2 = Conv2D(124, 3, 1, padding="same", activation="relu", name="block3_conv2")
        self.block3_conv3 = Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv3")
        # Block 4
        self.block4_conv1 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block4_conv1")
        self.block4_conv2 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block4_conv2")
        self.block4_conv3 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block4_conv3")
        # Block 5
        self.block5_conv1 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block5_conv1")
        self.block5_conv2 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block5_conv2")
        self.block5_conv3 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block5_conv3")

        # Dense Layers for Classification
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu', name='dense1')
        self.dense2 = Dense(9, activation='softmax', name='dense2')

        # loss function
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()

        # initialize latents
        self.latents = {}

    def save_latent(self, x, name):
        """Save specified latent space to dictionary.
        
        :param x: ouput of specified layer
        :param name: string, key of dictionary --> usually block or layer name
        :returns: none"""
        self.latents[name] = x
        pass

    def pool(style, name):
        """Pooling layer depending on style or content.
        
        :param style: True or False
        :param name: string to name the layer
        :returns: pool layer"""
        if style: # if true avgpool
            return AvgPool2D(2, name=name)
        else:
            return MaxPool2D(2, name=name)

    def call(self, inputs, dense=False, style=False, is_testing=False):
        
        # CNN block 1
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.pool(style, 'block1_pool')(x)
        if style: self.save_latent(x, style, 'block1')
        
        # CNN block 2
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.pool(style, 'block2_pool')(x)
        if style: self.save_latent(x, style, 'block2')

        # CNN block 3
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.pool(style, 'block3_pool')(x)
        if style: self.save_latent(x, style, 'block3')

        # CNN block 4
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.pool(style, 'block4_pool')(x)
        if style: self.save_latent(x, style, 'block4')

        # CNN block 5
        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.pool(style, 'block5_pool')(x)
        if style: self.save_latent(x, style, 'block5')

        # Dense Layers for Classification
        if dense:
            x = self.flatten(x)
            x = self.dense1(x)
            probs = self.dense2(x)
            return probs
        else:
            pass #?

    def loss(self, probs, labels):
        """Loss function using Categorical Crossentropy. Averages over batch.

        :param probs: probabilties resulting from model.call()
        :param labels: true labels of dataset
        :return: loss"""

        return tf.reduce_mean(self.class_loss(labels, probs))

    def accuracy(self, probs, labels):
        """Accuracy function. Compares true label to the category with largest probability. 
        Averages over batch.
        
        :param probs: probabilties resulting from model.call()
        :param labels: true labels of dataset
        :return: accuracy"""

        correct_predictions = tf.equal(tf.argmax(probs, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    """Train function for categorization of skin cancers given images

    :param model: CNN model to train
    :param train_inputs: training input images
    :param train_labels: one-hot labels of training images
    :return: none
    """
    inds = tf.range(0, train_inputs.shape[0], dtype=tf.int32) # indices representing images
    shuffled_inds = tf.random.shuffle(inds) # shuffled indices
    shuffled_inputs = tf.gather(train_inputs, shuffled_inds) # correlate indices to images/labels
    shuffled_labels = tf.gather(train_labels, shuffled_inds)

    # number of inputs for batching
    tot_size = train_inputs.shape[0]
    
    i = 0 # starting image index; initialize from 0
    ## start batching process ##
    while i < tot_size + model.batch_size - 1:
        if i + model.batch_size < tot_size:
            inputs_slice = shuffled_inputs[i:i+model.batch_size]
            labels_slice = shuffled_labels[i:i+model.batch_size]
        else: # just in case the tot_size - i is less than batch_size
            inputs_slice = shuffled_inputs[i:]
            labels_slice = shuffled_labels[i:]
        # end batching process ##

        with tf.GradientTape() as tape:
            probs = model.call(inputs_slice, dense=True, style=False, is_testing=False)
            loss = model.loss(probs, labels_slice)

            # print accuracy every 100 images
            if i % 100 == 0:
                train_acc = model.accuracy(probs, labels_slice)
                print(f'Accuracy on training set after {i} images: {train_acc}')

            grads = tape.gradient(loss, model.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        i += model.batch_size
    return None


def test(model, test_inputs, test_labels):
    """Train function for categorization of skin cancers given images

    :param model: CNN model to test
    :param train_inputs: testing input images
    :param train_labels: one-hot labels of testing images
    :return: Accuracy"""

    probs = model.call(test_inputs, dense=True, style=False, is_testing=True)
    return model.accuracy(probs, test_labels)


def main():
    """
    :return: none"""
    train_inputs, train_labels = get_data('../data/train')
    test_inputs, test_labels = get_data('../data/test')

    model = Model()
    # cycle through epochs
    for e in range(model.epochs):
        print(f'Start Epoch #{e+1}')
        train(model, train_inputs, train_labels)

    test_acc = test(model, test_inputs, test_labels)
    
    print(f'Test accuracy is = {test_acc*100} %')
    return


if __name__ == '__main__':
    main()
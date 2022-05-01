from __future__ import absolute_import
from tarfile import is_tarfile
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

        self.lr = 0.001 # learning rate for optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.dropoutrate = 0.3
        self.epsilon = 1E-5
        self.epochs = 3

        # Call layers
        self.dropout = tf.keras.layers.Dropout(self.dropoutrate)
        # Block 1
        self.block1_conv1 = Conv2D(8, 3, 1, padding="same", activation="relu", name="block1_conv1")
        self.block1_conv2 = Conv2D(8, 3, 1, padding="same", activation="relu", name="block1_conv2")
        # Block 2
        self.block2_conv1 = Conv2D(32, 3, 1, padding="same", activation="relu", name="block2_conv1")
        self.block2_conv2 = Conv2D(32, 3, 1, padding="same", activation="relu", name="block2_conv2")
        # Block 3
        self.block3_conv1 = Conv2D(64, 3, 1, padding="same", activation="relu", name="block3_conv1")
        self.block3_conv2 = Conv2D(64, 3, 1, padding="same", activation="relu", name="block3_conv2")
        self.block3_conv3 = Conv2D(64, 3, 1, padding="same", activation="relu", name="block3_conv3")
        # Block 4
        self.block4_conv1 = Conv2D(128, 3, 1, padding="same", activation="relu", name="block4_conv1")
        self.block4_conv2 = Conv2D(128, 3, 1, padding="same", activation="relu", name="block4_conv2")
        self.block4_conv3 = Conv2D(128, 3, 1, padding="same", activation="relu", name="block4_conv3")
        # Block 5
        self.block5_conv1 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block5_conv1")
        self.block5_conv2 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block5_conv2")
        self.block5_conv3 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block5_conv3")

        # Dense Layers for Classification
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu', name='dense1')
        self.dense2 = Dense(64, activation='relu', name='dense2')
        self.dense3 = Dense(9, activation='softmax', name='dense3')

        # loss function
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()

        # initialize latents
        self.latents = {}
        self.feature_latent = {}

    def save_latent(self, x, name):
        """Save specified latent space to dictionary.
        
        :param x: ouput of specified layer
        :param name: string, key of dictionary --> usually block or layer name
        :returns: none"""
        self.latents[name] = x
        pass

    def pool(self, style, name):
        """Pooling layer depending on style or content.
        
        :param style: True or False
        :param name: string to name the layer
        :returns: pool layer"""
        if style: # if true avgpool
            return AvgPool2D(2, name=name)
        else:
            return MaxPool2D(2, name=name)
    
    def batch_norm(self, x):
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        x = tf.nn.batch_normalization(x, mean, variance,  None, None, variance_epsilon=self.epsilon)
        return x

    def call(self, inputs, dense=False, style=False, is_training=False, feature=False):
        
        # CNN block 1
        x = self.block1_conv1(inputs)
        x = self.batch_norm(x)
        # x = self.dropout(x, training=is_training)
        x = self.block1_conv2(x)
        x = self.batch_norm(x)
        # x = self.dropout(x, training=is_training)
        x = self.pool(style, 'block1_pool')(x)
        if style: self.save_latent(x, 'block1')
        
        # CNN block 2
        x = self.block2_conv1(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x, training=is_training)
        x = self.block2_conv2(x)
        # x = self.batch_norm(x)
        x = self.pool(style, 'block2_pool')(x)
        if style: self.save_latent(x, 'block2')

        # CNN block 3
        x = self.block3_conv1(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x, training=is_training)
        x = self.block3_conv2(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x, training=is_training)
        x = self.block3_conv3(x)
        # x = self.batch_norm(x)
        x = self.pool(style, 'block3_pool')(x)
        if style: self.save_latent(x, 'block3')

        # CNN block 4
        x = self.block4_conv1(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x, training=is_training)
        x = self.block4_conv2(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x, training=is_training)
        x = self.block4_conv3(x)
        # x = self.batch_norm(x)
        x = self.pool(style, 'block4_pool')(x)
        # if style: self.save_latent(x, 'block4')
        if feature: self.feature_latent['block4'] = x

        # CNN block 5
        x = self.block5_conv1(x)
        x = self.batch_norm(x)
        x = self.dropout(x, training=is_training)
        x = self.block5_conv2(x)
        x = self.batch_norm(x)
        x = self.dropout(x, training=is_training)
        x = self.block5_conv3(x)
        x = self.batch_norm(x)
        x = self.pool(style, 'block5_pool')(x)
        if style: self.save_latent(x, 'block5')

        # Dense Layers for Classification
        if dense:
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dropout(x, training=is_training)
            x = self.dense2(x)
            x = self.dropout(x, training=is_training)
            probs = self.dense3(x)
            return probs
        
        if style:
            latents = self.latents
            return latents

        elif feature:
            feature_latent = self.feature_latent
            return feature_latent

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


def train(model, train_dataset):
    """Train function for categorization of skin cancers given images

    :param model: CNN model to train
    :param train_inputs: training input images
    :param train_labels: one-hot labels of training images
    :return: none
    """
    
    b = 1 # number batches
    buffer_sz = 3000
    # number of inputs for batching
    train_dataset = train_dataset.shuffle(buffer_sz)
    for train_inputs, train_labels in train_dataset:
        train_inputs /= 255.0 # normalize pixel values

        with tf.GradientTape() as tape:
            probs = model.call(train_inputs, dense=True, style=False, is_training=True, feature=False)
            loss = model.loss(probs, train_labels)

        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        acc = model.accuracy(probs, train_labels)
        print(f'batch, {b} accuracy: {acc*100}%')
        b += 1

    return None


def test(model, test_dataset):
    """Train function for categorization of skin cancers given images

    :param model: CNN model to test
    :param train_inputs: testing input images
    :param train_labels: one-hot labels of testing images
    :return: Accuracy"""

    for test_inputs, test_labels in test_dataset:
        test_inputs /= 255.0 # normalize pixel values
    probs = model.call(test_inputs, dense=True, style=False, is_training=False, feature=False)
    return model.accuracy(probs, test_labels).numpy()


def main():
    """
    :return: run training and testing"""

    model = Model()
    train_dataset, test_dataset = get_data('../ISIC_data/Train/', '../ISIC_data/Test/', 
                                            batch_sz=model.batch_size, shuffle=True, 
                                            image_sz=(256,256)) # try changing to (600,450) on GPU

    # cycle through epochs
    for e in range(model.epochs):
        print(f'Start Epoch #{e+1}')
        train(model, train_dataset)
        acc = test(model, test_dataset)
        print(f'Test Accuracy after epoch {e+1}: {acc*100}%')
        model.save_weights('../checkpoints/weights.h5')
        print('Weights Saved!')

    test_acc = test(model, test_dataset)
    
    print(f'Test accuracy is = {test_acc*100} %')

    # I don't think we need this
    # for m in model.layers:
    #     weights = np.array(m.get_weights())
    #     np.save(f'../checkpoints/{m.name}.npz', weights)

    
if __name__ == '__main__':
    main()
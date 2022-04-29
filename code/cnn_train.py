from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from get_data import get_data

class Model(tf.keras.Model):
    """CNN that is similar to VGG and has parameters to specify feature or texture selection.
    It is first trained as a classification model of different skin cancers.
    Then the CNN (withoout dense layers) is used for feature and texture selection."""
    def __init__(self):
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 9

        self.lr = 0.0005 # learning rate for optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()
        self.dropoutrate = 0.3
        self.epochs = 50

        # Call layers
        self.dropout = tf.keras.layers.Dropout(self.dropoutrate)
        # Block 1
        self.block1_conv1 = Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1")
        self.block1_conv2 = Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2")
        # Block 2
        self.block2_conv1 = Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1")
        self.block2_conv2 = Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2")
        # Block 3
        self.block3_conv1 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1")
        self.block3_conv2 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2")
        self.block3_conv3 = Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3")
        # Block 4
        self.block4_conv1 = Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1")
        self.block4_conv2 = Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2")
        self.block4_conv3 = Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3")
        # Block 5
        self.block5_conv1 = Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1")
        self.block5_conv2 = Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2")
        self.block5_conv3 = Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3")

        # Dense Layers for Classification
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu', name='dense1')
        self.dense2 = Dense(512, activation='relu', name='dense2')
        self.dense3 = Dense(9, activation='softmax', name='dense3')

        # initialize latents
        self.latents = {}
    
    def save_latent(self, x, style, name):
        if style:
            self.latents[name] = x

    def pool(self, style, name):
        if style: 
            return AvgPool2D(2, name=name)
        else:
            return MaxPool2D(2, name=name)


    def call(self, inputs, dense=False, style=False, is_training=False):
        
        # CNN block 1
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.pool(style, 'block1_pool')(x)
        self.save_latent(x, style, 'block1')
        
        # CNN block 2
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.pool(style, 'block2_pool')(x)
        self.save_latent(x, style, 'block2')

        # CNN block 3
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.pool(style, 'block3_pool')(x)
        self.save_latent(x, style, 'block3')

        # CNN block 4
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.pool(style, 'block4_pool')(x)
        self.save_latent(x, style, 'block4')

        # CNN block 5
        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.pool(style, 'block5_pool')(x)
        x = self.dropout(x, training=is_training)
        self.save_latent(x, style, 'block5')

        # Dense Layers for Classification
        if dense:
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            return x

    def loss_fn(self, probs, labels):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
    
    def accuracy(self, probs, labels):
        metric = tf.keras.metrics.sparse_categorical_accuracy(labels, probs)
        return list(metric).count(1)/len(metric)

def train(model, train_images, traing_labels):
    for images, labels in zip(train_images, traing_labels):
        with tf.GradientTape() as tape:
            images = tf.image.random_flip_left_right(images)
            probs = model.call(images, dense=True, style=False, is_training=True)
            loss = model.loss_fn(probs, labels)
            acc = model.accuracy(probs, labels)
            print(acc)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_images, test_labels):
    accuracy = []
    for images, labels in zip(test_images, test_labels):
        probs = model.call(images, dense=True, style=False, is_training=False)
        probs = np.array(probs, dtype=np.float)
        labels = np.array(labels, dtype=np.float)
        loss = model.loss_fn(probs, labels)
        acc = model.accuracy(probs, labels)
        accuracy.append(acc)
        print(loss, acc)
    return np.average(accuracy)

def main():
    model = Model()
    train_dataset, test_dataset = get_data('../ISIC_data/Train/', '../ISIC_data/Test/', batch_sz=model.batch_size, shuffle=True, image_sz=(256,256))
    train_images, traing_labels = preprocess(train_dataset)
    print("Finished train preprocess")
    test_images, test_labels = preprocess(test_dataset)
    print("Finished test preprocess")

    for e in range(model.epochs):
        print('Epoch #', e)
        train(model, train_images, traing_labels)
        acc = test(model, test_images, test_labels)
        print('Test Accuracy after epoch', e, ':', acc)
        model.save_weights('../checkpoints/weights.h5')
        print('Weights Saved!')
        model.save_model('../checkpoints/model.h5')
        print('Model Saved!')

def preprocess(data):
    image_data = []
    label_data = []
    for batch, (image, label) in (enumerate(data)):
        image_data.append(image/255.0)
        label_data.append(np.argmax(label, axis=1))
    image_data.pop()
    label_data.pop()
    return image_data, label_data

if __name__ == '__main__':
    main()
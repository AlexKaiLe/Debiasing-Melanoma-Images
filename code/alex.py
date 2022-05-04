import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import backend as K

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 64
        self.num_classes = 9
        self.dropoutrate = 0.3
        self.epsilon = 1E-5
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()

    def loss(self, probs, labels):
        return tf.reduce_mean(self.loss_function(labels, probs))
    
    def accuracy(self, probs, labels):
        correct_predictions = tf.equal(tf.argmax(probs, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def batch_norm(self, x):
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        return tf.nn.batch_normalization(x, mean, variance,  None, None, variance_epsilon=self.epsilon)

    def call(self, input_tensor, dense=False):
        # Block 1
        x = self.batch_norm(input_tensor)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        x = Dropout(self.dropoutrate)(x)

        if dense:
            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(9, activation='softmax', name='predictions')(x)
        return x

def train(model, train_dataset):  
    train_dataset = train_dataset.shuffle(3000)  
    for batch, (train_inputs, train_labels) in enumerate(train_dataset):
        train_inputs /= 255.0
        with tf.GradientTape() as tape:
            probs = model.call(train_inputs, dense=True)
            loss = model.loss(probs, train_labels)

        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        acc = float(model.accuracy(probs, train_labels))*100
        print("batch", batch, "accuracy", acc)
    

def test(model, test_dataset):
    total_acc = []
    for test_inputs, test_labels in test_dataset:
        test_inputs /= 255.0 
        probs = probs = model.call(test_inputs, dense=True)
        total_acc.append(model.accuracy(probs, test_labels))
    return np.average(total_acc)

def main():
    epochs = 50
    batch_sz = 100
    image_sz = (224, 224)
    train_file = "../ISIC_data/Train"
    test_file = "../ISIC_data/Test"
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_file, labels='inferred',color_mode='rgb',label_mode='categorical',batch_size=batch_sz, shuffle=True, image_size=image_sz)
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_file, labels='inferred', color_mode='rgb', label_mode='categorical', batch_size=batch_sz, shuffle=False, image_size=image_sz)
    
    model = Model()
    best_weight = 0
    for e in range(epochs):
        print(f'Starting Epoch #{e+1}')
        train(model, train_dataset)
        acc = test(model, test_dataset)
        print(f'Validation accuracy epoch {e+1}: {acc*100}%')
        
        if acc > best_weight:
            model.save_weights('../checkpoints/alex_best_weights_1.h5')
            print('alex_best_weights_1 Saved!', acc)
            best_weight = acc
        else:
            model.save_weights('../checkpoints/alex_weights_1.h5')
            print('alex_weights Saved!')
    
    test_acc = test(model, test_dataset)
    print(f'Test accuracy is = {test_acc*100} %')

if __name__ == '__main__':
    main()
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import alex_hyperparameters as hp
import os
import re

class myModel(tf.keras.Model):
    def __init__(self):
       super(myModel, self).__init__()
       self.train = True
       self.dense = True
       self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.learning_rate, momentum=hp.momentum)
       
       self.architecture = [
       # Block 1
       Conv2D(filters=32, kernel_size=(5, 5), padding='same', strides=(1, 1), activation='relu', name="block1_conv1", input_shape = (1, 224,224,3)),
       Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', name="block1_conv2"),
       MaxPool2D(pool_size=(2, 2), name="block1_pool"),

       # Block 2
       Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', name="block2_conv1"),
       Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', name="block2_conv2"),
       MaxPool2D(pool_size=(2, 2), name="block2_pool"),

       # Block 3
       Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', name="block3_conv1"),
       Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', name="block3_conv2"),
       MaxPool2D(pool_size=(4, 4), name="block3_pool"),
       Dropout(rate=0.3, name="block3_dropout"),

       # Block 4
       Flatten(name="block4_flatten", trainable = self.train),
       Dense(units=512, activation='relu', name="block4_dense"),

       # Block 5
       Dense(units=hp.num_classes, activation='softmax', name="block5_dense")
       ]
       

    def call(self, x):
       # for layer in self.architecture:
       #        x = layer(x)
       # return x
       
       if self.dense:
              MLP = 13
       else:
              MLP = 10
       for b, layer in enumerate(self.architecture):
              if b<MLP:
                     x = layer(x)
       return x

    @staticmethod
    def loss_fn(labels, predictions):
       return tf.keras.losses.sparse_categorical_crossentropy(
       labels, predictions, from_logits=False)

class modelSaver(tf.keras.callbacks.Callback):

    def __init__(self, checkpoint_dir, max_num_weights=5):
       super(modelSaver, self).__init__()
       self.checkpoint_dir = checkpoint_dir
       self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
       min_acc_file, max_acc_file, max_acc, num_weights = self.scan_weight_files()
       cur_acc = logs["val_sparse_categorical_accuracy"]

       if cur_acc > max_acc:
              save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(epoch, cur_acc)

       
       self.model.save_weights(self.checkpoint_dir + os.sep + "alex." + save_name)
       if self.max_num_weights > 0 and num_weights + 1 > self.max_num_weights:
              os.remove(self.checkpoint_dir + os.sep + min_acc_file)

    def scan_weight_files(self):
       min_acc = float('inf')
       max_acc = 0
       min_acc_file = ""
       max_acc_file = ""
       num_weights = 0

       files = os.listdir(self.checkpoint_dir)
       for weight_file in files:
              if weight_file.endswith(".h5"):
                     num_weights += 1
                     file_acc = float(re.findall(
                     r"[+-]?\d+\.\d+", weight_file.split("acc")[-1])[0])
                     if file_acc > max_acc:
                            max_acc = file_acc
                            max_acc_file = weight_file
                     if file_acc < min_acc:
                            min_acc = file_acc
                            min_acc_file = weight_file

       return min_acc_file, max_acc_file, max_acc, num_weights

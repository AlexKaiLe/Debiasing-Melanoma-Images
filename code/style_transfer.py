from __future__ import absolute_import
from tarfile import is_tarfile
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D
import numpy as np
import matplotlib.pyplot as plt
import os
from cnn_train_kento import Model

def main():
    model_init = Model() # old model with dense layers
    model = Model() # just the CNN (no dense layers)
    layer_names = ['dropout', 'block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 
                'block3_conv1', 'block3_conv2', 'block3_conv3', 
                'block4_conv1', 'block4_conv2', 'block4_conv3', 
                'block5_conv1', 'block5_conv2', 'block5_conv3', 
                'flatten', 'dense1', 'dense2', 'dense3']
    
    # for i, m in enumerate(model.layers[1:]):
    #     m.set_weights(np.load(f'../checkpoints/{m.name}.npy', allow_pickle=True))
    model_init(tf.zeros((1,256,256,3)), dense=True) 
    model(tf.zeros((1,256,256,3)), dense=False) 
    model_init.load_weights('../checkpoints/weights.h5') # load weights into old model
    for i, m in enumerate(model_init.layers[:-4]): # exclude layres from flatten --> dense3
        model.layers[i].set_weights(model_init.layers[i].get_weights())
    model.summary()
    
if __name__ == '__main__':
    main()
from __future__ import absolute_import
from tarfile import is_tarfile
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D
import numpy as np
import matplotlib.pyplot as plt
import os
from cnn_train_kento import Model, test
from get_data import get_data
# from preprocess import preprocess

def clear_latents(model):
    model.feature_latent = {}
    model.latents = {}

def main():
    model_init = Model() # old model with dense layers
    model = Model() # just the CNN (no dense layers)
    layer_names = ['dropout', 'block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 
                'block3_conv1', 'block3_conv2', 'block3_conv3', 
                'block4_conv1', 'block4_conv2', 'block4_conv3', 
                'block5_conv1', 'block5_conv2', 'block5_conv3', 
                'flatten', 'dense1', 'dense2', 'dense3']
    
    model_init(tf.zeros((1,256,256,3)), dense=True) 
    model(tf.zeros((1,256,256,3)), dense=False) 
    model_init.load_weights('../checkpoints/weights.h5') # load weights into old model
    for i, m in enumerate(model_init.layers[:-4]): # exclude layres from flatten --> dense3
        model.layers[i].set_weights(model_init.layers[i].get_weights())
    del model_init
    model.summary()
    train_dataset, test_dataset = get_data('../ISIC_data/Train/', '../ISIC_data/Test/', 
                                            batch_sz=1, shuffle=False, 
                                            image_sz=(256,256)) # try changing to (600,450) on GPU
    test_acc = test(model, test_dataset)
    print(f'Initial accuracy of model on classification: {test_acc*100} %')

    ##### STYLE PREPROCESS #####
    # style_path = '../skin_image_data/'
    # style_image = preprocess(style_path)
    for style_image, style_label in train_dataset:
        style_image/255.
        break
    ############################

    style_latents_dict = model.call(style_image, dense=False, style=True, is_training=False, feature=False)
    # print(style_latents_dict)

    ##### FEATURE PREPROCESS #####
    ## might need to add another argument for returning the latent space after a specific block
    ## becuase rn if I set style=True, it gives me a dictionary, but pooling is AVG not MAX
    feature_latents_dict = model.call(style_image, dense=False, style=False, is_training=False, feature=True)
    print(model.feature_latent)
    ###############################

    ##### GENERATE STATIC IMAGE #####
    generated_image = tf.Variable(tf.random.truncated_normal((1,256,256,3), mean=0., stddev=1))
    #################################

    
    
if __name__ == '__main__':
    main()
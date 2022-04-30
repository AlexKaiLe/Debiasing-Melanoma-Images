from __future__ import absolute_import
from re import X
from tarfile import is_tarfile
from click import style
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D
import numpy as np
import matplotlib.pyplot as plt
import os
from cnn_train_kento import Model, test
from get_data import get_data


class StyleTransfer:
    def __init__(self, model, input_image, style_image, feature_image):
        self.model = model # CNN model
        self.image = input_image # input image that is static --> to be changed
        self.style_image = style_image # image with style/texture
        self.feature_image = feature_image # image with features

        self.lossratio = 0.001 # Feature/Style
        self.num_iter = 20 # number of iterations

        # self.style_layer

        self.block_id = ['block1', 'block2', 'block3', 'block4', 'block5']

        self.style_latents_dict = model.call(style_image, dense=False, style=True, is_training=False, feature=False)
        self.feature_latents_dict = model.call(feature_image, dense=False, style=False, is_training=False, feature=True)
        
        self.lr = 0.001
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

    def train_step(self, input_image):
        with tf.GradientTape() as tape:
            x_style = self.model.call(input_image, dense=False, style=True, is_training=False, feature=True)
            x_feature = self.model.call(input_image, dense=False, style=False, is_training=False, feature=True)

            L_total = self.loss(x_style, x_feature)
        
            grads = tape.gradient(L_total, input_image)
            self.optimizer.apply_gradients(zip(grads, input_image))
            self.image.assign(tf.clip_by_value(input_image, clip_value_min=0.0, clip_value_max=1.0))

    
    def loss(self, x_style, x_feature):
        L_style = 0
        L_feature = 0

        for id in self.block_id:
            if id in self.style_latents_dict:

                A = self.style_latents_dict[id]
                F = x_style[id]

                _, h, w, M = np.shape(F)
                N = h*w

                G = self.gram_matrix(tf.convert_to_tensor(F))

                ### they multiply this value by weight for the layer, idk why look into it ###
                L_style += (1./(4*N**2*M**2))*tf.reduce_sum(tf.pow(G-A, 2))

            if id in self.feature_latents_dict:
                
                P = self.feature_latents_dict[id]
                F = x_feature[id]

                L_feature = 0.5*tf.reduce_sum(tf.pow(F-P, 2))

        alpha = self.loss_ratio
        beta = 1

        L_total = alpha*L_feature + beta*L_style
        return L_total


    def gram_matrix(self, input_tensor):
        # gram_matrix from https://www.tensorflow.org/tutorials/generative/style_transfer
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/num_locations


def clear_latents(model):
    model.feature_latent = {}
    model.latents = {}

def main():
    model_init = Model() # old model with dense layers
    model = Model() # just the CNN (no dense layers)
    
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

    ##### FEATURE PREPROCESS #####
    for feature_image, labels in train_dataset:
        feature_image/255.
        break

    ##### STYLE PREPROCESS #####
    style_image = tf.keras.preprocessing.image.load_img('../preprocessed_images/img13.jpg', target_size=(256, 256))
    style_image = tf.Variable(tf.reshape(tf.keras.preprocessing.image.img_to_array(style_image), (1, 256, 256, 3))/255., trainable=False)

    ##### GENERATE STATIC IMAGE #####
    generated_image = tf.Variable(tf.random.uniform((1,256,256,3), minval=0, maxval=1), name='Generated_Image', trainable=True)

    styletransfer = StyleTransfer(model, generated_image, style_image, feature_image)
    for i in range(styletransfer.num_iter):
        styletransfer.train_step(styletransfer.image)
    
    styletransfer.image*225.0
    
    
if __name__ == '__main__':
    main()
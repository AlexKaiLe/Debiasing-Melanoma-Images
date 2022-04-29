import numpy as np
import tensorflow as tf
import os

def get_data(train_file, test_file, batch_sz, shuffle, image_sz):
    """Get data from ISIC test/train files.
    
    :param train_file: file path to train data
    :param test_file: file path to test data
    :param batch_sz: batch size (int)
    :param shuffle: True or False (shuffle the data or not)
    :param image_sz: size of image, default is (256,256)
    
    :return: tuple of train_dataset, test_dataset as tf BatchDataset classes"""
    if image_sz is None:
        image_sz = (256, 256)
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_file, 
                                                                labels='inferred', 
                                                                label_mode='categorical', 
                                                                batch_size=batch_sz, 
                                                                shuffle=shuffle,
                                                                image_size=image_sz)
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_file, 
                                                                labels='inferred', 
                                                                label_mode='categorical', 
                                                                batch_size=batch_sz, 
                                                                shuffle=shuffle,
                                                                image_size=image_sz)
    return train_dataset, test_dataset
import numpy as np
import tensorflow as tf
import os

def get_data(train_file, test_file, batch_sz, shuffle):
    """Get data from ISIC test/train files.
    
    :param train_file: file path to train data
    :param test_file: file path to test data
    :param batch_sz: batch size (int)
    :param shuffle: True or False (shuffle the data or not)
    
    :return: tuple of train_dataset, test_dataset as tf BatchDataset classes"""
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_file, 
                                                                labels='inferred', 
                                                                label_mode='categorical', 
                                                                batch_size=batch_sz, 
                                                                shuffle=shuffle)
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_file, 
                                                                labels='inferred', 
                                                                label_mode='categorical', 
                                                                batch_size=batch_sz, 
                                                                shuffle=shuffle)
    return train_dataset, test_dataset
import os
import cv2
import os
import glob
import gc
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical

def gather_images(img_dir, xdim, ydim, nmax=5000) :
    label = 0
    label_names = []
    X = []
    y=[]
    print(os.listdir(img_dir))
    for dirname in os.listdir(img_dir):
        print(dirname)
        if dirname != ".DS_Store":
            label_names.append(dirname)
            data_path = img_dir + "/" + dirname + "/*"
            # print(data_path)
            files = glob.glob(data_path)
            for i, f1 in enumerate(files):
                img = cv2.imread(f1) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (xdim,ydim)) 
                X.append(np.array(img)) 
                y.append(label) 
            print(i+1,'images')
            label += 1
    X = np.array(X)
    y = np.array(y)
    gc.collect() 
    return X, y

def shuffle(x_data, y_data):
    print("Shuffling data")
    temp = list(zip(x_data, y_data))
    random.shuffle(temp)
    x_data, y_data = zip(*temp)
    return np.array(x_data), np.array(y_data)

def get_model():
    model_cnn = Sequential()

    # Block 1
    model_cnn.add(Conv2D(32, 3, input_shape=(224, 224, 3), padding="same", activation="relu", name="block1_conv1"))
    # model_cnn.add(Dropout(0.3))
    model_cnn.add(Conv2D(32, 3, 1, padding="same", activation="relu", name="block1_conv2"))
    # model_cnn.add(Dropout(0.3))
    model_cnn.add(MaxPool2D(2))
    # Block 2
    model_cnn.add(Conv2D(64, 3, padding="same", activation="relu", name="block2_conv1"))
    # model_cnn.add(Dropout(0.3))
    model_cnn.add(Conv2D(64, 3, 1, padding="same", activation="relu", name="block2_conv2"))
    # model_cnn.add(Dropout(0.3))
    model_cnn.add(MaxPool2D(2))
    # Block 3
    model_cnn.add(Conv2D(128, 3, padding="same", activation="relu", name="block3_conv1"))
    # model_cnn.add(Dropout(0.3))
    model_cnn.add(Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv2"))
    # model_cnn.add(Dropout(0.3))
    model_cnn.add(MaxPool2D(2))
    # Block 4
    # model_cnn.add(Conv2D(256, 3, 1, padding="same", activation="relu", name="block4_conv1"))
    # # model_cnn.add(Dropout(0.3))
    # model_cnn.add(Conv2D(256, 3, 1, padding="same", activation="relu", name="block4_conv2"))
    # # model_cnn.add(Dropout(0.3))
    # model_cnn.add(MaxPool2D(2))
    # # Block 5
    # model_cnn.add(Conv2D(256, 3, 1, padding="same", activation="relu", name="block5_conv1"))
    # # model_cnn.add(Dropout(0.3))
    # model_cnn.add(Conv2D(256, 3, 1, padding="same", activation="relu", name="block5_conv2"))
    # # model_cnn.add(Dropout(0.3))
    # model_cnn.add(MaxPool2D(2))

    # Dense Layers for Classification
    model_cnn.add(Dropout(0.3))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu', name='dense1'))
    model_cnn.add(Dense(64, activation='relu', name='dense2'))
    model_cnn.add(Dense(9, activation='softmax', name='dense3'))
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, momentum=0.01)
    model_cnn.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    return model_cnn

def loss_fn(labels, predictions):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=False)

def main():
    print("train images")
    X_train, y_train = gather_images("../ISIC_data/Train", 224, 224, 1000)
    print("tests images")
    X_test,y_test = gather_images("../ISIC_data/Test", 224, 224,1000)
    X_train = X_train / 255
    X_test = X_test / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    model = get_model()
    train_cnn = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=200, verbose=1)
    model.summary()
    scores = train_cnn.evaluate(X_test, y_test, verbose=0)
    print("Score : %.2f%%" % (scores[1]*100))
    model.save_weights('../checkpoints/weights.h5')
    print('Weights Saved!')

if __name__ == '__main__':
    main()
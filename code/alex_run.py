import os
import sys
import tensorflow as tf
from alex_models import YourModel
from alex_preprocess import Datasets
import alex_hyperparameters as hp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(model, datasets, init_epoch):
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    model.evaluate( x=test_data,verbose=1,)


def main():
    init_epoch = 0

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = Datasets("../ISIC_data")

    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    train(model, datasets, init_epoch)
    model.save_weights('../checkpoints/new_weights.h5')
    print('new_weights Saved!')
    test(model, datasets.test_data)

main()

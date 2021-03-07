import os
from absl import app

import tensorflow as tf

from alexnet.alexnet_model import alexnet
from alexnet.alexnet_preprocessing import build_train_dataset


IMAGENET_DIRECTORY = '/data/image-net'

TFRECORD_TRAINING_DIRECTORY = 'tfrecord_train'
TFRECORD_VALIDATION_DIRECTORY = 'tfrecord_val'

BATCH_SIZE = 128


def main(_):
    # create model
    model = alexnet()
    model.summary()

    # load dataset
    dataset_train = build_train_dataset(
        os.path.join(IMAGENET_DIRECTORY, TFRECORD_TRAINING_DIRECTORY),
        BATCH_SIZE
    )

    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')
    losses = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=opt, loss=losses, metrics=metrics)

    history = model.fit(dataset_train, epochs=1, verbose=1)


if __name__ == '__main__':
    tf.compat.v2.enable_v2_behavior()
    app.run(main)

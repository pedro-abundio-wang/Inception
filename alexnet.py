import os
from absl import app

import tensorflow as tf

from alexnet.alexnet_model import alexnet
from alexnet.alexnet_preprocessing import build_train_dataset
from alexnet.alexnet_preprocessing import build_valid_dataset


IMAGENET_DIRECTORY = '/data/image-net'

TFRECORD_TRAINING_DIRECTORY = 'tfrecord_train'
TFRECORD_VALIDATION_DIRECTORY = 'tfrecord_val'

BATCH_SIZE = 128

# Dataset constants
_NUM_TRAIN_IMAGES = 1281167
_NUM_EVAL_IMAGES = 50000

def main(_):

    # create model
    model = alexnet()
    model.summary()

    # load dataset
    dataset_train = build_train_dataset(
        os.path.join(IMAGENET_DIRECTORY, TFRECORD_TRAINING_DIRECTORY),
        BATCH_SIZE
    )

    dataset_val = build_valid_dataset(
        os.path.join(IMAGENET_DIRECTORY, TFRECORD_VALIDATION_DIRECTORY),
        BATCH_SIZE
    )

    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')
    losses = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=opt, loss=losses, metrics=metrics)

    learning_rate_schedules = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=0.0001)

    callbacks = [
        learning_rate_schedules,
    ]

    # trained the network for roughly 90 cycles through the training set of 1.2 million images
    history = model.fit(dataset_train, validation_data=dataset_val,
                        epochs=1, callbacks=callbacks, verbose=1)

if __name__ == '__main__':
    tf.compat.v2.enable_v2_behavior()
    app.run(main)

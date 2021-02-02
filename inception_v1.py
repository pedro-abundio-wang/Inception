# Copyright 2021 The Pedro Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inception V1 model for Keras.
Reference:
  - [Going Deeper with Convolutions](
      https://arxiv.org/abs/1409.4842)
"""

from tensorflow import keras

def InceptionV1(
        input_shape=None,
        classes=1000,
        classifier_activation='softmax'):
    """Instantiates the Inception v1 architecture.

    Reference:
    - [Going Deeper with Convolutions](
        https://arxiv.org/abs/1409.4842)

    Note that the data format convention used by the model is
    the one specified in the `tf.keras.backend.image_data_format()`.

    Arguments:
      input_shape: the input shape should be
        `(224, 224, 3)` (with `channels_last` data format) or
        `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      classes: number of classes to classify images into.
        Default to 1000.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Set `classifier_activation=None` to return the logits
        of the "top" layer.

    Returns:
      A `keras.Model` instance.
    """
    img_input = keras.layers.Input(shape=input_shape, name='input')

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # stage1
    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='stage1_conv7x7')(img_input)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stage1_pool')(x)

    # stage2
    x = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage2_conv3x3_reduce')(x)
    x = keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage2_conv3x3')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stage2_pool')(x)

    # stage3a
    branch1x1 = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage3a_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage3a_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage3a_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage3a_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage3a_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage3a_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage3a_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage3a')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage3b
    branch1x1 = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage3b_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage3b_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage3b_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage3b_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage3b_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage3b_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage3b_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage3b')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage3_pool
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stage3_pool')(x)

    # stage4a
    branch1x1 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4a_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4a_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=208, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage4a_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4a_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage4a_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage4a_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4a_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage4a')([branch1x1, branch3x3, branch5x5, branch_pool])

    # aux classifier (stage4a)
    y = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='stage4a_aux_pool')(x)
    y = keras.layers.Conv2D(filter=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4a_aux_conv')(y)
    y = keras.layers.Dense(units=1024, activation='relu', name='stage4a_aux_dense')(y)
    y = keras.layers.Dropout(rate=0.7, name='stage4a_aux_dropout')(y)
    y = keras.layers.Dense(units=1000, activation=classifier_activation, name='stage4a_aux_classifier')(y)

    # stage4b
    branch1x1 = keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4b_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=112, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4b_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=224, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage4b_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4b_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage4b_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage4b_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4b_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage4b')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage4c
    branch1x1 = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4c_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4c_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage4c_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4c_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage4c_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage4c_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4c_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage4c')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage4d
    branch1x1 = keras.layers.Conv2D(filters=112, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4d_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=144, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4d_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=288, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage4d_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4d_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage4d_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage4d_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4d_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage4d')([branch1x1, branch3x3, branch5x5, branch_pool])

    # aux classifier (stage4d)
    z = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='stage4d_aux_pool')(x)
    z = keras.layers.Conv2D(filter=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4d_aux_conv')(z)
    z = keras.layers.Dense(units=1024, activation='relu', name='stage4d_aux_dense')(z)
    z = keras.layers.Dropout(rate=0.7, name='stage4d_aux_dropout')(z)
    z = keras.layers.Dense(units=1000, activation=classifier_activation, name='stage4d_aux_classifier')(z)

    # stage4e
    branch1x1 = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4e_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4e_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=320, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage4e_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4e_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage4e_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage4e_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage4e_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage4e')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage4_pool
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stage4_pool')(x)

    # stage5a
    branch1x1 = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage5a_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage5a_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=320, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage5a_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage5a_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage5a_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage5a_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage5a_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage5a')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage5b
    branch1x1 = keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage5b_conv1x1')(x)

    branch3x3 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage5b_conv3x3_reduce')(x)
    branch3x3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage5b_conv3x3')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage5b_conv5x5_reduce')(x)
    branch5x5 = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='stage5b_conv5x5')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='stage5b_pool')(x)
    branch_pool = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage5b_pool_reduce')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage5b')([branch1x1, branch3x3, branch5x5, branch_pool])

    # classifier
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = keras.layers.Dropout(rate=0.4, name='dropout')(x)
    x = keras.layers.Dense(units=classes, activation=classifier_activation, name='predictions')(x)

    # Create model.
    inputs = img_input
    model = keras.Model(inputs=inputs, outputs=[x, y, z], name='inception_v1')

    return model


def preprocess_input(x, data_format=None):
    return keras.applications.imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


def decode_predictions(preds, top=5):
    return keras.applications.imagenet_utils.decode_predictions(preds, top=top)


# compile model
model = InceptionV1(input_shape=(224, 224, 3))

model.summary()

losses = {
    'category_output': 'sparse_categorical_crossentropy',
    'stage4a_aux': 'sparse_categorical_crossentropy',
    'stage4d_aux': 'sparse_categorical_crossentropy',
}
losses_weights = {
    'category_output': 1.0,
    'stage4a_aux': 0.3,
    'stage4d_aux': 0.3,
}

opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')

model.compile(optimizer=opt, loss=losses, loss_weights=losses_weights, metrics=['accuracy'])

# train model

trainX = None
trainY = None
testX = None
testY = None
EPOCHS = None

H = model.fit(x=trainX, y={'category_output': trainY, 'stage4a_aux': trainY, 'stage4d_aux': trainY},
              validation_data=(testX, {'category_output': trainY, 'stage4a_aux': trainY, 'stage4d_aux': trainY}),
              epochs=EPOCHS,
              verbose=1)

# save the model to disk
# print("[INFO] serializing network...")
# model.save(args["model"], save_format="h5")
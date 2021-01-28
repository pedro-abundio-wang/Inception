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
    img_input = keras.layers.Input(shape=input_shape)

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # stage_1
    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='stage_1_conv1')(img_input)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stage_1_pool')(x)

    # stage_2
    x = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='stage_2_conv1')(x)
    x = keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stage_2_conv2')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stage_2_pool')(x)

    # stage_3_a
    branch1x1 = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_3_a')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage_3_b
    branch1x1 = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_3_b')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage_3_pool
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stage_3_pool')(x)

    # stage_4_a
    branch1x1 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=208, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_4_a')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage_4_b
    branch1x1 = keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=112, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=224, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_4_b')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage_4_c
    branch1x1 = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_4_c')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage_4_d
    branch1x1 = keras.layers.Conv2D(filters=112, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=144, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=288, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_4_d')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage_4_e
    branch1x1 = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=320, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_4_e')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage_4_pool
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stage_4_pool')(x)

    # stage_5_a
    branch1x1 = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=320, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_5_a')([branch1x1, branch3x3, branch5x5, branch_pool])

    # stage_5_b
    branch1x1 = keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

    branch5x5 = keras.layers.Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

    branch_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pool)

    x = keras.layers.Concatenate(axis=channel_axis, name='stage_5_b')([branch1x1, branch3x3, branch5x5, branch_pool])

    # classifier
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(units=classes, activation=classifier_activation, name='predictions')(x)

    # Create model.
    inputs = img_input
    model = keras.Model(inputs=inputs, outputs=x, name='inception_v1')

    return model


def preprocess_input(x, data_format=None):
    return keras.applications.imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


def decode_predictions(preds, top=5):
    return keras.applications.imagenet_utils.decode_predictions(preds, top=top)
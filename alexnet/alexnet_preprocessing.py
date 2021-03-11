"""
ImageNet consists of variable-resolution images, while our system requires a constant input dimensionality.

Therefore, we down-sampled the images to a fixed resolution of 256x256.

Given a rectangular image, we first rescaled the image such that the shorter side was of length 256,
and then cropped out the central patch from the resulting image.

We did not pre-process the images 256x256 in any other way,
except for subtracting the mean activity over the training set from each pixel.

So we trained our network on the (centered) raw RGB values of the pixels.
"""

import os
from typing import Tuple
import tensorflow as tf


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_IMAGENET_SIZE = 227
_NUM_PARALLEL_CALLS = 4
_DATASET_BUFFER_SIZE = 8 * 1024 * 1024
_CYCLE_LENGTH = 4


def image_mean_subtraction(image, means):
    """Subtracts the given means from each image channel.
    For example:
      means = [123.68, 116.78, 103.94]
      image = image_mean_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.
    Returns:
      the centered image.
    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def preprocess(image_bytes: tf.Tensor,
               label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decodes and reshapes a raw image Tensor."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    center_crop_size = tf.cast(tf.minimum(image_height, image_width), tf.int32)

    offset_height = (image_height - center_crop_size + 1) // 2
    offset_width = (image_width - center_crop_size + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            center_crop_size, center_crop_size])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    image = tf.image.resize(
        image, [_IMAGENET_SIZE, _IMAGENET_SIZE],
        method=tf.image.ResizeMethod.BILINEAR)

    # mean subtraction
    image = image_mean_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

    return image, label


def parse_record(record: tf.train.Example) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses an ImageNet TFRecord."""
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, ''),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64, -1),
    }

    features = tf.io.parse_single_example(record, keys_to_features)
    image_bytes = tf.reshape(features['image/encoded'], shape=[])
    label = tf.cast(tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    return image_bytes, label


def build_train_dataset(data_path: str, batch_size: int) -> tf.data.Dataset:
    """Builds a standard ImageNet dataset with only reshaping preprocessing."""
    file_pattern = os.path.join(data_path, 'tfrecord_train-0000*-of-01024')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    dataset = dataset.interleave(
        lambda name: tf.data.TFRecordDataset(name, buffer_size=_DATASET_BUFFER_SIZE),
        cycle_length=_CYCLE_LENGTH,
        num_parallel_calls=_NUM_PARALLEL_CALLS)

    dataset = dataset.map(
        lambda x: preprocess(*parse_record(x)),
        num_parallel_calls=_NUM_PARALLEL_CALLS
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def data_augmentation(dataset_train):
    """by extracting random 227x227 patches (and their horizontal reflections) from the
    256x256 images and training our network on these extracted patches
    This increases the size of our training set by a factor of 2048"""


def build_valid_dataset(data_path: str, batch_size: int) -> tf.data.Dataset:
    """Builds a standard ImageNet dataset with only reshaping preprocessing."""
    file_pattern = os.path.join(data_path, 'tfrecord_val-0000*-of-00128')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)

    dataset = dataset.interleave(
        lambda name: tf.data.TFRecordDataset(name, buffer_size=_DATASET_BUFFER_SIZE),
        cycle_length=_CYCLE_LENGTH,
        num_parallel_calls=_NUM_PARALLEL_CALLS)

    dataset = dataset.map(
        lambda x: preprocess(*parse_record(x)),
        num_parallel_calls=_NUM_PARALLEL_CALLS
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset
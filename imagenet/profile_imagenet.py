"""Simple script that profiles ImageNet data loading from GCS."""
import os
import time
from typing import Tuple

from absl import app
from absl import logging

import tensorflow as tf

# The width/height of an ImageNet image.
_IMAGENET_SIZE = 224
_CROP_PADDING = 32
_CENTER_CROP_RATIO = _IMAGENET_SIZE / (_IMAGENET_SIZE + _CROP_PADDING)

IMAGENET_DIRECTORY = '/data/image-net'

TFRECORD_TRAINING_DIRECTORY = 'tfrecord_train'
TFRECORD_VALIDATION_DIRECTORY = 'tfrecord_val'

BATCH_SIZE = 1024
NUM_STEPS = 100


def preprocess(image_bytes: tf.Tensor,
               label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decodes and reshapes a raw image Tensor."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        (_CENTER_CROP_RATIO *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return tf.compat.v1.image.resize(
        image, [_IMAGENET_SIZE, _IMAGENET_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False), label


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


def build_tf_dataset(data_path: str, batch_size: int) -> tf.data.Dataset:
    """Builds a standard ImageNet dataset with only reshaping preprocessing."""
    file_pattern = os.path.join(data_path, 'tfrecord_train-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    dataset = dataset.interleave(
        lambda name: tf.data.TFRecordDataset(name, buffer_size=buffer_size),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=1024).repeat()
    dataset = dataset.map(lambda x: preprocess(*parse_record(x)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(8)
    return dataset


def benchmark_dataset(dataset: tf.data.Dataset, num_steps: int) -> float:
    """Benchmarks a tf.data.Dataset for a certain number of a steps."""
    logging.info('Starting dataset benchmarks.')
    it = iter(dataset)
    start_time = time.perf_counter()
    for _ in range(num_steps):
        next(it)
    end_time = time.perf_counter()
    return end_time - start_time


def run(data_dir: str, batch_size: int, num_steps: int):
    """Creates and logs the benchmark of data loading."""
    logging.info('Data dir: %s', data_dir)
    logging.info('Batch size: %d', batch_size)
    logging.info('Number of steps: %d', num_steps)
    dataset = build_tf_dataset(data_path=data_dir,
                               batch_size=batch_size)
    benchmarks = benchmark_dataset(dataset, num_steps=num_steps)
    logging.info('Benchmarks: Completed in %f s.', benchmarks)
    return benchmarks


def main(_):
    run(data_dir=os.path.join(IMAGENET_DIRECTORY, TFRECORD_TRAINING_DIRECTORY),
        batch_size=BATCH_SIZE,
        num_steps=NUM_STEPS)


if __name__ == '__main__':
    tf.compat.v2.enable_v2_behavior()
    app.run(main)



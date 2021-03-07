import os
from typing import Tuple
import tensorflow as tf


_IMAGENET_SIZE = 227
_NUM_PARALLEL_CALLS = 4


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

    return tf.image.resize(
        image, [_IMAGENET_SIZE, _IMAGENET_SIZE],
        method=tf.image.ResizeMethod.BILINEAR), label


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
    file_pattern = os.path.join(data_path, 'tfrecord_train-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    dataset = dataset.interleave(
        lambda name: tf.data.TFRecordDataset(name, buffer_size=buffer_size),
        cycle_length=16,
        num_parallel_calls=_NUM_PARALLEL_CALLS)

    dataset = dataset.shuffle(buffer_size=1024).repeat()
    dataset = dataset.map(lambda x: preprocess(*parse_record(x)),
                          num_parallel_calls=_NUM_PARALLEL_CALLS)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(8)
    return dataset

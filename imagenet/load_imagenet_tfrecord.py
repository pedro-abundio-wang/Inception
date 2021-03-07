import tensorflow as tf

# Create a dictionary describing the features.
image_feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/colorspace': tf.io.FixedLenFeature([], tf.string),
    'image/channels': tf.io.FixedLenFeature([], tf.int64),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    'image/class/synset': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


raw_image_dataset = tf.data.TFRecordDataset('/data/image-net/tfrecord_train/tfrecord_train-00731-of-01024')

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)



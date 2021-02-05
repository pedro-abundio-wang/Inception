import tensorflow as tf
from tensorflow import keras


class IdentityBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(IdentityBlock, self).__init(**kwargs)
        self.filters = filters

    def call(self, inputs):
        x = inputs
        x = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.Add()([x, inputs])
        x = keras.layers.ReLU()(x)
        return x

    def get_config(self):
        config = super(IdentityBlock, self).get_config()
        config.update({"filters": self.filters})
        return config


class DownsamplingIdentityBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(DownsamplingIdentityBlock, self).__init__(**kwargs)
        self.filters = filters

    def call(self, inputs):
        x = inputs
        x = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        inputs = keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(2, 2), padding='valid')(inputs)
        inputs = keras.layers.BatchNormalization(axis=-1)(inputs)
        x = keras.layers.Add()([x, inputs])
        x = keras.layers.ReLU()(x)
        return x

    def get_config(self):
        config = super(DownsamplingIdentityBlock, self).get_config()
        config.update({"filters": self.filters})
        return config


class BottleneckIdentityBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(BottleneckIdentityBlock, self).__init__(**kwargs)
        self.filters = filters

    def call(self, inputs):
        x = inputs
        x = keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=4 * self.filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.Add()([x, inputs])
        x = keras.layers.ReLU()(x)
        return x

    def get_config(self):
        config = super(BottleneckIdentityBlock, self).get_config()
        config.update({"filters": self.filters})
        return config


class DownsamplingBottleneckIdentityBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(DownsamplingBottleneckIdentityBlock, self).__init__(**kwargs)
        self.filters = filters

    def call(self, inputs):
        x = inputs
        x = keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(2, 2), padding='valid')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=4 * self.filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        inputs = keras.layers.Conv2D(filters=4 * self.filters, kernel_size=(1, 1), strides=(2, 2), padding='valid')(inputs)
        inputs = keras.layers.BatchNormalization(axis=-1)(inputs)
        x = keras.layers.Add()([x, inputs])
        x = keras.layers.ReLU()(x)
        return x

    def get_config(self):
        config = super(DownsamplingBottleneckIdentityBlock, self).get_config()
        config.update({"filters": self.filters})
        return config

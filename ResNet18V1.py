import tensorflow as tf
from tensorflow import keras
from ResNetV1Block import IdentityBlock
from ResNetV1Block import DownsamplingIdentityBlock

class ResNet18V1(keras.Model):
    def __init__(self, num_classes=1000):
        super(ResNet18V1, self).__init__()
        self.num_classes = num_classes

    def call(self, inputs):
        # stage1
        x = inputs
        x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # stage2
        x = IdentityBlock(filters=64)(x)
        x = IdentityBlock(filters=64)(x)
        # stage3
        x = DownsamplingIdentityBlock(filters=128)(x)
        x = IdentityBlock(filters=128)(x)
        # stage4
        x = DownsamplingIdentityBlock(filters=256)(x)
        x = IdentityBlock(filters=256)(x)
        # stage5
        x = DownsamplingIdentityBlock(filters=512)(x)
        x = IdentityBlock(filters=512)(x)
        # classifier
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(units=self.num_classes, activation='softmax', name='predictions')(x)
        return x


resnet = ResNet18V1()
resnet.summary()
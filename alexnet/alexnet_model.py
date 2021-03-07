import tensorflow as tf


def alexnet(input_shape=(227, 227, 3),
            classes=1000,
            classifier_activation='softmax'):

    img_input = tf.keras.layers.Input(shape=input_shape, name='input')

    # stage1
    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', name='stage1_conv11x11')(img_input)
    x = tf.keras.layers.ReLU(name='stage1_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage1_bn')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stage1_pool')(x)

    # stage2
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', name='stage2_conv5x5')(x)
    x = tf.keras.layers.ReLU(name='stage2_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage2_bn')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stage2_pool')(x)

    # stage3
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name='stage3_conv3x3')(x)
    x = tf.keras.layers.ReLU(name='stage3_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage3_bn')(x)

    # stage4
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name='stage4_conv3x3')(x)
    x = tf.keras.layers.ReLU(name='stage4_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage4_bn')(x)

    # stage5
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='stage5_conv3x3')(x)
    x = tf.keras.layers.ReLU(name='stage5_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage5_bn')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stage5_pool')(x)
    x = tf.keras.layers.Flatten(name='stage5_flatten')(x)

    # stage6
    x = tf.keras.layers.Dense(units=4096, name='stage6_fc')(x)
    x = tf.keras.layers.ReLU(name='stage6_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage6__bn')(x)
    x = tf.keras.layers.Dropout(rate=0.5, name='stage6_dropout')(x)

    # stage7
    x = tf.keras.layers.Dense(units=4096, name='stage7_fc')(x)
    x = tf.keras.layers.ReLU(name='stage7_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage7_bn')(x)
    x = tf.keras.layers.Dropout(rate=0.5, name='stage7_dropout')(x)

    # classifier
    x = tf.keras.layers.Dense(units=classes, activation=classifier_activation, name='predictions')(x)

    # Create model.
    inputs = img_input
    model = tf.keras.Model(inputs=inputs, outputs=x, name='AlexNet')

    return model

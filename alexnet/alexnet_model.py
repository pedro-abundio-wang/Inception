import tensorflow as tf


def alexnet(input_shape=(227, 227, 3),
            classes=1000,
            classifier_activation='softmax'):

    img_input = tf.keras.layers.Input(shape=input_shape, name='input')

    # stage1
    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005), name='stage1_conv11x11')(img_input)
    x = tf.keras.layers.ReLU(name='stage1_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage1_bn')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stage1_pool')(x)

    # stage2
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005), name='stage2_conv5x5')(x)
    x = tf.keras.layers.ReLU(name='stage2_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage2_bn')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stage2_pool')(x)

    # stage3
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005), name='stage3_conv3x3')(x)
    x = tf.keras.layers.ReLU(name='stage3_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage3_bn')(x)

    # stage4
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005), name='stage4_conv3x3')(x)
    x = tf.keras.layers.ReLU(name='stage4_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage4_bn')(x)

    # stage5
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005), name='stage5_conv3x3')(x)
    x = tf.keras.layers.ReLU(name='stage5_relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='stage5_bn')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stage5_pool')(x)
    x = tf.keras.layers.Flatten(name='stage5_flatten')(x)

    # fc
    x = tf.keras.layers.Dense(units=4096, kernel_regularizer=tf.keras.regularizers.l2(0.0005), name='fc')(x)
    x = tf.keras.layers.ReLU(name='relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='bn')(x)
    x = tf.keras.layers.Dropout(rate=0.5, name='dropout')(x)

    x = tf.keras.layers.Dense(units=4096, kernel_regularizer=tf.keras.regularizers.l2(0.0005), name='fc_')(x)
    x = tf.keras.layers.ReLU(name='relu_')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name='bn_')(x)
    x = tf.keras.layers.Dropout(rate=0.5, name='dropout_')(x)

    # classifier
    x = tf.keras.layers.Dense(units=classes, activation=classifier_activation,
                              kernel_regularizer=tf.keras.regularizers.l2(0.0005), name='predictions')(x)

    # Create model.
    inputs = img_input
    model = tf.keras.Model(inputs=inputs, outputs=x, name='AlexNet')

    return model

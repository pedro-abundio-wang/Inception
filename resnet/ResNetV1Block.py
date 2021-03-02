import tensorflow as tf
from tensorflow import keras


def identity_block(x, filters, stage, block):
    
    base_name = stage + block
    
    # shortcut connection
    x_shortcut = x
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=base_name+'_conv1')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn1')(x)
    x = keras.layers.ReLU(name=base_name+'_relu1')(x)
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=base_name+'_conv2')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn2')(x)
    
    x = keras.layers.Add(name=base_name+'_add')([x, x_shortcut])
    x = keras.layers.ReLU(name=base_name+'_relu2')(x)
    
    return x


def identity_block_downsampling(x, filters, stage, block):
    
    base_name = stage + block
    
    # shortcut connection
    x_shortcut = x
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same', name=base_name+'_conv1')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn1')(x)
    x = keras.layers.ReLU(name=base_name+'_relu1')(x)
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=base_name+'_conv2')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn2')(x)
    
    x_shortcut = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding='valid', name=base_name+'_shortcut_conv')(x_shortcut)
    x_shortcut = keras.layers.BatchNormalization(axis=-1, name=base_name+'_shortcut_bn')(x_shortcut)
    
    x = keras.layers.Add(name=base_name+'_add')([x, x_shortcut])
    x = keras.layers.ReLU(name=base_name+'_relu2')(x)    
    
    return x


def bottleneck_identity_block(x, filters, stage, block):
    
    base_name = stage + block
    
    # shortcut connection
    x_shortcut = x
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=base_name+'_conv1')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn1')(x)
    x = keras.layers.ReLU(name=base_name+'_relu1')(x)
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=base_name+'_conv2')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn2')(x)
    x = keras.layers.ReLU(name=base_name+'_relu2')(x)
    
    x = keras.layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=base_name+'_conv3')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn3')(x)
    
    x = keras.layers.Add(name=base_name+'_add')([x, x_shortcut])
    x = keras.layers.ReLU(name=base_name+'_relu3')(x)
    
    return x


def bottleneck_identity_block_conv(x, filters, stage, block):
    
    base_name = stage + block
    
    # shortcut connection
    x_shortcut = x
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=base_name+'_conv1')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn1')(x)
    x = keras.layers.ReLU(name=base_name+'_relu1')(x)
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=base_name+'_conv2')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn2')(x)
    x = keras.layers.ReLU(name=base_name+'_relu2')(x)
    
    x = keras.layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=base_name+'_conv3')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn3')(x)
    
    x_shortcut = keras.layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=base_name+'_shortcut_conv')(x_shortcut)
    x_shortcut = keras.layers.BatchNormalization(axis=-1, name=base_name+'_shortcut_bn')(x_shortcut)
    
    x = keras.layers.Add(name=base_name+'_add')([x, x_shortcut])
    x = keras.layers.ReLU(name=base_name+'_relu3')(x)
    
    return x


def bottleneck_identity_block_downsampling(x, filters, stage, block):

    base_name = stage + block
    
    # shortcut connection
    x_shortcut = x
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding='valid', name=base_name+'_conv1')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn1')(x)
    x = keras.layers.ReLU(name=base_name+'_relu1')(x)
    
    x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=base_name+'_conv2')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn2')(x)
    x = keras.layers.ReLU(name=base_name+'_relu2')(x)
    
    x = keras.layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=base_name+'_conv3')(x)
    x = keras.layers.BatchNormalization(axis=-1, name=base_name+'_bn3')(x)

    x_shortcut = keras.layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(2, 2), padding='valid', name=base_name+'_shortcut_conv')(x_shortcut)
    x_shortcut = keras.layers.BatchNormalization(axis=-1, name=base_name+'_shortcut_bn')(x_shortcut)
    
    x = keras.layers.Add(name=base_name+'_add')([x, x_shortcut])
    x = keras.layers.ReLU(name=base_name+'_relu3')(x)
    
    return x
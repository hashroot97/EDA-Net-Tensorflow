import tensorflow as tf
from keras.utils import to_categorical
import os
import cv2
import numpy as np

def downsample_block_win_smaller_wout(input_tensor, block_number, in_channels, out_channels):
    
    conv_filters = out_channels - in_channels
    
    with tf.variable_scope('downsample_block_' + str(block_number), reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, in_channels, conv_filters],
                                  dtype=tf.float32)
        biases = tf.get_variable('biases',
                                 shape=[conv_filters],
                                 dtype=tf.float32)
        pre_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.bias_add(pre_conv, biases)
        
        max_pool= tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')
        
        concat = tf.concat([conv, max_pool], axis=-1)
        
        with tf.variable_scope('bn_relu', reuse=tf.AUTO_REUSE) as scope:
            bn = tf.layers.batch_normalization(concat)
            relu = tf.nn.relu(bn)
            out = relu
        
    return out

def downsample_block_win_greater_wout(input_tensor, block_number, in_channels, out_channels):
    
    conv_filters = out_channels
    
    with tf.variable_scope('downsample_block_'+ str(block_number), reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, in_channels, conv_filters],
                                  dtype=tf.float32)
        biases = tf.get_variable('biases',
                                 shape=[conv_filters],
                                 dtype=tf.float32)
        pre_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.bias_add(pre_conv, biases)
        
        with tf.variable_scope('bn_relu', reuse=tf.AUTO_REUSE) as scope:
            bn = tf.layers.batch_normalization(conv)
            relu = tf.nn.relu(bn)
            out = relu
    return out

def asym_block(input_tensor, inchannels, outchannels, dilation=1):
    
    if dilation>1:
        dilation_name = '_d'
    else:
        dilation_name = ''
    
    with tf.variable_scope('asym_conv' + dilation_name, reuse=tf.AUTO_REUSE) as scope:

        with tf.variable_scope('asym_3x1', reuse=tf.AUTO_REUSE) as scope:

            weights = tf.get_variable('weights',
                                      shape=[3, 1, inchannels, outchannels],
                                      dtype=tf.float32)
            biases = tf.get_variable('biases',
                                     shape=[outchannels],
                                     dtype=tf.float32)
            pre_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], dilations=[1, dilation, dilation, 1], padding='SAME')
            conv_asym_3x1 = tf.nn.bias_add(pre_conv, biases)

        with tf.variable_scope('asym_1x3', reuse=tf.AUTO_REUSE) as scope:

            weights = tf.get_variable('weights',
                                      shape=[1, 3, inchannels, outchannels],
                                      dtype=tf.float32)
            biases = tf.get_variable('biases',
                                     shape=[outchannels],
                                     dtype=tf.float32)
            pre_conv = tf.nn.conv2d(conv_asym_3x1, weights, strides=[1, 1, 1, 1], dilations=[1, dilation, dilation, 1], padding='SAME')
            conv_asym_1x3 = tf.nn.bias_add(pre_conv, biases)
            
        with tf.variable_scope('bn_relu', reuse=tf.AUTO_REUSE) as scope:
            
            bn = tf.layers.batch_normalization(conv_asym_1x3)
            relu = tf.nn.relu(bn)
        out = relu
    return out

def pointwise_conv(input_tensor, inchannels, outchannels):
    with tf.variable_scope('1x1_conv', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                  shape=[1, 1, inchannels, outchannels],
                                  dtype=tf.float32)
        biases = tf.get_variable('biases',
                                 shape=[outchannels],
                                 dtype=tf.float32)
        pre_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv_1x1 = tf.nn.bias_add(pre_conv, biases)
        bn_1x1 = tf.layers.batch_normalization(pre_conv)
        relu_1x1 = tf.nn.relu(bn_1x1)
        out_1x1 = relu_1x1
    return out_1x1

def eda_block(input_tensor, module_number, block_number, inchannels, growth_rate, dilation_rate):
    
    with tf.variable_scope('EDAblock_' + str(module_number) + '_' +  str(block_number), reuse=tf.AUTO_REUSE) as scope:

        conv_1x1 = pointwise_conv(input_tensor, inchannels, growth_rate)

        asym_1 = asym_block(conv_1x1, growth_rate, growth_rate)

        asym_2_d = asym_block(asym_1, growth_rate, growth_rate, dilation=dilation_rate)

        keep_prob = tf.constant([0.98], dtype=tf.float32,shape=())
        dropout = tf.nn.dropout(asym_2_d, keep_prob=keep_prob)
        
        concat = tf.concat([input_tensor, dropout], axis=-1)
#         print(concat)
    return concat

def get_eda_module_1(input_tensor, growth_rate, inchannels):
    with tf.variable_scope('EDAmodule_1', reuse=tf.AUTO_REUSE) as scope:
        eda_block_1_1 = eda_block(input_tensor, 1, 1, inchannels, growth_rate, 1)
        eda_block_1_2 = eda_block(eda_block_1_1, 1, 2, 100, growth_rate, 1)
        eda_block_1_3 = eda_block(eda_block_1_2, 1, 3, 140, growth_rate, 1)
        eda_block_1_4 = eda_block(eda_block_1_3, 1, 4, 180, growth_rate, 2)
        eda_block_1_4 = eda_block(eda_block_1_4, 1, 5, 220, growth_rate, 2)
    return eda_block_1_4

def get_eda_module_2(input_tensor, growth_rate, inchannels):
    with tf.variable_scope('EDAmodule_2', reuse=tf.AUTO_REUSE) as scope:
        eda_block_2_1 = eda_block(input_tensor, 2, 1, inchannels, growth_rate, 2)
        eda_block_2_2 = eda_block(eda_block_2_1, 2, 2, 170, growth_rate, 2)
        eda_block_2_3 = eda_block(eda_block_2_2, 2, 3, 210, growth_rate, 4)
        eda_block_2_4 = eda_block(eda_block_2_3, 2, 4, 250, growth_rate, 4)
        eda_block_2_5 = eda_block(eda_block_2_4, 2, 5, 290, growth_rate, 8)
        eda_block_2_6 = eda_block(eda_block_2_5, 2, 6, 330, growth_rate, 8)
        eda_block_2_7 = eda_block(eda_block_2_6, 2, 7, 370, growth_rate, 16)
        eda_block_2_8 = eda_block(eda_block_2_7, 2, 8, 410, growth_rate, 16)
    return eda_block_2_8

def projection_layer(input_tensor, inchannels, outchannels):
    
    with tf.variable_scope('Projection_Layer', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                  shape=[1, 1, inchannels, outchannels],
                                  dtype=tf.float32)
        biases = tf.get_variable('biases',
                                 shape=[outchannels],
                                 dtype=tf.float32)
        pre_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
        projection_1x1 = tf.nn.bias_add(pre_conv, biases)
    return projection_1x1

def upsample_image(input_tensor):
    with tf.variable_scope('Upsampling_8x') as scope:
        resized_image = tf.image.resize_images(input_tensor, (512, 1024))
    return resized_image

def build_model(input_image, num_classes):
    
    with tf.variable_scope('Preprocess', reuse=tf.AUTO_REUSE) as scope:
        input_resized_image = tf.image.resize_images(input_image, (512, 1024))
        print('Input Image : ', input_resized_image)
    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE) as scope:
        downsample1 = downsample_block_win_smaller_wout(input_resized_image, 1, 3, 15)
        downsample2 = downsample_block_win_smaller_wout(downsample1, 2, 15, 60)
        eda_module_1 = get_eda_module_1(downsample2, 40, 60)
        downsample3 = downsample_block_win_greater_wout(eda_module_1, 3, 260, 130)
        eda_module_2 = get_eda_module_2(downsample3, 40, 130)
        projection = projection_layer(eda_module_2, 450, num_classes)
    # with tf.variable_scope('Postprocess', reuse=tf.AUTO_REUSE) as scope:
        # resized_image = upsample_image(projection)
        # resized_image = tf.argmax(resized_image, axis=-1)
    return projection, projection

def get_loss(logits, labels):
    with tf.variable_scope('Loss', reuse=tf.AUTO_REUSE) as scope:
        resized_labels = tf.image.resize_images(labels, (64, 128))
        resized_labels = tf.dtypes.cast(resized_labels, dtype=tf.int32)
        encoded_labels = tf.one_hot(resized_labels, axis=-1, depth=40)
        encoded_labels = tf.reshape(encoded_labels, shape=[-1, 64, 128, 40])
        print('Labels : ', encoded_labels)
        print('Logits : ', logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=encoded_labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
    return loss

def get_train_op(loss, learning_rate):
    with tf.variable_scope('Optimizer', reuse=tf.AUTO_REUSE) as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        train_op = optimizer.minimize(loss)
    return train_op



# Taken from: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py

import numpy as np, tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19:
    def __init__(self, vgg16_npy_path):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

    def __call__(self, rgb, reuse=False):
        with tf.variable_scope('vgg', reuse=reuse):
            rgb_scaled = rgb * 255.0

            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])

            conv1_1 = self.conv_layer(bgr, "conv1_1")
            conv1_2 = self.conv_layer(conv1_1, "conv1_2")
            pool1 = self.max_pool(conv1_2, 'pool1')

            conv2_1 = self.conv_layer(pool1, "conv2_1")
            conv2_2 = self.conv_layer(conv2_1, "conv2_2")
            pool2 = self.max_pool(conv2_2, 'pool2')

            conv3_1 = self.conv_layer(pool2, "conv3_1")
            conv3_2 = self.conv_layer(conv3_1, "conv3_2")
            conv3_3 = self.conv_layer(conv3_2, "conv3_3")
            conv3_4 = self.conv_layer(conv3_3, "conv3_4")
            pool3 = self.max_pool(conv3_4, 'pool3')

            conv4_1 = self.conv_layer(pool3, "conv4_1")
            conv4_2 = self.conv_layer(conv4_1, "conv4_2")
            conv4_3 = self.conv_layer(conv4_2, "conv4_3")
            conv4_4 = self.conv_layer(conv4_3, "conv4_4")
            pool4 = self.max_pool(conv4_4, 'pool4')

            conv5_1 = self.conv_layer(pool4, "conv5_1")
            conv5_2 = self.conv_layer(conv5_1, "conv5_2")
            conv5_3 = self.conv_layer(conv5_2, "conv5_3")
            conv5_4 = self.conv_layer(conv5_3, "conv5_4")
            pool5 = self.max_pool(conv5_4, 'pool5')

            return pool1, pool2, pool3, pool4, pool5

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        filter = self.data_dict[name][0]
        return tf.get_variable(name='filter', shape=filter.shape,
                               initializer=tf.constant_initializer(filter),
                               trainable=False)

    def get_bias(self, name):
        bias = self.data_dict[name][1]
        return tf.get_variable(name='bias', shape=bias.shape,
                               initializer=tf.constant_initializer(bias),
                               trainable=False)
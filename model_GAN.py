import numpy as np
# import os
import tensorflow as tf
# from PIL import Image
# import utility as Utility
# import argparse
import math

class SPIGAN():
    def __init__(self, img_channel, anno_channel, seed, base_channel, keep_prob, path_to_vgg19, base_fc_node=1024,
                 base_channel_predictor=16):
        # self.NOISE_UNIT_NUM = noise_unit_num  # 200
        self.IMG_CHANNEL = img_channel  # 3
        self.SEG_CLASS = 19
        self.DEPTH_CLASS = 1
        self.ANNO_CHANNEL = anno_channel # 4
        self.SEED = seed
        np.random.seed(seed=self.SEED)
        self.BASE_CHANNEL = base_channel  # 32
        self.BASE_CHANNEL_PRE = base_channel_predictor
        self.KEEP_PROB = keep_prob
        self.BASE_FC_NODE = base_fc_node
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.vgg19_npy = np.load(path_to_vgg19, encoding='latin1').item()
        self.vgg_trainable = False
        self.var_dict = {}

    def leaky_relu(self, x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def gaussian_noise(self, input, std):  # used at discriminator
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32, seed=self.SEED)
        return input + noise

    def conv2d(self, input, in_channel, out_channel, k_size, stride, seed):
        w = tf.get_variable('w', [k_size, k_size, in_channel, out_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding="SAME", name='conv') + b
        return conv

    def max_pool(self, input, k_size, stride):
        pool = tf.nn.max_pool(input, [1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                              padding="SAME", name='pool')
        return pool

    def conv2d_transpose(self, input, in_channel, out_channel, k_size, stride, seed):
        # print("input, ", input.get_shape().as_list())
        w = tf.get_variable('w', [k_size, k_size, out_channel, in_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        out_shape = tf.stack(
            [tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, tf.constant(out_channel)])
        # print("input[0], ", input[0].get_shape().as_list())
        # print("out_shape, ", out_shape.get_shape().as_list())
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=out_shape, strides=[1, stride, stride, 1],
                                        padding="SAME") + b
        # print("deconv, in def, ", deconv.get_shape().as_list())
        return deconv

    # def batch_norm(self, input):
    #     shape = input.get_shape().as_list()
    #     n_out = shape[-1]
    #     scale = tf.get_variable('scale', [n_out], initializer=tf.constant_initializer(1.0))
    #     beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0))
    #     batch_mean, batch_var = tf.nn.moments(input, [0])
    #     bn = tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.0001, name='batch_norm')
    #     return bn

    def batch_norm_train(self, inputs, pop_mean, pop_var, beta, scale, decay=0.999):
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 1e-4)


    def batch_norm_wrapper(self, inputs, decay=0.999, is_training=True):
        # beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), dtype=tf.float32, name='beta')
        beta = tf.get_variable('beta', [inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        # scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), dtype=tf.float32, name='gamma')
        scale = tf.get_variable('gamma', [inputs.get_shape()[-1]], initializer=tf.constant_initializer(1.0), dtype=tf.float32)
        # pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False, name='moving_mean')
        pop_mean = tf.get_variable('moving_mean', [inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0),
                                  trainable=False , dtype=tf.float32)
        # pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False, name='moving_variance')
        pop_var = tf.get_variable('moving_variance', [inputs.get_shape()[-1]], initializer=tf.constant_initializer(1.0),
                                  trainable=False, dtype=tf.float32)
        result = tf.cond(is_training, lambda:self.batch_norm_train(inputs, pop_mean, pop_var, beta, scale, decay),
                         lambda:tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 1e-4))
        return result


    def fully_connect(self, input, in_num, out_num, seed):
        w = tf.get_variable('w', [in_num, out_num], initializer=tf.random_normal_initializer
        (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_num], initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input, w, name='fc') + b
        return fc

    def residual_block(self, input, channel, kernel, stride, seed):
        conv_b1 = self.conv2d(input, channel, channel, kernel, stride, seed)
        re_b1 = tf.nn.relu(conv_b1)
        conv_b2 = self.conv2d(re_b1, channel, channel, kernel, stride, seed)
        add_b2 = conv_b2 + input
        re_b2 = tf.nn.relu(add_b2)
        return re_b2

    def generator(self, xs, reuse=False, is_training=True):  # x is expected [n, 128, 128, 1]
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer1 c7s1-32 nx128x128x3 -> conv nx64x64x32
                conv1 = self.conv2d(xs, self.IMG_CHANNEL, self.BASE_CHANNEL, 7, 2, self.SEED)
                bn1 = self.batch_norm_wrapper(conv1, is_training=is_training)
                re1 = tf.nn.relu(bn1)

            with tf.variable_scope("layer2"):  # layer2 conv nx64x64x64 -> nx32x32x128
                conv2 = self.conv2d(re1, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, 3, 2, self.SEED)
                bn2 = self.batch_norm_wrapper(conv2, is_training=is_training)
                re2 = tf.nn.relu(bn2)

            with tf.variable_scope("layer3"):  # layer3
                re3 = self.residual_block(re2, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer4"):  # layer4
                re4 = self.residual_block(re3, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer5"):  # layer5
                re5 = self.residual_block(re4, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer6"):  # layer6
                re6 = self.residual_block(re5, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer7"):  # layer7
                re7 = self.residual_block(re6, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer8"):  # layer8
                re8 = self.residual_block(re7, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer9"):  # layer9
                re9 = self.residual_block(re8, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer10"):  # layer10
                re10 = self.residual_block(re9, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer11"):  # layer11
                re11 = self.residual_block(re10, self.BASE_CHANNEL * 2, 3, 1, self.SEED)

            with tf.variable_scope("layer12"):  # layer6 deconv nx4*4*4c -> nx4x4x64 -> nx8x8x64
                deconv12 = self.conv2d_transpose(re11, self.BASE_CHANNEL * 2, self.BASE_CHANNEL, 3, 2, self.SEED)
                bn12 = self.batch_norm_wrapper(deconv12, is_training=is_training)
                re12 = tf.nn.relu(bn12)

            with tf.variable_scope("layer13"):  # layer6 deconv nx4*4*4c -> nx4x4x64 -> nx8x8x64
                deconv13 = self.conv2d_transpose(re12, self.BASE_CHANNEL * 2, self.IMG_CHANNEL, 7, 2, self.SEED)
                # bn13 = self.batch_norm_wrapper(deconv13, is_training=is_training)
                # rl12 = tf.nn.relu(bn12)
        return deconv13


    def discriminator(self, x, reuse=False, is_training=True, keep_prob=1.0):  # x[n, 128, 128, 3]
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer 1 conv [n, 128, 128, 3] -> [n, 64, 64, c]
                conv1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL*2, 4, 2, self.SEED)
                bn1 = self.batch_norm_wrapper(conv1, is_training=is_training)
                lr1 = self.leaky_relu(bn1, alpha=0.2)

            with tf.variable_scope("layer2"):  # layer 2 conv [n, 64, 64, c] -> [n, 32, 32, 2c]
                conv2 = self.conv2d(lr1, self.BASE_CHANNEL*2, self.BASE_CHANNEL*4, 4, 2, self.SEED)
                bn2 = self.batch_norm_wrapper(conv2, is_training=is_training)
                lr2 = self.leaky_relu(bn2, alpha=0.2)

            with tf.variable_scope("layer3"):  # layer 3
                conv3 = self.conv2d(lr2, self.BASE_CHANNEL*4, 1, 4, 2, self.SEED)
                self.sigmoid = tf.nn.sigmoid(conv3)

        return self.sigmoid


    def task_predictor(self, x, reuse=False, is_training=True, keep_prob=1.0):  # x[n, 128, 128, 3]
        with tf.variable_scope('task_predictor', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer 1
                conv1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL_PRE, 3, 1, self.SEED)
                re1 = tf.nn.relu(conv1)
                conv2 = self.conv2d(re1, self.BASE_CHANNEL_PRE, self.BASE_CHANNEL_PRE, 3, 1, self.SEED)
                re2 = tf.nn.relu(conv2)
                pool2 = self.max_pool(re2, 2, 2)

            with tf.variable_scope("layer2"):  # layer 2
                conv3 = self.conv2d(pool2, self.BASE_CHANNEL_PRE, self.BASE_CHANNEL_PRE*2, 3, 1, self.SEED)
                re3 = tf.nn.relu(conv3)
                conv4 = self.conv2d(re3, self.BASE_CHANNEL_PRE*2, self.BASE_CHANNEL_PRE*2, 3, 1, self.SEED)
                re4 = tf.nn.relu(conv4)
                pool4 = self.max_pool(re4, 2, 2)

            with tf.variable_scope("layer3"):  # layer 3
                conv5 = self.conv2d(pool4, self.BASE_CHANNEL_PRE*2, self.BASE_CHANNEL_PRE*4, 3, 1, self.SEED)
                re5 = tf.nn.relu(conv5)
                conv6 = self.conv2d(re5, self.BASE_CHANNEL_PRE*4, self.BASE_CHANNEL_PRE*4, 3, 1, self.SEED)
                re6 = tf.nn.relu(conv6)
                conv7 = self.conv2d(re6, self.BASE_CHANNEL_PRE*4, self.BASE_CHANNEL_PRE*4, 3, 1, self.SEED)
                re7 = tf.nn.relu(conv7)
                pool7 = self.max_pool(re7, 2, 2)

            with tf.variable_scope("layer4"):  # layer 4
                conv8 = self.conv2d(pool7, self.BASE_CHANNEL_PRE*4, self.BASE_CHANNEL_PRE*8, 3, 1, self.SEED)
                re8 = tf.nn.relu(conv8)
                conv9 = self.conv2d(re8, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 3, 1, self.SEED)
                re9 = tf.nn.relu(conv9)
                conv10 = self.conv2d(re9, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 3, 1, self.SEED)
                re10 = tf.nn.relu(conv10)
                pool10 = self.max_pool(re10, 2, 2)

            with tf.variable_scope("layer5"):  # layer 5
                conv11 = self.conv2d(pool10, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 3, 1, self.SEED)
                re11 = tf.nn.relu(conv11)
                conv12 = self.conv2d(re11, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 3, 1, self.SEED)
                re12 = tf.nn.relu(conv12)
                conv13 = self.conv2d(re12, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 3, 1, self.SEED)
                re13 = tf.nn.relu(conv13)
                pool13 = self.max_pool(re13, 2, 2)

            with tf.variable_scope("layer6"):  # layer 6
                conv14 = self.conv2d(pool13, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 1, 1, self.SEED)
                re14 = tf.nn.relu(conv14)
                drop14 = tf.nn.dropout(re14, keep_prob=keep_prob)
                conv15 = self.conv2d(drop14, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 1, 1, self.SEED)
                re15 = tf.nn.relu(conv15)
                drop15 = tf.nn.dropout(re15, keep_prob=keep_prob)

            with tf.variable_scope("layer7"):  # layer 7
                conv16 = self.conv2d(drop15, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 1, 1, self.SEED)
                re16 = tf.nn.relu(conv16)
                deconv1 = self.conv2d_transpose(re16, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 3, 2, self.SEED)
                re_d1 = tf.nn.relu(deconv1)
                add1 = pool10  + re_d1

            with tf.variable_scope("layer8"):  # layer 8
                conv_d1 = self.conv2d(add1, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*8, 3, 3, self.SEED)
                re_d1 = tf.nn.relu(conv_d1)
                deconv2 = self.conv2d_transpose(re_d1, self.BASE_CHANNEL_PRE*8, self.BASE_CHANNEL_PRE*4, 3, 2, self.SEED)
                re_d2 = tf.nn.relu(deconv2)
                add2 = pool7  + re_d2

            with tf.variable_scope("layer9"):  # layer 9
                conv_d2 = self.conv2d(add2, self.BASE_CHANNEL_PRE*4, self.BASE_CHANNEL_PRE*4, 4, 4, self.SEED)
                re_d2 = tf.nn.relu(conv_d2)
                deconv3 = self.conv2d_transpose(re_d2, self.BASE_CHANNEL_PRE*4, self.BASE_CHANNEL_PRE, 8, 8, self.SEED)
                re_d3 = tf.nn.relu(deconv3)
                conv_d3 = self.conv2d(re_d3, self.BASE_CHANNEL_PRE, self.SEG_CLASS, 4, 4, self.SEED)
                softmax_d3 = tf.nn.softmax(conv_d3)

        return softmax_d3

    def privileged_network(self, x, reuse=False, is_training=True, keep_prob=1.0):  # x[n, 128, 128, 3]
        with tf.variable_scope('task_predictor', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer 1
                conv1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL_PRE, 3, 1, self.SEED)
                re1 = tf.nn.relu(conv1)
                conv2 = self.conv2d(re1, self.BASE_CHANNEL_PRE, self.BASE_CHANNEL_PRE, 3, 1, self.SEED)
                re2 = tf.nn.relu(conv2)
                pool2 = self.max_pool(re2, 2, 2)

            with tf.variable_scope("layer2"):  # layer 2
                conv3 = self.conv2d(pool2, self.BASE_CHANNEL_PRE, self.BASE_CHANNEL_PRE * 2, 3, 1, self.SEED)
                re3 = tf.nn.relu(conv3)
                conv4 = self.conv2d(re3, self.BASE_CHANNEL_PRE * 2, self.BASE_CHANNEL_PRE * 2, 3, 1, self.SEED)
                re4 = tf.nn.relu(conv4)
                pool4 = self.max_pool(re4, 2, 2)

            with tf.variable_scope("layer3"):  # layer 3
                conv5 = self.conv2d(pool4, self.BASE_CHANNEL_PRE * 2, self.BASE_CHANNEL_PRE * 4, 3, 1, self.SEED)
                re5 = tf.nn.relu(conv5)
                conv6 = self.conv2d(re5, self.BASE_CHANNEL_PRE * 4, self.BASE_CHANNEL_PRE * 4, 3, 1, self.SEED)
                re6 = tf.nn.relu(conv6)
                conv7 = self.conv2d(re6, self.BASE_CHANNEL_PRE * 4, self.BASE_CHANNEL_PRE * 4, 3, 1, self.SEED)
                re7 = tf.nn.relu(conv7)
                pool7 = self.max_pool(re7, 2, 2)

            with tf.variable_scope("layer4"):  # layer 4
                conv8 = self.conv2d(pool7, self.BASE_CHANNEL_PRE * 4, self.BASE_CHANNEL_PRE * 8, 3, 1, self.SEED)
                re8 = tf.nn.relu(conv8)
                conv9 = self.conv2d(re8, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 3, 1, self.SEED)
                re9 = tf.nn.relu(conv9)
                conv10 = self.conv2d(re9, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 3, 1, self.SEED)
                re10 = tf.nn.relu(conv10)
                pool10 = self.max_pool(re10, 2, 2)

            with tf.variable_scope("layer5"):  # layer 5
                conv11 = self.conv2d(pool10, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 3, 1, self.SEED)
                re11 = tf.nn.relu(conv11)
                conv12 = self.conv2d(re11, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 3, 1, self.SEED)
                re12 = tf.nn.relu(conv12)
                conv13 = self.conv2d(re12, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 3, 1, self.SEED)
                re13 = tf.nn.relu(conv13)
                pool13 = self.max_pool(re13, 2, 2)

            with tf.variable_scope("layer6"):  # layer 6
                conv14 = self.conv2d(pool13, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 1, 1, self.SEED)
                re14 = tf.nn.relu(conv14)
                drop14 = tf.nn.dropout(re14, keep_prob=keep_prob)
                conv15 = self.conv2d(drop14, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 1, 1, self.SEED)
                re15 = tf.nn.relu(conv15)
                drop15 = tf.nn.dropout(re15, keep_prob=keep_prob)

            with tf.variable_scope("layer7"):  # layer 7
                conv16 = self.conv2d(drop15, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 1, 1, self.SEED)
                re16 = tf.nn.relu(conv16)
                deconv1 = self.conv2d_transpose(re16, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 3, 2,
                                                self.SEED)
                re_d1 = tf.nn.relu(deconv1)
                add1 = pool10 + re_d1

            with tf.variable_scope("layer8"):  # layer 8
                conv_d1 = self.conv2d(add1, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 8, 3, 3, self.SEED)
                re_d1 = tf.nn.relu(conv_d1)
                deconv2 = self.conv2d_transpose(re_d1, self.BASE_CHANNEL_PRE * 8, self.BASE_CHANNEL_PRE * 4, 3, 2,
                                                self.SEED)
                re_d2 = tf.nn.relu(deconv2)
                add2 = pool7 + re_d2

            with tf.variable_scope("layer9"):  # layer 9
                conv_d2 = self.conv2d(add2, self.BASE_CHANNEL_PRE * 4, self.BASE_CHANNEL_PRE * 4, 4, 4, self.SEED)
                re_d2 = tf.nn.relu(conv_d2)
                deconv3 = self.conv2d_transpose(re_d2, self.BASE_CHANNEL_PRE * 4, self.BASE_CHANNEL_PRE, 8, 8,
                                                self.SEED)
                re_d3 = tf.nn.relu(deconv3)
                conv_d3 = self.conv2d(re_d3, self.BASE_CHANNEL_PRE, self.DEPTH_CLASS, 4, 4, self.SEED)

        return conv_d3

    def VGG19(self, rgb):
        """
                load variable from npy to build the VGG
                :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
                :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
                """
        rgb_scaled = rgb * 255
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        with tf.variable_scope("encoder"):
            with tf.variable_scope("encoder_1"):
                self.conv1_1 = self.conv_layer_VGG(bgr, 3, 64, "conv1_1")
                self.conv1_2 = self.conv_layer_VGG(self.conv1_1, 64, 64, "conv1_2")
                self.pool1 = self.max_pool_VGG(self.conv1_2, 'pool1')

            with tf.variable_scope("encoder_2"):
                self.conv2_1 = self.conv_layer_VGG(self.pool1, 64, 128, "conv2_1")
                self.conv2_2 = self.conv_layer_VGG(self.conv2_1, 128, 128, "conv2_2")
                self.pool2 = self.max_pool_VGG(self.conv2_2, 'pool2')
            with tf.variable_scope("encoder_3"):
                self.conv3_1 = self.conv_layer_VGG(self.pool2, 128, 256, "conv3_1")
                self.conv3_2 = self.conv_layer_VGG(self.conv3_1, 256, 256, "conv3_2")
                self.conv3_3 = self.conv_layer_VGG(self.conv3_2, 256, 256, "conv3_3")
                self.conv3_4 = self.conv_layer_VGG(self.conv3_3, 256, 256, "conv3_4")
                self.pool3 = self.max_pool_VGG(self.conv3_4, 'pool3')
            with tf.variable_scope("encoder_4"):
                self.conv4_1 = self.conv_layer_VGG(self.pool3, 256, 512, "conv4_1")
                self.conv4_2 = self.conv_layer_VGG(self.conv4_1, 512, 512, "conv4_2")
                self.conv4_3 = self.conv_layer_VGG(self.conv4_2, 512, 512, "conv4_3")
                self.conv4_4 = self.conv_layer_VGG(self.conv4_3, 512, 512, "conv4_4")
                self.pool4 = self.max_pool_VGG(self.conv4_4, 'pool4')
            with tf.variable_scope("encoder_5"):
                self.conv5_1 = self.conv_layer_VGG(self.pool4, 512, 512, "conv5_1")
                self.conv5_2 = self.conv_layer_VGG(self.conv5_1, 512, 512, "conv5_2")

        return self.conv1_2, self.conv2_2, self.conv3_2, self.conv4_2, self.conv5_2
    
    def max_pool_VGG(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    def conv_layer_VGG(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def cal_input_num(self, input_num):
        stddev = math.sqrt(2 / (input_num))
        return stddev

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        stdd = self.cal_input_num(filter_size * filter_size * in_channels)
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], mean=0.0, stddev=stdd)
        filters = self.get_var(initial_value, name, 0, name + "_filters")
        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")
        return filters, biases


    def get_var(self, initial_value, name, idx, var_name):
        if self.vgg19_npy is not None and name in self.vgg19_npy:
            value = self.vgg19_npy[name][idx]
            print("vgg19 ", name, " is restored")
        else:
            value = initial_value
            print("vgg19 ", name, " is not exist")
        if self.vgg_trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
        self.var_dict[(name, idx)] = var
        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()
        return var


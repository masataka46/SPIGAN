import numpy as np
# import os
import tensorflow as tf
# from PIL import Image
# import utility as Utility
# import argparse
import math

class SPIGAN1():
    def __init__(self, noise_unit_num, img_channel, seed, base_channel, keep_prob):
        self.NOISE_UNIT_NUM = noise_unit_num  # 200
        self.IMG_CHANNEL = img_channel  # 3
        self.SEED = seed
        np.random.seed(seed=self.SEED)
        self.BASE_CHANNEL = base_channel  # 16
        self.KEEP_PROB = keep_prob

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

    def batch_norm(self, input):
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        scale = tf.get_variable('scale', [n_out], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0))
        batch_mean, batch_var = tf.nn.moments(input, [0])
        bn = tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.0001, name='batch_norm')
        return bn

    def instance_norm(self, input):
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        scale = tf.get_variable('scale', [n_out], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0))
        batch_mean, batch_var = tf.nn.moments(input, [0])
        bn = tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.0001, name='batch_norm')
        return bn


    def fully_connect(self, input, in_num, out_num, seed):
        w = tf.get_variable('w', [in_num, out_num], initializer=tf.random_normal_initializer
        (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_num], initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input, w, name='fc') + b
        return fc

    def encoder(self, x, reuse=False, is_training=True):  # x is expected [n, 128, 128, 1]
        with tf.variable_scope('encoder', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer1 conv nx128x128x1 -> nx64x64xc
                conv1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL, 7, 2, self.SEED)
                bn1 = tf.layers.batch_normalization(conv1, training=is_training)
                lr1 = self.leaky_relu(bn1, alpha=0.1)

            with tf.variable_scope("layer2"):  # layer2 conv nx64x64xc -> nx32x32x2c
                conv2 = self.conv2d(lr1, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, 4, 2, self.SEED)
                # bn2 = self.batch_norm(conv2)
                bn2 = tf.layers.batch_normalization(conv2, training=is_training)
                lr2 = self.leaky_relu(bn2, alpha=0.1)

            with tf.variable_scope("layer3"):  # layer3 conv nx32x32x2c -> nx16x16x4c
                conv3 = self.conv2d(lr2, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, 4, 2, self.SEED)
                # bn3 = self.batch_norm(conv3)
                bn3 = tf.layers.batch_normalization(conv3, training=is_training)
                lr3 = self.leaky_relu(bn3, alpha=0.1)

            with tf.variable_scope("layer4"):  # layer4 conv nx16x16x4c -> nx8x8x4c
                conv4 = self.conv2d(lr3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                # bn4 = self.batch_norm(conv4)
                bn4 = tf.layers.batch_normalization(conv4, training=is_training)
                lr4 = self.leaky_relu(bn4, alpha=0.1)

            with tf.variable_scope("layer5"):  # layer5 conv nx8x8x4c -> nx4x4x4c
                conv5 = self.conv2d(lr4, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                # bn5 = self.batch_norm(conv5)
                bn5 = tf.layers.batch_normalization(conv5, training=is_training)
                lr5 = self.leaky_relu(bn5, alpha=0.1)

            with tf.variable_scope("layer6"):  # layer6 fc nx4x4x4c -> nx200
                shape6 = tf.shape(lr5)
                reshape6 = tf.reshape(lr5, [shape6[0], shape6[1] * shape6[2] * shape6[3]])
                _, shape6_h, shape6_w, shape6_c = lr5.get_shape().as_list()
                self.shape6_hwc = shape6_h*shape6_w*shape6_c
                fc6 = self.fully_connect(reshape6, self.shape6_hwc, self.NOISE_UNIT_NUM, self.SEED)
        return fc6


    def decoder(self, z, reuse=False, is_training=True):  # z is expected [n, 200]
        with tf.variable_scope('decoder', reuse=reuse):
            # print("self.shape6_hwc, ", self.shape6_hwc)
            with tf.variable_scope("layer1"):  # layer1 fc nx200 -> nx4*4*4c
                fc1 = self.fully_connect(z, self.NOISE_UNIT_NUM, self.shape6_hwc, self.SEED)
                # bn1 = self.batch_norm(fc1)
                bn1 = tf.layers.batch_normalization(fc1, training=is_training)
                rl1 = tf.nn.relu(bn1)

            with tf.variable_scope("layer2"):  # layer2 fc nx4*4*4c -> nx4*4*4c
                fc2 = self.fully_connect(rl1, self.shape6_hwc, 4 * 4 * self.BASE_CHANNEL * 4, self.SEED)
                # bn2 = self.batch_norm(fc2)
                bn2 = tf.layers.batch_normalization(fc2, training=is_training)
                rl2 = tf.nn.relu(bn2)
                # print("rl2, ", rl2.get_shape().as_list())
            with tf.variable_scope("layer3"):  # layer3 deconv nx4*4*4c -> nx4x4x64 -> nx8x8x64
                shape = tf.shape(rl2)
                reshape3 = tf.reshape(rl2, [shape[0], 4, 4, self.BASE_CHANNEL * 4])
                # print("reshape3, ", reshape3.get_shape().as_list())
                deconv3 = self.conv2d_transpose(reshape3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                # print("deconv3, ", deconv3.get_shape().as_list())
                # bn3 = self.batch_norm(deconv3)
                bn3 = tf.layers.batch_normalization(deconv3, training=is_training)
                rl3 = tf.nn.relu(bn3)
                # print("rl3, ", rl3.get_shape().as_list())

            with tf.variable_scope("layer4"):  # layer4 deconv nx8x8x64 -> nx16x16x64
                deconv4 = self.conv2d_transpose(rl3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                tanh4 = tf.tanh(deconv4)

            with tf.variable_scope("layer5"):  # layer5 deconv nx16x16x64 -> nx32x32x32
                deconv5 = self.conv2d_transpose(tanh4, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 2, 4, 2, self.SEED)
                tanh5 = tf.tanh(deconv5)

            with tf.variable_scope("layer6"):  # layer6 deconv nx32x32x32 -> nx64x64x16
                deconv6 = self.conv2d_transpose(tanh5, self.BASE_CHANNEL * 2, self.BASE_CHANNEL, 4, 2, self.SEED)
                tanh6 = tf.tanh(deconv6)

            with tf.variable_scope("layer7"):  # layer7 deconv nx64x64x16 -> nx128x128x3
                deconv7 = self.conv2d_transpose(tanh6, self.BASE_CHANNEL, self.IMG_CHANNEL, 7, 2, self.SEED)
                tanh7 = tf.tanh(deconv7)
        return tanh7


    def discriminator(self, x, z, reuse=False, is_training=True):  # z[n, 200], x[n, 128, 128, 3]
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope("x_layer1"):  # layer x1 conv [n, 128, 128, 3] -> [n, 64, 64, c]
                convx1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL, 7, 2, self.SEED)
                lrx1 = self.leaky_relu(convx1, alpha=0.1)
                dropx1 = tf.layers.dropout(lrx1, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("x_layer2"):  # layer x2 conv [n, 64, 64, c] -> [n, 32, 32, c]
                convx2 = self.conv2d(dropx1, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, 4, 2, self.SEED)
                lrx2 = self.leaky_relu(convx2, alpha=0.1)
                dropx2 = tf.layers.dropout(lrx2, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("x_layer3"):  # layer x3 conv [n, 32, 32, 2c] -> [n, 16, 16, 4c]
                convx3 = self.conv2d(dropx2, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, 4, 2, self.SEED)
                lrx3 = self.leaky_relu(convx3, alpha=0.1)
                dropx3 = tf.layers.dropout(lrx3, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("x_layer4"):  # layer x2 conv [n, 16, 16, 4c] -> [n, 8, 8, 4c] -> [n, 8*8*4c]
                convx4 = self.conv2d(dropx3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                # bnx4 = self.batch_norm(convx4)
                bnx4 = tf.layers.batch_normalization(convx4, training=is_training)
                lrx4 = self.leaky_relu(bnx4, alpha=0.1)
                dropx4 = tf.layers.dropout(lrx4, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)
                shapex4 = tf.shape(dropx4)
                reshapex4 = tf.reshape(dropx4, [shapex4[0], shapex4[1] * shapex4[2] * shapex4[3]])
                # _, shapex4_h, shapex4_w, shapex4_c = dropx4.get_shape().as_list()
                # print("shapex4_h, shapex4_w, shapex4_c, ", shapex4_h, shapex4_w, shapex4_c)
                # self.shape4x_hwc = shapex4_h*shapex4_w*shapex4_c

            with tf.variable_scope("z_layer1"):  # layer1 fc [n, 200] -> [n, 512]
                fcz1 = self.fully_connect(z, self.NOISE_UNIT_NUM, 512, self.SEED)
                lrz1 = self.leaky_relu(fcz1, alpha=0.1)
                dropz1 = tf.layers.dropout(lrz1, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("y_layer1"):  # y_layer1 fc [n, 8*8*4c+512], [n, 1024]
                cony1 = tf.concat([reshapex4, dropz1], axis=1)
                fcy1 = self.fully_connect(cony1, 8*8*4*self.BASE_CHANNEL + 512, 1024, self.SEED)
                lry1 = self.leaky_relu(fcy1, alpha=0.1)
                self.dropy1 = tf.layers.dropout(lry1, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("y_fc_logits"):
                logits = self.fully_connect(self.dropy1, 1024, 1, self.SEED)
                self.sigmoid = tf.nn.sigmoid(logits)

        return self.dropy1, self.sigmoid


class SIMGAN():
    def __init__(self, noise_unit_num, img_channel, anno_channel, seed, base_channel, keep_prob, base_fc_node=1024):
        self.NOISE_UNIT_NUM = noise_unit_num  # 200
        self.IMG_CHANNEL = img_channel  # 3
        self.ANNO_CHANNEL = anno_channel # 4
        self.SEED = seed
        np.random.seed(seed=self.SEED)
        self.BASE_CHANNEL = base_channel  # 32
        self.KEEP_PROB = keep_prob
        self.BASE_FC_NODE = base_fc_node


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

    


class SPIGAN(object):
    def __init__(self, input_channel, output_channel, base_channel, class_num, in_width, in_height, focal_loss_gamma,
                 seed):
        # self.INPUT_IMAGE_SIZE = 128
        self.INPUT_WIDTH = in_width
        self.INPUT_HEIGHT = in_height
        self.INPUT_CHANNEL = input_channel
        self.OUTPUT_CHANNEL = output_channel
        self.CLASS_NUM = class_num
        self.BASE_CHANNEL = base_channel
        self.FOCAL_LOSS_GAMMA = focal_loss_gamma
        self.CONCATENATE_AXIS = -1
        self.FIRST_CONV_FILTER_SIZE = 7
        self.CONV_FILTER_SIZE = 4
        self.BLOCK_FILTER_SIZE = 3
        self.CONV_STRIDE = 2
        self.BLOCK_STRIDE = 1
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        self.TVERSKY_LOSS_ALPHA = 0.85
        self.TVERSKY_LOSS_BETA = 1.0 - self.TVERSKY_LOSS_ALPHA
        self.SEED = seed

    def cal_input_num(self, input_num):
        stddev = math.sqrt(2 / (input_num))
        return stddev

    def generator(self, x, tar, keep_prob, is_training=True):
        with tf.name_scope("resnet_v2_50"):
            with tf.name_scope("conv1"):  # 128x128x3 -> 64x64x base_chan
                stdd_init = self.cal_input_num(self.INPUT_CHANNEL * self.FIRST_CONV_FILTER_SIZE * self.FIRST_CONV_FILTER_SIZE)
                # tf.summary.image('input_img', x, 30)
                conv1_W = tf.Variable(tf.truncated_normal([self.FIRST_CONV_FILTER_SIZE, self.FIRST_CONV_FILTER_SIZE,
                                                               self.INPUT_CHANNEL, self.BASE_CHANNEL]
                , mean=0.0, stddev=stdd_init, seed=self.SEED), dtype=tf.float32, name='weights')
                conv1_b = tf.Variable(tf.zeros([self.BASE_CHANNEL]), dtype=tf.float32, name='biases')
                conv_init = tf.nn.conv2d(x, conv1_W, [1, 2, 2, 1], padding="SAME") + conv1_b
                conv_init_relu = tf.nn.relu(conv_init)

            with tf.name_scope("block1"):  # 64x64x64 -> 32x32x256
                with tf.name_scope("unit_1"):
                    b1_u1 = self._add_Bottle_Neck(conv_init_relu, self.BASE_CHANNEL, self.BASE_CHANNEL,
                                        self.BASE_CHANNEL * 4, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_2"):
                    b1_u2 = self._add_Bottle_Neck(b1_u1, self.BASE_CHANNEL * 4, self.BASE_CHANNEL,
                                        self.BASE_CHANNEL * 4, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_3"):
                    b1_u3 = self._add_Bottle_Neck(b1_u2, self.BASE_CHANNEL * 4, self.BASE_CHANNEL,
                                        self.BASE_CHANNEL * 4, self.BLOCK_FILTER_SIZE, 2, self.SEED, is_training=is_training)

            with tf.name_scope("block2"):  # 32x32x256 -> 16x16x512
                with tf.name_scope("unit_1"):
                    b2_u1 = self._add_Bottle_Neck(b1_u3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 2,
                                        self.BASE_CHANNEL * 8, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_2"):
                    b2_u2 = self._add_Bottle_Neck(b2_u1, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 2,
                                        self.BASE_CHANNEL * 8, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_3"):
                    b2_u3 = self._add_Bottle_Neck(b2_u2, self.BASE_CHANNEL * 8, self.BASE_CHANNEL* 2,
                                        self.BASE_CHANNEL * 8, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_4"):
                    b2_u4 = self._add_Bottle_Neck(b2_u3, self.BASE_CHANNEL * 8, self.BASE_CHANNEL* 2,
                                        self.BASE_CHANNEL * 8, self.BLOCK_FILTER_SIZE, 2, self.SEED, is_training=is_training)

            with tf.name_scope("block3"):  # 16x16x512 -> 8x8x1024
                with tf.name_scope("unit_1"):
                    b3_u1 = self._add_Bottle_Neck(b2_u4, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 4,
                                        self.BASE_CHANNEL * 16, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_2"):
                    b3_u2 = self._add_Bottle_Neck(b3_u1, self.BASE_CHANNEL * 16, self.BASE_CHANNEL * 4,
                                        self.BASE_CHANNEL * 16, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_3"):
                    b3_u3 = self._add_Bottle_Neck(b3_u2, self.BASE_CHANNEL * 16, self.BASE_CHANNEL* 4,
                                        self.BASE_CHANNEL * 16, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_4"):
                    b3_u4 = self._add_Bottle_Neck(b3_u3, self.BASE_CHANNEL * 16, self.BASE_CHANNEL* 4,
                                        self.BASE_CHANNEL * 16, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_5"):
                    b3_u5 = self._add_Bottle_Neck(b3_u4, self.BASE_CHANNEL * 16, self.BASE_CHANNEL* 4,
                                        self.BASE_CHANNEL * 16, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_6"):
                    b3_u6 = self._add_Bottle_Neck(b3_u5, self.BASE_CHANNEL * 16, self.BASE_CHANNEL* 4,
                                        self.BASE_CHANNEL * 16, self.BLOCK_FILTER_SIZE, 2, self.SEED, is_training=is_training)

            with tf.name_scope("block4"):  # 8x8x1024 -> 8x8x2048
                with tf.name_scope("unit_1"):
                    b4_u1 = self._add_Bottle_Neck(b3_u6, self.BASE_CHANNEL * 16, self.BASE_CHANNEL * 8,
                                        self.BASE_CHANNEL * 32, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_2"):
                    b4_u2 = self._add_Bottle_Neck(b4_u1, self.BASE_CHANNEL * 32, self.BASE_CHANNEL * 8,
                                        self.BASE_CHANNEL * 32, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

                with tf.name_scope("unit_3"):
                    b4_u3 = self._add_Bottle_Neck(b4_u2, self.BASE_CHANNEL * 32, self.BASE_CHANNEL* 8,
                                        self.BASE_CHANNEL * 32, self.BLOCK_FILTER_SIZE, self.BLOCK_STRIDE, self.SEED, is_training=is_training)

            with tf.name_scope("postnorm"):
                # beta = tf.Variable(tf.zeros([self.BASE_CHANNEL * 32]), dtype=tf.float32, name='beta')
                # scale = tf.Variable(tf.ones([self.BASE_CHANNEL * 32]), dtype=tf.float32, name='gamma')
                # batch_mean, batch_var = tf.nn.moments(b4_u3, [0, 1, 2])
                # postnorm = tf.nn.batch_normalization(b4_u3, batch_mean, batch_var, beta, scale, 0.0001)
                postnorm = self.batch_norm_wrapper(b4_u3, is_training=is_training)

        # all_vars = tf.all_variables()
        half_trainable_vars = tf.trainable_variables()
        half_all_vars = tf.all_variables()


        with tf.name_scope("decorder"):
            with tf.name_scope("block1"):
                with tf.name_scope("unit_1"):
                    db1_u1 = self._add_UpBlock(postnorm, b3_u6, self.BASE_CHANNEL * 32, self.BASE_CHANNEL * 16,
                                               self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 8, 3, 1, self.SEED, keep_prob, is_training=is_training)
            with tf.name_scope("block2"):
                with tf.name_scope("unit_1"):
                    db2_u1 = self._add_UpBlock(db1_u1, b2_u4, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 8,
                                               self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, 3, 1, self.SEED, keep_prob, is_training=is_training)

            with tf.name_scope("block3"):
                with tf.name_scope("unit_1"):
                    db3_u1 = self._add_UpBlock(db2_u1, b1_u3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4,
                                               self.BASE_CHANNEL, self.BASE_CHANNEL, 3, 1, self.SEED, keep_prob, is_training=is_training)

            with tf.name_scope("block4"):
                with tf.name_scope("unit_1"):
                    db4_u1 = self._add_UpBlock(db3_u1, conv_init_relu, self.BASE_CHANNEL, self.BASE_CHANNEL,
                                               self.BASE_CHANNEL // 2, self.BASE_CHANNEL // 2, 3, 1, self.SEED, keep_prob, is_training=is_training)

            with tf.name_scope("conv_last"):
                concat_last = tf.concat([db4_u1, x], axis=3)
                stdd_last = self.cal_input_num((self.BASE_CHANNEL // 2 + self.INPUT_CHANNEL) * 3 * 3)
                convL_W = tf.Variable(tf.truncated_normal([3, 3, (self.BASE_CHANNEL // 2 + self.INPUT_CHANNEL), self.OUTPUT_CHANNEL]
                                                          , mean=0.0, stddev=stdd_last, seed=self.SEED),
                                      dtype=tf.float32, name='weights')
                convL_b = tf.Variable(tf.zeros([self.OUTPUT_CHANNEL]), dtype=tf.float32, name='biases')
                self.out = tf.nn.conv2d(concat_last, convL_W, [1, 1, 1, 1], padding="SAME") + convL_b
                self.prob = tf.nn.softmax(self.out)

            with tf.name_scope("loss_accuracy"):
                # self.total_loss, self.accuracy = self.get_focal_loss_accuracy(self.prob, tar)
                self.total_loss, self.accuracy = self.get_Tversky_loss_accuracy(self.prob, tar)
        return self.total_loss, self.accuracy, self.prob, half_trainable_vars, half_all_vars

    def batch_norm_train(self, inputs, pop_mean, pop_var, beta, scale, decay=0.99):
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 1e-4)

    def batch_norm_wrapper(self, inputs, decay=0.99, is_training=True):
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), dtype=tf.float32, name='beta')
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), dtype=tf.float32, name='gamma')
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False, name='moving_mean')
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False, name='moving_variance')
        #tf.cond(c, func1, func2)
        result = tf.cond(is_training, lambda:self.batch_norm_train(inputs, pop_mean, pop_var, beta, scale, decay),
                         lambda:tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 1e-4))
        # if is_training:
        #     batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        #     train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        #     train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        #     with tf.control_dependencies([train_mean, train_var]):
        #         return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 1e-4)
        # else:
        #     return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 1e-4)
        return result

    def _add_Bottle_Neck(self, sequence, in_filter_count, hidden_filter_count, out_filter_count, filter_size, last_stride,
                         seed, is_training=True): #there is conv in short cut
        with tf.name_scope("bottleneck_v2"):
            preact = self._Base_Unit_PreAct(sequence, in_filter_count, "preact", is_training=is_training)
            conv1 = self._Base_Unit(preact, in_filter_count, hidden_filter_count, 1, 1, seed, "conv1",
                                    batchNorm_relu_nonBias_flag=True, is_training=is_training)
            conv2 = self._Base_Unit(conv1, hidden_filter_count, hidden_filter_count, 3, 1, seed, "conv2",
                                    batchNorm_relu_nonBias_flag=True, is_training=is_training)
            conv3 = self._Base_Unit(conv2, hidden_filter_count, out_filter_count, 1, last_stride, seed, "conv3",
                                    batchNorm_relu_nonBias_flag=False, is_training=is_training)
            if in_filter_count != out_filter_count:
                preact = self._conv_in_shortcut(preact, in_filter_count, out_filter_count, 1, 1, seed)
            elif last_stride==2:
                preact = self._pool_in_shortcut(preact, in_filter_count, stride=2)
            add = tf.add(conv3, preact)
            return add


    def _Base_Unit(self, sequence, in_filter_count, out_filter_count, filter_size, stride, seed, name,
                   batchNorm_relu_nonBias_flag=True, is_training=True):
        with tf.name_scope(name):
            stdd = self.cal_input_num(filter_size * filter_size * in_filter_count)
            convW = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_filter_count, out_filter_count]
                , mean=0.0, stddev=stdd, seed=seed), dtype=tf.float32, name='weights')
            if batchNorm_relu_nonBias_flag:
                conv = tf.nn.conv2d(sequence, convW, [1, stride, stride, 1], padding="SAME")
                with tf.name_scope("BatchNorm"):
                    # beta = tf.Variable(tf.zeros([out_filter_count]), dtype=tf.float32, name='beta')
                    # scale = tf.Variable(tf.ones([out_filter_count]), dtype=tf.float32, name='gamma')
                    # pop_mean = tf.Variable(tf.zeros([out_filter_count]), trainable=False)
                    # pop_var = tf.Variable(tf.ones([out_filter_count]), trainable=False)
                    conv = self.batch_norm_wrapper(conv, is_training=is_training)
                conv = tf.nn.relu(conv)
            else:
                convb = tf.Variable(tf.zeros([out_filter_count]), dtype=tf.float32, name='biases')
                conv = tf.nn.conv2d(sequence, convW, [1, stride, stride, 1], padding="SAME") + convb
            return conv


    def _Base_Unit_PreAct(self, sequence, in_filter_count, name, is_training=True):
        with tf.name_scope(name):
            # with tf.name_scope("BatchNorm"):
            # beta = tf.Variable(tf.zeros([in_filter_count]), dtype=tf.float32, name='beta')
            # scale = tf.Variable(tf.ones([in_filter_count]), dtype=tf.float32, name='gamma')
            # batch_mean, batch_var = tf.nn.moments(sequence, [0, 1, 2])
            # sequence = tf.nn.batch_normalization(sequence, batch_mean, batch_var, beta, scale, 0.0001)
            sequence = self.batch_norm_wrapper(sequence, is_training=is_training)
            relu = tf.nn.relu(sequence)
            return relu


    def _conv_in_shortcut(self, sequence, in_filter_count, out_filter_count, filter_size, stride, seed):
        with tf.name_scope('shortcut'):
            stdd = self.cal_input_num(filter_size * filter_size * in_filter_count)
            convW = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_filter_count, out_filter_count]
                , mean=0.0, stddev=stdd, seed=seed), dtype=tf.float32, name='weights')
            convb = tf.Variable(tf.zeros([out_filter_count]), dtype=tf.float32, name='biases')
            conv = tf.nn.conv2d(sequence, convW, [1, stride, stride, 1], padding="SAME") + convb
            return conv


    def _pool_in_shortcut(self, sequence, filter_size, stride):
        with tf.name_scope('shortcut'):
            pool = tf.nn.max_pool(sequence, [1, filter_size, filter_size, 1], [1, stride, stride, 1], padding="SAME")
            return pool


    def _add_UpBlock(self, sequence, sequence_sho, in_filter_count_seq, in_filter_count_sho, hidden_filter_count,
                     out_filter_count, filter_size, stride, seed, keep_prob, is_training=True):
        dec = self._add_BottleNeck_dec(sequence, sequence_sho, in_filter_count_seq, in_filter_count_sho, hidden_filter_count,
                                       out_filter_count, filter_size, stride, seed, keep_prob, is_training=is_training)
        return dec


    def _add_BottleNeck_dec(self, sequence, sequence_sho, in_filter_count_seq, in_filter_count_sho, hidden_filter_count,
                            out_filter_count, filter_size, stride, seed, keep_prob, is_training=True):
        # relu1 = tf.nn.relu(sequence)
        concat1 = tf.concat([sequence, sequence_sho], axis=3)
        # chan_seq = tf.shape(sequence)[0]
        # chan_seq_sho = tf.shape(sequence_sho)[0]

        stdd1 = self.cal_input_num(filter_size * filter_size * (in_filter_count_seq + in_filter_count_sho))
        conv1W = tf.Variable(tf.truncated_normal([filter_size, filter_size, (in_filter_count_seq + in_filter_count_sho),
                                                  hidden_filter_count], mean=0.0, stddev=stdd1, seed=seed),
                             dtype=tf.float32, name='weights1')
        conv1b = tf.Variable(tf.zeros([hidden_filter_count]), dtype=tf.float32, name='biases1')
        conv1 = tf.nn.conv2d(concat1, conv1W, [1, stride, stride, 1], padding="SAME") + conv1b

        # beta2 = tf.Variable(tf.zeros([hidden_filter_count]), dtype=tf.float32, name='beta')
        # scale2 = tf.Variable(tf.ones([hidden_filter_count]), dtype=tf.float32, name='gamma')
        # batch_mean2, batch_var2 = tf.nn.moments(conv1, [0, 1, 2])
        # bn2 = tf.nn.batch_normalization(conv1, batch_mean2, batch_var2, beta2, scale2, 0.0001)
        bn2 = self.batch_norm_wrapper(conv1, is_training=is_training)

        relu2 = tf.nn.relu(bn2)

        stdd3 = self.cal_input_num(filter_size * filter_size * hidden_filter_count)
        conv3W = tf.Variable(tf.truncated_normal(
            [filter_size, filter_size, hidden_filter_count, (in_filter_count_seq + in_filter_count_sho)]
            , mean=0.0, stddev=stdd3, seed=seed), dtype=tf.float32, name='weights2')
        conv3b = tf.Variable(tf.zeros([(in_filter_count_seq + in_filter_count_sho)]), dtype=tf.float32, name='biases2')
        conv3 = tf.nn.conv2d(relu2, conv3W, [1, stride, stride, 1], padding="SAME") + conv3b

        add3 = tf.add(conv3, concat1)

        temp_batch_size_0 = tf.shape(add3)[0]
        temp_batch_size_1 = tf.shape(add3)[1]
        temp_batch_size_2 = tf.shape(add3)[2]
        temp_batch_size_3 = tf.shape(add3)[3]
        output_shape = tf.stack([temp_batch_size_0, temp_batch_size_1 * 2, temp_batch_size_2 * 2, out_filter_count])
        stdd4 = self.cal_input_num(filter_size * filter_size * out_filter_count)
        conv4W = tf.Variable(tf.truncated_normal([filter_size, filter_size, out_filter_count, (in_filter_count_seq + in_filter_count_sho)],
                                                 mean=0.0, stddev=stdd4, seed=seed), dtype=tf.float32,
                             name='weights3')
        conv4b = tf.Variable(tf.zeros([out_filter_count]), dtype=tf.float32, name='biases3')
        conv4 = tf.nn.conv2d_transpose(add3, conv4W, output_shape=output_shape, strides=[1, 2, 2, 1],
                                       padding="SAME") + conv4b
        drop4 = tf.nn.dropout(conv4, keep_prob=keep_prob)
        return drop4


    def get_crossEntropy_accuracy(self, prob, tar):
        crossEntropy_loss = - tf.reduce_mean(tf.multiply(tar, tf.log(tf.clip_by_value(prob, 1e-10, 1.0))),
                                             name='crossEntropy_loss')
        correct_prediction = tf.equal(tf.argmax(prob, 3), tf.argmax(tar, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return crossEntropy_loss, accuracy


    def get_focal_loss_accuracy(self, prob, tar, gamma=2.0, alpha=0.75):
        alpha_tf = tf.constant([alpha, 1.0 - alpha], dtype=tf.float32)
        focal_coeff = tf.multiply(tf.pow(1.0 - prob, gamma), alpha_tf)
        focal_loss = - tf.reduce_mean(
            tf.multiply(focal_coeff, tf.multiply(tar, tf.log(tf.clip_by_value(prob, 1e-10, 1.0)))),
            name='softmax_loss')
        correct_prediction = tf.equal(tf.argmax(prob, 3), tf.argmax(tar, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return focal_loss, accuracy


    def get_Tversky_loss_accuracy(self, prob, tar):
        prob_class0, _ = tf.split(prob, [1, self.OUTPUT_CHANNEL - 1], axis=3)
        tar_class0, _ = tf.split(tar, [1, self.OUTPUT_CHANNEL - 1], axis=3)
        intersection = tf.reduce_sum(tf.multiply(prob_class0, tar_class0))
        # prob_class0_shape = [tf.shape(prob_class0)[0], tf.shape(prob_class0)[1], tf.shape(prob_class0)[2], 1]
        # const_1_img = tf.constant(1.0, dtype=tf.float32, shape=prob_class0_shape)
        const_1_img = tf.constant(1.0, dtype=tf.float32)
        false_positive = tf.reduce_sum(tf.multiply(const_1_img - tar_class0, prob_class0))
        false_negative = tf.reduce_sum(tf.multiply(tar_class0, const_1_img - prob_class0))
        # tversky_loss_alpha = tf.constant(self.TVERSKY_LOSS_ALPHA, shape=tf.shape(false_positive), dtype=tf.float32)
        # tversky_loss_beta = tf.constant(self.TVERSKY_LOSS_BETA, shape=tf.shape(false_negative), dtype=tf.float32)
        tversky_loss_alpha = tf.constant(self.TVERSKY_LOSS_ALPHA, dtype=tf.float32)
        tversky_loss_beta = tf.constant(self.TVERSKY_LOSS_BETA, dtype=tf.float32)
        tversky_index = tf.div(intersection, (intersection + tversky_loss_alpha * false_positive + tversky_loss_beta * false_negative))
        # const_1 = tf.constant(1.0, shape=tf.shape(tversky_index), dtype=tf.float32)
        loss = const_1_img - tversky_index

        correct_prediction = tf.equal(tf.argmax(prob, 3), tf.argmax(tar, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return loss, accuracy


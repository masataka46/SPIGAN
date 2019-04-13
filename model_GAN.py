import numpy as np
# import os
import tensorflow as tf
# from PIL import Image
# import utility as Utility
# import argparse

class BiGAN_gyoza():
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
        self.BASE_CHANNEL = base_channel  # 16
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

    def batch_norm(self, input):
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

    def refiner(self, back, anno, reuse=False, is_training=True):  # x is expected [n, 128, 128, 1]
        with tf.variable_scope('refiner', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer1 nx128x128x3 x2img -> conv nx128x128x6 -> nx64x64xc
                concat1 = tf.concat([back, anno], axis=3)
                conv1 = self.conv2d(concat1, self.IMG_CHANNEL+self.ANNO_CHANNEL, self.BASE_CHANNEL, 7, 2, self.SEED)
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

            with tf.variable_scope("layer6"):  # layer6 deconv nx4*4*4c -> nx4x4x64 -> nx8x8x64
                deconv6 = self.conv2d_transpose(lr5, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                bn6 = tf.layers.batch_normalization(deconv6, training=is_training)
                rl6 = tf.nn.relu(bn6)

            with tf.variable_scope("layer7"):  # layer7 deconv nx8x8x64 -> nx16x16x64
                deconv7 = self.conv2d_transpose(rl6, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                bn7 = tf.layers.batch_normalization(deconv7, training=is_training)
                relu7 = tf.relu(bn7)

            with tf.variable_scope("layer8"):  # layer8 deconv nx16x16x64 -> nx32x32x32
                deconv8 = self.conv2d_transpose(relu7, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 2, 4, 2, self.SEED)
                bn8 = tf.layers.batch_normalization(deconv8, training=is_training)
                relu8 = tf.relu(bn8)

            with tf.variable_scope("layer9"):  # layer9 deconv nx32x32x32 -> nx64x64x16
                deconv9 = self.conv2d_transpose(relu8, self.BASE_CHANNEL * 2, self.BASE_CHANNEL, 4, 2, self.SEED)
                bn9 = tf.layers.batch_normalization(deconv9, training=is_training)
                relu9 = tf.relu(bn9)

            with tf.variable_scope("layer10"):  # layer10 deconv nx64x64x16 -> nx128x128x4
                deconv10 = self.conv2d_transpose(relu9, self.BASE_CHANNEL, self.IMG_CHANNEL, 7, 2, self.SEED)
                rgb10, alpha10 = tf.split(deconv10, [3, 1], axis=3)
                rgb_out = tf.tanh(rgb10)
                alpha_out = tf.nn.sigmoid(alpha10)
        return rgb_out, alpha_out


    def discriminator(self, x, reuse=False, is_training=True, keep_prob=1.0):  # x[n, 128, 128, 3]
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer 1 conv [n, 128, 128, 3] -> [n, 64, 64, c]
                conv1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL, 7, 2, self.SEED)
                bn1 = tf.layers.batch_normalization(conv1, training=is_training)
                lr1 = self.leaky_relu(bn1, alpha=0.1)

            with tf.variable_scope("layer2"):  # layer 2 conv [n, 64, 64, c] -> [n, 32, 32, 2c]
                conv2 = self.conv2d(lr1, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, 4, 2, self.SEED)
                bn2 = tf.layers.batch_normalization(conv2, training=is_training)
                lr2 = self.leaky_relu(bn2, alpha=0.1)

            with tf.variable_scope("layer3"):  # layer 3 conv [n, 32, 32, 2c] -> [n, 16, 16, 4c]
                conv3 = self.conv2d(lr2, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, 4, 2, self.SEED)
                bn3 = tf.layers.batch_normalization(conv3, training=is_training)
                lr3 = self.leaky_relu(bn3, alpha=0.1)

            with tf.variable_scope("layer4"):  # layer 4 conv [n, 16, 16, 4c] -> [n, 8, 8, 4c]
                conv4 = self.conv2d(lr3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                bn4 = tf.layers.batch_normalization(conv4, training=is_training)
                lr4 = self.leaky_relu(bn4, alpha=0.1)

            with tf.variable_scope("layer5"):  # layer 5 conv [n, 8, 8, 4c] -> [n, 4, 4, 4c]
                conv5 = self.conv2d(lr4, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4, 3, 2, self.SEED)
                bn5 = tf.layers.batch_normalization(conv5, training=is_training)
                lr5 = self.leaky_relu(bn5, alpha=0.1)

            with tf.variable_scope("layer6"):  # layer6 fc [n, 4, 4, 4c] -> [n, 4*4*4c] -> [n, 1024]
                shape5 = tf.shape(lr5)
                reshape6 = tf.reshape(lr5, [shape5[0], shape5[1] * shape5[2] * shape5[3]])
                b6, h6, w6, c6 = lr5.get_shape().as_list()
                fc6 = self.fully_connect(reshape6, h6*w6*c6, self.BASE_FC_NODE, self.SEED)
                lr6 = self.leaky_relu(fc6, alpha=0.1)
                self.drop6 = tf.layers.dropout(lr6, rate=1.0 - keep_prob, name='dropout', training=is_training)

            with tf.variable_scope("layer7"): # layer7 fc [n, 1024] -> [n, 1]
                logits = self.fully_connect(self.drop6, 1024, 1, self.SEED)
                self.sigmoid = tf.nn.sigmoid(logits)

        return self.drop6, self.sigmoid



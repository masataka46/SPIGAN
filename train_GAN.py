import numpy as np
import os
import tensorflow as tf
import utility as Utility
import argparse
from model_GAN import SPIGAN as Model
from make_datasets import Make_dataset as Make_datasets
import time

class MainProcess(object):
    def __init__(self, batch_size=8, log_file_name='log01', epoch=100,
                 syn_dir_name='', real_train_dir_name='', real_val_dir_name='', syn_seg_dir_name='', real_seg_dir_name='',
                 depth_dir_name='', valid_span=1, restored_model_name='', save_model_span=10, base_channel=16,
                 path_to_vgg19='', output_img_span=1, base_channel_pre=16):

        self.batch_size = batch_size
        self.logfile_name = log_file_name
        self.epoch = epoch
        self.syn_dir_name = syn_dir_name
        self.real_train_dir_name = real_train_dir_name
        self.real_val_dir_name = real_val_dir_name
        self.syn_seg_dir_name = syn_seg_dir_name
        self.real_seg_dir_name = real_seg_dir_name
        self.depth_dir_name = depth_dir_name
        self.img_width = 320
        self.img_height = 320
        self.img_width_be_crop_syn = 640
        self.img_width_be_crop_real = 760
        self.img_height_be_crop = 380
        self.img_channel = 3
        self.anno_channel = 19
        self.depth_channel = 1
        self.base_channel = base_channel
        self.base_channel_pre = base_channel_pre
        self.test_data_sample = 5 * 5
        self.l2_norm = 0.001
        self.keep_prob_rate = 0.5
        self.loss_alpha = 1.0
        self.loss_beta = 0.5
        self.loss_gamma = 0.1
        self.loss_delta = 0.33
        self.seed = 1234
        self.crop_flag = True
        self.valid_span = valid_span
        self.path_to_vgg19 = path_to_vgg19
        np.random.seed(seed=self.seed)
        self.board_dir_name = 'tensorboard/' + self.logfile_name
        self.out_img_dir = 'out_images' #output image file
        self.out_model_dir = 'out_models' #output model file
        self.restore_model_name = restored_model_name
        self.save_model_span = save_model_span
        self.output_img_span = output_img_span
        self.reconst_lambda = 0.1

        try:
            os.mkdir('log')
        except:
            pass
        try:
            os.mkdir('tensorboard')
        except:
            pass
        try:
            os.mkdir('out_graph')
        except:
            pass
        try:
            os.mkdir(self.out_img_dir)
        except:
            pass
        try:
            os.mkdir(self.out_model_dir)
        except:
            pass
        try:
            os.mkdir('./out_histogram') #for debug
        except:
            pass
        try:
            os.mkdir('./out_images_Debug') #for debug
        except:
            pass

        self.model = Model(self.img_channel, self.anno_channel, self.seed, self.base_channel, self.keep_prob_rate,
                           self.path_to_vgg19, base_channel_predictor=self.base_channel_pre)

        '''
        syn_dir_name, real_train_dir_name, real_val_dir_name, syn_seg_dir_name, real_seg_dir_name, depth_dir_name,
                 img_width, img_height, img_width_be_crop_syn, img_width_be_crop_real, img_height_be_crop
        '''
        self.make_datasets = Make_datasets(self.syn_dir_name, self.real_train_dir_name, self.real_val_dir_name, self.syn_seg_dir_name,
                                           self.real_seg_dir_name, self.depth_dir_name, self.img_width, self.img_height,
                                  self.img_width_be_crop_syn, self.img_width_be_crop_real, self.img_height_be_crop,
                                           self.seed, self.crop_flag)

        self.x_s = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channel], name='x_s')  # synthesis image
        self.x_r = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channel], name='x_r')  # real image
        self.seg = tf.placeholder(tf.int32, [None, self.img_height, self.img_width], name='seg_label')
        self.pi = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.depth_channel], name='depth_label')
        self.tar_d_r = tf.placeholder(tf.float32, [None, self.img_height//8, self.img_width//8, 1], name='target_discriminator_for_real')
        self.tar_d_f = tf.placeholder(tf.float32, [None, self.img_height//8, self.img_width//8, 1], name='target_discriminator_for_fake')
        self.x_r_v = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channel], name='x_r_v')  # real image to T

        self.is_training = tf.placeholder(tf.bool, name = 'is_training')
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

        with tf.variable_scope('generator_model'):
            self.g_out = self.model.generator(self.x_s, reuse=False, is_training=self.is_training)

        with tf.variable_scope('discriminator_model'):
            self.d_out_r = self.model.discriminator(self.x_r, reuse=False, is_training=self.is_training, keep_prob=self.keep_prob) #real
            self.d_out_f = self.model.discriminator(self.g_out, reuse=True, is_training=self.is_training, keep_prob=self.keep_prob) #fake

        with tf.variable_scope('task_predictor_model'):
            self.t_out_s = self.model.task_predictor(self.x_s, reuse=False, is_training=self.is_training, keep_prob=self.keep_prob) #synthesis
            self.t_out_g = self.model.task_predictor(self.g_out, reuse=True, is_training=self.is_training, keep_prob=self.keep_prob) #generated
            self.t_out_r = self.model.task_predictor(self.x_r_v, reuse=True, is_training=self.is_training, keep_prob=self.keep_prob) #real


        with tf.variable_scope('privileged_network_model'):
            self.p_out_s = self.model.privileged_network(self.x_s, reuse=False, is_training=self.is_training, keep_prob=self.keep_prob) #synthesis
            self.p_out_g = self.model.privileged_network(self.g_out, reuse=True, is_training=self.is_training, keep_prob=self.keep_prob) #generated


        with tf.name_scope("loss"):
            #adversarial loss
            self.loss_dis_r = tf.reduce_mean(tf.square(self.d_out_r - self.tar_d_r), name='Loss_dis_real') #loss related to real
            self.loss_dis_f = tf.reduce_mean(tf.square(self.d_out_f - self.tar_d_f), name='Loss_dis_fake') #loss related to fake
            self.loss_adv = self.loss_dis_r + self.loss_dis_f

            #task prediction loss
            seg_oneHot = tf.one_hot(self.seg, self.anno_channel, dtype=tf.float32)
            self.loss_task_s = - tf.reduce_mean(tf.multiply(seg_oneHot, tf.log(tf.clip_by_value(self.t_out_s, 1e-10, 1.0))),
                                                 name='task_prediction_loss_for_synthesis')
            correct_prediction_s = tf.equal(tf.argmax(self.t_out_s, 3), tf.argmax(seg_oneHot, 3))
            self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction_s, "float"))
            self.loss_task_g = - tf.reduce_mean(tf.multiply(seg_oneHot, tf.log(tf.clip_by_value(self.t_out_g, 1e-10, 1.0))),
                                                 name='task_prediction_loss_for_generated')
            correct_prediction_g = tf.equal(tf.argmax(self.t_out_g, 3), tf.argmax(seg_oneHot, 3))
            self.accuracy_g = tf.reduce_mean(tf.cast(correct_prediction_g, "float"))
            self.loss_task = self.loss_task_s + self.loss_task_g

            #PI regularization
            self.loss_PI_s = tf.reduce_mean(tf.abs(self.p_out_s - self.pi))
            self.loss_PI_g = tf.reduce_mean(tf.abs(self.p_out_g - self.pi))
            self.loss_PI = self.loss_PI_s + self.loss_PI_g

            #perceptual loss
            self.conv1_2_s, self.conv2_2_s, self.conv3_2_s, self.conv4_2_s, self.conv5_2_s = self.model.vgg19(self.x_s, reuse=False)
            self.conv1_2_g, self.conv2_2_g, self.conv3_2_g, self.conv4_2_g, self.conv5_2_g = self.model.vgg19(self.g_out, reuse=True)
            conv1_2_shape = self.conv1_2_s.get_shape().as_list()
            conv2_2_shape = self.conv2_2_s.get_shape().as_list()
            conv3_2_shape = self.conv3_2_s.get_shape().as_list()
            conv4_2_shape = self.conv4_2_s.get_shape().as_list()
            conv5_2_shape = self.conv5_2_s.get_shape().as_list()
            conv1_2_lambda = tf.Variable(1./(conv1_2_shape[1] * conv1_2_shape[2] * conv1_2_shape[3]), dtype=tf.float32)
            conv2_2_lambda = tf.Variable(1./(conv2_2_shape[1] * conv2_2_shape[2] * conv2_2_shape[3]), dtype=tf.float32)
            conv3_2_lambda = tf.Variable(1./(conv3_2_shape[1] * conv3_2_shape[2] * conv3_2_shape[3]), dtype=tf.float32)
            conv4_2_lambda = tf.Variable(1./(conv4_2_shape[1] * conv4_2_shape[2] * conv4_2_shape[3]), dtype=tf.float32)
            conv5_2_lambda = tf.Variable(1./(conv5_2_shape[1] * conv5_2_shape[2] * conv5_2_shape[3]), dtype=tf.float32)
            self.conv1_2 = tf.multiply(conv1_2_lambda, tf.reduce_sum(tf.abs(self.conv1_2_s - self.conv1_2_g)))
            self.conv2_2 = tf.multiply(conv2_2_lambda, tf.reduce_sum(tf.abs(self.conv2_2_s - self.conv2_2_g)))
            self.conv3_2 = tf.multiply(conv3_2_lambda, tf.reduce_sum(tf.abs(self.conv3_2_s - self.conv3_2_g)))
            self.conv4_2 = tf.multiply(conv4_2_lambda, tf.reduce_sum(tf.abs(self.conv4_2_s - self.conv4_2_g)))
            self.conv5_2 = tf.multiply(conv5_2_lambda, tf.reduce_sum(tf.abs(self.conv5_2_s - self.conv5_2_g)))
            self.loss_perc = self.conv1_2 + self.conv2_2 + self.conv3_2 + self.conv4_2 + self.conv5_2

            #total loss
            self.loss_dis_total = self.loss_alpha * self.loss_adv
            self.loss_task_total = self.loss_beta * self.loss_task
            self.loss_PI_total = self.loss_gamma * self.loss_PI
            self.loss_gen_total = self.loss_alpha * self.loss_dis_f + self.loss_beta * self.loss_task + \
                                  self.loss_gamma * self.loss_PI + self.loss_delta * self.loss_perc

        tf.summary.scalar('self.loss_dis_r', self.loss_dis_r)
        tf.summary.scalar('self.loss_dis_f', self.loss_dis_f)
        tf.summary.scalar('self.loss_adv', self.loss_adv)
        tf.summary.scalar('self.loss_task_s', self.loss_task_s)
        tf.summary.scalar('self.accuracy_s', self.accuracy_s)
        tf.summary.scalar('self.loss_task_g', self.loss_task_g)
        tf.summary.scalar('self.accuracy_g', self.accuracy_g)
        tf.summary.scalar('self.loss_PI_s', self.loss_PI_s)
        tf.summary.scalar('self.loss_PI_g', self.loss_PI_g)
        tf.summary.scalar('self.loss_PI', self.loss_PI)
        tf.summary.scalar('self.conv1_2', self.conv1_2)
        tf.summary.scalar('self.conv2_2', self.conv2_2)
        tf.summary.scalar('self.conv3_2', self.conv3_2)
        tf.summary.scalar('self.conv4_2', self.conv4_2)
        tf.summary.scalar('self.conv5_2', self.conv5_2)
        tf.summary.scalar('self.loss_perc', self.loss_perc)
        tf.summary.scalar('self.loss_dis_total', self.loss_dis_total)
        tf.summary.scalar('self.loss_task_total', self.loss_task_total)
        tf.summary.scalar('self.loss_PI_total', self.loss_PI_total)
        tf.summary.scalar('self.loss_gen_total', self.loss_gen_total)
        self.merged = tf.summary.merge_all()

        with tf.name_scope("graphkeys"):
            # t_vars = tf.trainable_variables()
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
            tas_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="task_predictor")
            pri_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="privileged_network")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.name_scope("train"):
            with tf.control_dependencies(update_ops):
                self.train_dis = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(self.loss_dis_total,
                                                                                var_list=dis_vars, name='Adam_dis')
                self.train_gen = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.5).minimize(self.loss_gen_total,
                                                                                var_list=gen_vars, name='Adam_gen')
                self.train_tas = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(self.loss_task_total,
                                                                                var_list=tas_vars, name='Adam_tas')
                self.train_pri = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(self.loss_PI_total,
                                                                                var_list=pri_vars, name='Adam_pri')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.board_dir_name, self.sess.graph)
        # saver
        self.saver = tf.train.Saver()

        if self.restore_model_name != '':
            self.saver.restore(self.sess, self.restore_model_name)
            print("model ", self.restore_model_name, " is restored.")


    def train(self):#training phase
        start = time.time()
        # log_list = []
        # log_list.append(['epoch', 'AUC', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'threshold'])
        log_list = ['epoch', 'AUC', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1', 'threshold']
        Utility.save_1row_to_csv(log_list, 'log/' + self.logfile_name + '_auc.csv')
        #training loop
        print("start training")
        for epoch in range(0, self.epoch):
            self.sess.run(tf.local_variables_initializer())
            sum_loss_dis_f_D = np.float32(0)
            sum_loss_dis_f_G = np.float32(0)
            sum_loss_dis_r_D = np.float32(0)
            sum_loss_perc = np.float32(0)
            sum_loss_dis_total = np.float32(0)
            sum_loss_gen_total = np.float32(0)
            sum_loss_task_total = np.float32(0)
            sum_loss_task_s = np.float32(0)
            sum_loss_task_g = np.float32(0)
            sum_loss_PI_s = np.float32(0)
            sum_loss_PI_g = np.float32(0)
            sum_loss_PI_total = np.float32(0)


            len_data_syn = self.make_datasets.make_data_for_1_epoch()

            for i in range(0, len_data_syn, self.batch_size):
                if i % (self.batch_size * 100) == 0:
                    print("i = ", i)
                syns_np, segs_np, depths_np, reals_np = self.make_datasets.get_data_for_1_batch(i, self.batch_size)
                tar_1 = self.make_datasets.make_target_1_0(1.0, len(syns_np), self.img_width//8, self.img_height//8) #1 ->
                tar_0 = self.make_datasets.make_target_1_0(0.0, len(syns_np), self.img_width//8, self.img_height//8) #0 ->

                self.sess.run(self.train_dis, feed_dict={self.x_s:syns_np, self.x_r:reals_np,
                                self.tar_d_r:tar_1, self.tar_d_f:tar_0, self.is_training:True, self.keep_prob:self.keep_prob_rate})
                self.sess.run(self.train_tas, feed_dict={self.x_s:syns_np, self.seg:segs_np,
                                                         self.is_training:True, self.keep_prob:self.keep_prob_rate})
                self.sess.run(self.train_pri, feed_dict={self.x_s:syns_np, self.pi:depths_np,
                                                         self.is_training:True, self.keep_prob:self.keep_prob_rate})
                self.sess.run(self.train_gen, feed_dict={self.x_s:syns_np, self.seg:segs_np, self.pi:depths_np,
                                self.tar_d_f:tar_1, self.is_training:True, self.keep_prob:self.keep_prob_rate})

                # sess.run(train_dec_opt, feed_dict={z_:z, x_: img_batch, d_dis_f_: tar_g_1, is_training_:True})
                #train encoder
                # sess.run(train_enc, feed_dict={x_:img_batch, d_dis_r_: tar_g_0, is_training_:True, z_:z})
                # sess.run(train_enc_opt, feed_dict={x_:img_batch, d_dis_r_: tar_g_0, is_training_:True})

                # loss for discriminator
                loss_dis_total_, loss_dis_r_D_, loss_dis_f_D_ = self.sess.run([self.loss_dis_total, self.loss_dis_r, self.loss_dis_f],
                                                                     feed_dict={self.x_s:syns_np, self.x_r:reals_np,
                                self.tar_d_r:tar_1, self.tar_d_f:tar_0, self.is_training:False, self.keep_prob:1.0})

                #loss for task predictor
                loss_task_total_, loss_task_s_, loss_task_g_ = self.sess.run([self.loss_task_total, self.loss_task_s, self.loss_task_g],
                                                                                 feed_dict={self.x_s:syns_np, self.seg:segs_np,
                                                         self.is_training:False, self.keep_prob:1.0})
                #loss for PI
                loss_PI_total_, loss_PI_s_, loss_PI_g_ = self.sess.run([self.loss_PI_total, self.loss_PI_s, self.loss_PI_g],
                                                                       feed_dict={self.x_s:syns_np, self.pi:depths_np,
                                                         self.is_training:False, self.keep_prob:1.0})

                #perceptual loss
                loss_perc_ = self.sess.run(self.loss_perc, feed_dict={self.x_s:syns_np,  self.tar_d_f:tar_1, 
                                                                      self.is_training:False, self.keep_prob:1.0})
                
                #generstor loss
                loss_gen_total_, loss_dis_f_G_ = self.sess.run([self.loss_gen_total, self.loss_dis_f], 
                                                               feed_dict={self.x_s:syns_np, self.seg:segs_np, self.pi:depths_np,
                                self.tar_d_f:tar_1, self.is_training:False, self.keep_prob:1.0})
                

                merged_ = self.sess.run(self.merged, feed_dict={self.x_s:syns_np, self.x_r:reals_np, self.seg:segs_np, self.pi:depths_np,
                                self.tar_d_r:tar_1, self.tar_d_f:tar_0, self.is_training:False, self.keep_prob:1.0})

                self.summary_writer.add_summary(merged_, epoch)

                sum_loss_dis_f_D += loss_dis_f_D_ * len(syns_np)
                sum_loss_dis_r_D += loss_dis_r_D_ * len(syns_np)
                sum_loss_dis_f_G += loss_dis_f_G_ * len(syns_np)
                sum_loss_perc += loss_perc_ * len(syns_np)
                sum_loss_dis_total += loss_dis_total_ * len(syns_np)
                sum_loss_gen_total += loss_gen_total_ * len(syns_np)
                sum_loss_task_total += loss_task_total_ * len(syns_np)
                sum_loss_task_s += loss_task_s_ * len(syns_np)
                sum_loss_task_g += loss_task_g_ * len(syns_np)
                sum_loss_PI_s += loss_PI_s_ * len(syns_np)
                sum_loss_PI_g += loss_PI_g_ * len(syns_np)
                sum_loss_PI_total += loss_PI_total_ * len(syns_np)

            dif_sec = time.time() - start
            hour = int(dif_sec // 3600)
            min = int((dif_sec - hour * 3600) // 60)
            sec = int(dif_sec - hour * 3600 - min * 60)
            print("---------------------------------------------------------------------------------------------")
            print(epoch, ", total time: {}hour, {}min, {}sec".format(hour, min, sec))
            print("epoch = {:}, Generator Total Loss = {:.4f}, Discriminator Total Loss = {:.4f}, "
                  "Task Predictor Total Loss = {:.4f}, Privileged Network Total Loss = {:.4f}".format(
                epoch, sum_loss_gen_total / len_data_syn, sum_loss_dis_total / len_data_syn,
                sum_loss_task_total / len_data_syn, sum_loss_PI_total / len_data_syn))
            print("Discriminator Real Loss = {:.4f}, Discriminator Fake Loss = {:.4f}, Discriminator Fake Loss for G = {:.4f}".format(
                sum_loss_dis_r_D / len_data_syn, sum_loss_dis_f_D / len_data_syn, sum_loss_dis_f_G / len_data_syn))
            print("Task Predictor Loss for Synthesis = {:.4f}, Task Predictor Loss for Generated = {:.4f}".format(
                sum_loss_task_s / len_data_syn, sum_loss_task_g / len_data_syn))
            print("Privileged information Loss for Synthesis = {:.4f}, Privileged information Loss for Generated = {:.4f}".format(
                sum_loss_PI_s / len_data_syn, sum_loss_PI_g / len_data_syn))
            print("Perceptual Loss = {:.4f}".format(sum_loss_perc / len_data_syn))


            # if epoch % self.valid_span == 0:
            #     print("validation phase")
            #     #TODO


            if epoch % self.output_img_span == 0:
                print("output image now....")
                syns_np, segs_np, depths_np, reals_np, real_segs_np = self.make_datasets.get_data_for_1_batch_for_output()
                t_out_s_, t_out_g_, g_out_ = self.sess.run( [self.t_out_s, self.t_out_g, self.g_out],
                                                    feed_dict={self.x_s: syns_np, self.is_training: False, self.keep_prob: 1.0})
                t_out_r_ = self.sess.run( self.t_out_r,
                                                    feed_dict={self.x_r_v: reals_np, self.is_training: False, self.keep_prob: 1.0})
                Utility.make_output_img(syns_np, g_out_, t_out_s_, t_out_g_, segs_np, epoch, self.logfile_name, self.out_img_dir)
                Utility.make_output_img_for_real(reals_np, t_out_r_, real_segs_np, epoch, self.logfile_name, self.out_img_dir)

            # save model
            if epoch % self.save_model_span == 0 and epoch != 0:
                saver2 = tf.train.Saver()
                _ = saver2.save(self.sess, './out_models/model_' + self.logfile_name + '_' + str(epoch) + '.ckpt')


if __name__ == '__main__':
    def parser():
        parser = argparse.ArgumentParser(description='train simGAN for oil')
        parser.add_argument('--batch_size', '-b', type=int, default=10, help='Number of images in each mini-batch')
        parser.add_argument('--log_file_name', '-lf', type=str, default='log19021501', help='log file name')
        parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch')
        parser.add_argument('--syn_dir_name', '-sdn', type=str, default='/media/webfarmer/HDCZ-UT/dataset/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/RGB/',
                            help='path to synthesis data')
        parser.add_argument('--real_train_dir_name', '-rtn', type=str, default='/media/webfarmer/HDCZ-UT/dataset/cityScape/data/leftImg8bit/train/',
                            help='path to real training data')
        parser.add_argument('--real_val_dir_name', '-rvn', type=str, default='/media/webfarmer/HDCZ-UT/dataset/cityScape/data/leftImg8bit/val/',
                            help='path to real validation data')
        parser.add_argument('--syn_seg_dir_name', '-ssn', type=str, default='/media/webfarmer/HDCZ-UT/dataset/SYNTHIA_RAND_CITYSCAPES/segmentation_annotation/SYNTHIA/GT/parsed_LABELS/',
                            help='path to synthesis label data')
        parser.add_argument('--real_seg_dir_name', '-rsn', type=str, default='/media/webfarmer/HDCZ-UT/dataset/SYNTHIA_RAND_CITYSCAPES/segmentation_annotation/Parsed_CityScape/val/',
                            help='path to real label data')
        parser.add_argument('--depth_dir_name', '-ddn', type=str, default='/media/webfarmer/HDCZ-UT/dataset/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/Depth/Depth/',
                            help='path to depth data')
        parser.add_argument('--path_to_vgg', '-pvg', type=str, default='./vgg19.npy',
                            help='path to vgg19 parameters')
        parser.add_argument('--valid_span', '-vs', type=int, default=1, help='validation span')
        parser.add_argument('--restore_model_name', '-rmn', type=str, default='', help='restored model name')
        parser.add_argument('--save_model_span', '-ss', type=int, default=1, help='span of saving model')
        parser.add_argument('--base_channel', '-bc', type=int, default=16, help='number of base channel')
        parser.add_argument('--base_channel_pre', '-bcp', type=int, default=16, help='number of base channel predictor')
        parser.add_argument('--output_img_span', '-ois', type=int, default=1, help='output image span')

        return parser.parse_args()
    args = parser()

    main_process = MainProcess(batch_size=args.batch_size, log_file_name=args.log_file_name, epoch=args.epoch,
                               syn_dir_name=args.syn_dir_name, real_train_dir_name=args.real_train_dir_name,
                               real_val_dir_name=args.real_val_dir_name, syn_seg_dir_name=args.syn_seg_dir_name,
                               real_seg_dir_name=args.real_seg_dir_name, depth_dir_name=args.depth_dir_name,
                               valid_span=args.valid_span, restored_model_name=args.restore_model_name,
                               save_model_span=args.save_model_span, base_channel=args.base_channel,
                               path_to_vgg19=args.path_to_vgg, output_img_span=args.output_img_span,
                               base_channel_pre=args.base_channel_pre)

    main_process.train()


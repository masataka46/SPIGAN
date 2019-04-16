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
                 depth_dir_name='', valid_span=1, restored_model_name='', save_model_span=10, base_channel=16, path_to_vgg19=''):
        #global variants
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
        self.test_data_sample = 5 * 5
        self.l2_norm = 0.001
        self.keep_prob_rate = 0.5
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
        self.reconst_lambda = 0.1
        # SAVE_MODEL_ITERATE_SPAN = args.save_model_iterate_span
        # BEFORE_BREAK_EVEN_POINTS = np.ones((8), dtype=np.float32) * 0.5 # (recall, precision, f1)

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

        self.model = Model(self.img_channel, self.anno_channel, self.seed, self.base_channel, self.keep_prob_rate, self.path_to_vgg19)

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

        # self.d_dis_f = tf.placeholder(tf.float32, [None, 1], name='d_dis_f') #target of discriminator related to generator
        # self.d_dis_r = tf.placeholder(tf.float32, [None, 1], name='d_dis_r') #target of discriminator related to real image
        self.tp_BEP = tf.placeholder(tf.float32, shape=(), name='tp_BEP')
        self.tn_BEP = tf.placeholder(tf.float32, shape=(), name='tn_BEP')
        self.fp_BEP = tf.placeholder(tf.float32, shape=(), name='fp_BEP')
        self.fn_BEP = tf.placeholder(tf.float32, shape=(), name='fn_BEP')
        self.precision_BEP = tf.placeholder(tf.float32, shape=(), name='precision_BEP')
        self.recall_BEP = tf.placeholder(tf.float32, shape=(), name='recall_BEP')
        self.f1_BEP = tf.placeholder(tf.float32, shape=(), name='f1_BEP')
        self.score_A_BEP = tf.placeholder(tf.float32, shape=(), name='score_A_BEP')
        self.is_training = tf.placeholder(tf.bool, name = 'is_training')
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

        # self.score_A_pred = tf.placeholder(tf.float32, [None, 1], name='score_A_pred')
        # self.score_A_tar = tf.placeholder(tf.float32, [None, 1], name='score_A_tar')

        with tf.variable_scope('generator_model'):
            self.g_out = self.model.generator(self.x_s, reuse=False, is_training=self.is_training)

        with tf.variable_scope('discriminator_model'):
            #stream around discriminator
            self.d_out_r = self.model.discriminator(self.x_r, reuse=False, is_training=self.is_training, keep_prob=self.keep_prob) #real
            self.d_out_f = self.model.discriminator(self.g_out, reuse=True, is_training=self.is_training, keep_prob=self.keep_prob) #fake

        with tf.variable_scope('task_predictor_model'):
            self.t_out_s = self.model.task_predictor(self.x_s, reuse=False, is_training=self.is_training, keep_prob=self.keep_prob) #synthesis
            self.t_out_g = self.model.task_predictor(self.g_out, reuse=True, is_training=self.is_training, keep_prob=self.keep_prob) #generated

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
            self.conv1_2_s, self.conv2_2_s, self.conv3_2_s, self.conv4_2_s, self.conv5_2_s = self.model.vgg19(self.x_s)
            self.conv1_2_g, self.conv2_2_g, self.conv3_2_g, self.conv4_2_g, self.conv5_2_g = self.model.vgg19(self.g_out)
            self.conv1_2 = tf.reduce_mean(tf.abs(self.conv1_2_s - self.conv1_2_g))
            self.conv2_2 = tf.reduce_mean(tf.abs(self.conv2_2_s - self.conv2_2_g))
            self.conv3_2 = tf.reduce_mean(tf.abs(self.conv3_2_s - self.conv3_2_g))
            self.conv4_2 = tf.reduce_mean(tf.abs(self.conv4_2_s - self.conv4_2_g))
            self.conv5_2 = tf.reduce_mean(tf.abs(self.conv5_2_s - self.conv5_2_g))
            self.loss_perc = self.conv1_2 + self.conv2_2 + self.conv3_2 + self.conv4_2 + self.conv5_2

            #total loss
            self.loss_dis_total = self.loss_adv
            self.loss_task_total = self.loss_task
            self.loss_PI_total = self.loss_PI
            self.loss_gen_total = self.loss_adv + self.loss_task + self.loss_PI
        # with tf.name_scope("score"):
        #     self.l_g = tf.reduce_mean(tf.abs(self.x - self.x_z_x), axis=(1,2,3))
        #     self.l_FM = tf.reduce_mean(tf.abs(self.drop3_r - self.drop3_re), axis=1)
        #     # score_A =  SCORE_ALPHA * l_g + (1.0 - self.score_alpha) * self.l_FM
        # 
        # with tf.name_scope("optional_loss"):
        #     loss_dec_opt = loss_dec_total + CYCLE_LAMBDA * l_g
        #     loss_enc_opt = loss_enc_total + CYCLE_LAMBDA * l_g

        # with tf.name_scope('metrics'):
        #     auc, update_op_auc = tf.metrics.auc(score_A_tar_, score_A_pred_, num_thresholds=AUC_COORD_NUM)
        #     tp_BEP_cal = tp_BEP_
        #     tn_BEP_cal = tn_BEP_
        #     fp_BEP_cal = fp_BEP_
        #     fn_BEP_cal = fn_BEP_
        #     precision_BEP_cal = precision_BEP_
        #     recall_BEP_cal = recall_BEP_
        #     f1_BEP_cal = f1_BEP_
        #     score_A_BEP_cal = score_A_BEP_
            # metrics_total = tp_BEP_cal + tn_BEP_cal + fp_BEP_cal + fn_BEP_cal + precision_BEP_cal + recall_BEP_cal + f1_BEP_cal + score_A_BEP_cal
        #     tarT = tf.argmax(tar, axis=3, output_type=tf.int32)
        #     probT = tf.argmax(prob, axis=3, output_type=tf.int32)
        #     # m_iou, conf_mat = util.cal_mean_IOU(tarT, probT, class_num)
        #     fp, fp_op = tf.metrics.false_positives(tarT, probT)
        #     tp, tp_op = tf.metrics.true_positives(tarT, probT)
        #     fn, fn_op = tf.metrics.false_negatives(tarT, probT)
        #     tn, tn_op = tf.metrics.true_negatives(tarT, probT)
        #     iou_target = tn / (tn + fp + fn + 1e-8)
        #     iou_back = tp / (tp + fp + fn + 1e-8)
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

        # summa_tp = tf.summary.scalar('tp_BEP_cal', tp_BEP_cal)
        # summa_tn = tf.summary.scalar('tn_BEP_cal', tn_BEP_cal)
        # summa_fp = tf.summary.scalar('fp_BEP_cal', fp_BEP_cal)
        # summa_fn = tf.summary.scalar('fn_BEP_cal', fn_BEP_cal)
        # summa_preci = tf.summary.scalar('precision_BEP_cal', precision_BEP_cal)
        # summa_recal = tf.summary.scalar('recall_BEP_cal', recall_BEP_cal)
        # summa_f1 = tf.summary.scalar('f1_BEP_cal', f1_BEP_cal)
        # summa_A = tf.summary.scalar('score_A_BEP_cal', score_A_BEP_cal)

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
                self.train_gen = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss_gen_total,
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
            sum_loss_dis_f_forD = np.float32(0)
            sum_loss_dis_f_forR = np.float32(0)
            sum_loss_dis_r_forD = np.float32(0)
            sum_loss_reconst = np.float32(0)
            sum_loss_dis_total = np.float32(0)
            sum_loss_ref_total = np.float32(0)

            len_data_syn, len_data_real = self.make_datasets.make_data_for_1_epoch()

            for i in range(0, len_data_syn, self.batch_size):
                if i % (self.batch_size * 100) == 0:
                    print("i = ", i)
                syns_np, segs_np, depths_np, reals_np = self.make_datasets.get_data_for_1_batch(i, self.batch_size)
                tar_1 = self.make_datasets.make_target_1_0(1.0, len(syns_np), self.img_width//8, self.img_height//8) #1 ->
                tar_0 = self.make_datasets.make_target_1_0(0.0, len(syns_np), self.img_width//8, self.img_height//8) #0 ->
                #train discriminator
                # g_out_ = self.sess.run(self.g_out, feed_dict={self.x_s:syns_np, self.x_r:reals_np, self.seg:segs_np, self.pi:depths_np,
                #                 self.tar_d_f:tar_1, self.is_training:True, self.keep_prob:self.keep_prob_rate})
                # print("g_out_.shape, ", g_out_.shape)
                self.sess.run(self.train_dis, feed_dict={self.x_s:syns_np, self.x_r:reals_np,
                                self.tar_d_r:tar_1, self.tar_d_f:tar_0, self.is_training:True, self.keep_prob:self.keep_prob_rate})
                self.sess.run(self.train_tas, feed_dict={self.x_s:syns_np, self.seg:segs_np,
                                                         self.is_training:True, self.keep_prob:self.keep_prob_rate})
                self.sess.run(self.train_pri, feed_dict={self.x_s:syns_np, self.pi:depths_np,
                                                         self.is_training:True, self.keep_prob:self.keep_prob_rate})
                self.sess.run(self.train_gen, feed_dict={self.x_s:syns_np, self.x_r:reals_np, self.seg:segs_np, self.pi:depths_np,
                                self.tar_d_f:tar_1, self.is_training:True, self.keep_prob:self.keep_prob_rate})
                # sess.run(train_dec_opt, feed_dict={z_:z, x_: img_batch, d_dis_f_: tar_g_1, is_training_:True})
                #train encoder
                # sess.run(train_enc, feed_dict={x_:img_batch, d_dis_r_: tar_g_0, is_training_:True, z_:z})
                # sess.run(train_enc_opt, feed_dict={x_:img_batch, d_dis_r_: tar_g_0, is_training_:True})

                # loss for discriminator
                loss_dis_total_, loss_dis_r_forD_, loss_dis_f_forD_ = self.sess.run([self.loss_dis_total, self.loss_dis_r_forD, self.loss_dis_f_forD],
                                                                     feed_dict={self.back:back_batch, self.anno:anno_batch,
                                self.real_img:real_img_batch, self.is_training:False, self.keep_prob:1.0})

                #loss for decoder
                loss_ref_total_, loss_dis_f_forR_, loss_reconst_ = self.sess.run([self.loss_ref_total, self.loss_dis_f_forR, self.loss_reconst],
                                                                                 feed_dict={self.back:back_batch, self.anno:anno_batch,
                                self.is_training:False, self.keep_prob:1.0})

                # #loss for encoder
                # loss_enc_total_ = sess.run(loss_enc_total, feed_dict={x_: img_batch, d_dis_r_: tar_g_0, is_training_:False})

                #for tensorboard
                merged_ = self.sess.run(self.merged, feed_dict={self.back:back_batch, self.anno:anno_batch,
                                self.real_img:real_img_batch, self.is_training:True, self.keep_prob:self.keep_prob_rate})

                self.summary_writer.add_summary(merged_, epoch)

                sum_loss_dis_f_forD += loss_dis_f_forD_ * len(back_batch)
                sum_loss_dis_r_forD += loss_dis_r_forD_ * len(back_batch)
                sum_loss_dis_f_forR += loss_dis_f_forR_ * len(back_batch)
                sum_loss_reconst += loss_reconst_ * len(back_batch)
                sum_loss_dis_total += loss_dis_total_ * len(back_batch)
                sum_loss_ref_total += loss_ref_total_ * len(back_batch)

                # if i % SAVE_MODEL_ITERATE_SPAN == 0 and i != 0:
                #     saver2 = tf.train.Saver()
                #     _ = saver2.save(sess, './out_models/model_' + LOGFILE_NAME + '_' + str(epoch) + '_' + str(i) +  '.ckpt')
            dif_sec = time.time() - start
            hour = int(dif_sec // 3600)
            min = int((dif_sec - hour * 3600) // 60)
            sec = int(dif_sec - hour * 3600 - min * 60)
            print("----------------------------------------------------------------------")
            print(epoch, ", total time: {}hour, {}min, {}sec".format(hour, min, sec))
            print("epoch = {:}, Refiner Total Loss = {:.4f}, Discriminator Total Loss = {:.4f}".format(
                epoch, sum_loss_ref_total / len_data, sum_loss_dis_total / len_data))
            print("Discriminator Real Loss = {:.4f}, Discriminator Refined Loss = {:.4f}".format(
                sum_loss_dis_r_forD / len_data, sum_loss_dis_r_forD / len_data))
            print("Refiner Adversarial Loss = {:.4f}, Refiner Reconstruction Loss = {:.4f}".format(
                sum_loss_dis_f_forR / len_data, sum_loss_reconst / len_data))
            '''
            if epoch % VALID_SPAN == 0:
                print("validation phase")
                score_A_np = np.zeros((0, 2), dtype=np.float32)
                score_A_pred = np.zeros((0, 1), dtype=np.float32)
                score_A_tar = np.zeros((0, 1), dtype=np.float32)
                val_data_num = len(make_datasets.test_file_tar_list)
                for i in range(0, val_data_num, BATCH_SIZE):
                    img_batch, tars_batch = make_datasets.get_valid_data_for_1_batch(i, BATCH_SIZE)
                    score_A_ = sess.run(score_A, feed_dict={x_:img_batch, is_training_:False})
                    score_A_re = np.reshape(score_A_, (-1, 1))
                    tars_batch_re = np.reshape(tars_batch, (-1, 1))

                    score_A_np_tmp = np.concatenate((score_A_re, tars_batch_re), axis=1)
                    score_A_np = np.concatenate((score_A_np, score_A_np_tmp), axis=0) # score_A_np = [[1.8, 0.], [0.8, 1.], ...]

                    score_A_pred = np.concatenate((score_A_pred, score_A_re), axis=0)
                    score_A_tar = np.concatenate((score_A_tar, tars_batch_re), axis=0)

                score_A_pred_max = np.max(score_A_pred)
                score_A_pred_min = np.min(score_A_pred)
                score_A_pred_norm = (score_A_pred - score_A_pred_min) / ((score_A_pred_max - score_A_pred_min + 1e-8))
                auc_, update_op_auc_ = sess.run([auc, update_op_auc], feed_dict={score_A_pred_:score_A_pred_norm,
                                                                                 score_A_tar_:score_A_tar, is_training_:False})
                print("auc_, ", auc_)
                # print("update_op_auc_, ", update_op_auc_)

                tp_BEP, fp_BEP, tn_BEP, fn_BEP, precision_BEP, recall_BEP, f1_BEP, score_A_BEP = Utility.compute_precision_recall(score_A_np, BEFORE_BREAK_EVEN_POINTS) #standardize_flag
                auc_log = Utility.make_ROC_graph(score_A_np, 'out_graph/' + LOGFILE_NAME, epoch)
                BEFORE_BREAK_EVEN_POINTS = (recall_BEP, precision_BEP, f1_BEP)
                Utility.save_histogram_of_norm_abnorm_score(score_A_np, LOGFILE_NAME, epoch, standardize_flag=STANDARDIZE_FLAG)
                # print("precision:{:.4f}".format(precision_BEP))
                print("at break even points, tp:{}, fp:{}, tn:{}, fn:{}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}, AUC:{:.4f}"
                    .format(tp_BEP, fp_BEP, tn_BEP, fn_BEP, precision_BEP, recall_BEP, f1_BEP, auc_log))
                # log_list.append([epoch, auc_log, tp, fp, tn, fn, precision, recall, threshold])
                log_list= [epoch, auc_log, tp_BEP, fp_BEP, tn_BEP, fn_BEP, precision_BEP, recall_BEP, f1_BEP, score_A_BEP]
                Utility.save_1row_to_csv(log_list, 'log/' + LOGFILE_NAME + '_auc.csv')
                img_batch_ok, _ = make_datasets.get_valid_data_for_1_batch(0, 10)
                img_batch_ng, _ = make_datasets.get_valid_data_for_1_batch(val_data_num - 11, 10)

                x_z_x_ok = sess.run(x_z_x, feed_dict={x_:img_batch_ok, is_training_:False})
                x_z_x_ng = sess.run(x_z_x, feed_dict={x_:img_batch_ng, is_training_:False})

                Utility.make_output_img(img_batch_ng, img_batch_ok, x_z_x_ng, x_z_x_ok, epoch, LOGFILE_NAME, OUT_IMG_DIR)
                print("tp_BEP, ", tp_BEP)
                summa_result = sess.run([summa_tp, summa_tn, summa_fp, summa_fn, summa_preci, summa_recal, summa_f1, summa_A],
                                        feed_dict={tp_BEP_:tp_BEP, tn_BEP_:tn_BEP, fp_BEP_:fp_BEP, fn_BEP_:fn_BEP,
                            precision_BEP_:precision_BEP, recall_BEP_:recall_BEP, f1_BEP_:f1_BEP, score_A_BEP_:score_A_BEP})
                for j in range(len(summa_result)):
                    summary_writer.add_summary(summa_result[j], epoch)
            '''
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
        parser.add_argument('--save_model_span', '-ss', type=int, default=10, help='span of saving model')
        parser.add_argument('--base_channel', '-bc', type=int, default=8, help='number of base channel')

        return parser.parse_args()
    args = parser()

    main_process = MainProcess(batch_size=args.batch_size, log_file_name=args.log_file_name, epoch=args.epoch,
                               syn_dir_name=args.syn_dir_name, real_train_dir_name=args.real_train_dir_name,
                               real_val_dir_name=args.real_val_dir_name, syn_seg_dir_name=args.syn_seg_dir_name,
                               real_seg_dir_name=args.real_seg_dir_name, depth_dir_name=args.depth_dir_name,
                               valid_span=args.valid_span, restored_model_name=args.restore_model_name,
                               save_model_span=args.save_model_span, base_channel=args.base_channel, path_to_vgg19=args.path_to_vgg)
    # batch_size = 8, log_file_name = 'log01', epoch = 100,
    # syn_dir_name = '', real_train_dir_name = '', real_val_dir_name = '', syn_seg_dir_name = '', real_seg_dir_name = '',
    # depth_dir_name = '', valid_span = 1, restored_model_name = '', save_model_span = 10, base_channel = 16
    # if args.predict_no_anno_phase:
    #     main_process.predict_no_anno_Model()
    #
    # # elif args.predict_phase:
    # #     main_process.predict()
    #
    # elif args.predict_no_anno_2size_phase:
    #     main_process.predict_no_anno_2size()
    #
    # elif args.evaluate_files_phase:
    #     main_process.evaluate_specified_files()
    #
    # else:
    #     main_process.trainModel()
    main_process.train()


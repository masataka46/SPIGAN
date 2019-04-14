import numpy as np
import os
import tensorflow as tf
import utility as Utility
import argparse
from model_GAN import SPIGAN as Model
from make_datasets import Make_datasets_gyoza as Make_datasets
import time

class MainProcess(object):
    def __init__(self, batch_size=8, log_file_name='log01', epoch=100, base_channel=16, noise_unit_num=200,
                               train_dir_name='', test_dir_name='', valid_span=1, restored_model_name='',
                            save_model_span=10):
        #global variants
        self.batch_size = batch_size
        self.logfile_name = log_file_name
        self.epoch = epoch
        self.train_dir_name = train_dir_name
        self.test_dir_name = test_dir_name
        self.img_width = 128
        self.img_height = 128
        self.img_width_be_crop = 128
        self.img_height_be_crop = 128
        self.img_channel = 3
        self.anno_channel = 4
        self.base_channel = base_channel
        self.noise_unit_num = noise_unit_num
        # NOISE_MEAN = 0.0
        # NOISE_STDDEV = 1.0
        self.test_data_sample = 5 * 5
        self.l2_norm = 0.001
        self.keep_prob_rate = 0.5
        self.seed = 1234
        # SCORE_ALPHA = 0.9 # using for cost function
        self.crop_flag = False
        # STANDARDIZE_FLAG = args.standardize_flag
        self.valid_span = valid_span
        # EXPORT_FLAG = args.export_flag
        # COMPUTE_SCORE_FLAG = args.compute_score_flag
        # PREDICT_WITH_THRESHOLD_FLAG = args.predict_with_threshold_flag
        # SCORE_A_THRESHOLD = args.score_A_threshold
        np.random.seed(seed=self.seed)
        self.board_dir_name = 'tensorboard/' + self.logfile_name
        self.out_img_dir = 'out_images' #output image file
        self.out_model_dir = 'out_models' #output model file
        # CYCLE_LAMBDA = 1.0
        # AUC_COORD_NUM = 200
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

        self.model = Model(self.noise_unit_num, self.img_channel, self.anno_channel, self.seed, self.base_channel, self.keep_prob_rate)


        self.make_datasets = Make_datasets(self.train_dir_name, self.test_dir_name, self.img_width, self.img_height,
                                  self.img_width_be_crop, self.img_height_be_crop, self.seed, self.crop_flag)

        self.back = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channel], name='back')  # back image
        self.anno = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.anno_channel], name='anno')  # annotation image
        self.real_img = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channel], name='real_image')  # real image

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

        with tf.variable_scope('refiner_model'):
            self.rgb_out, self.alpha_out = self.model.refiner(self.back, self.anno, reuse=False, is_training=self.is_training)

        # with tf.variable_scope('decoder_model'):
        #     self.x_dec = self.model.decoder(self.z, reuse=False, is_training=self.is_training)
        #     self.x_z_x = self.model.decoder(self.z_enc, reuse=True, is_training=self.is_training) # for cycle consistency

        with tf.variable_scope('paste_back_and_anno'):
            self.refined_img = self.rgb_out * self.alpha_out + self.back * (1. - self.alpha_out)

        with tf.variable_scope('discriminator_model'):
            #stream around discriminator
            self.drop6_r, self.sigmoid_r = self.model.discriminator(self.real_img, reuse=False, is_training=self.is_training, keep_prob=self.keep_prob) #real pair
            self.drop6_f, self.sigmoid_f = self.model.discriminator(self.refined_img, reuse=True, is_training=self.is_training, keep_prob=self.keep_prob) #real pair
            # self.drop6_re, self.sigmoid_re = self.model.discriminator(self.x_z_x, self.z_enc, reuse=True, is_training=self.is_training) #fake pair

        with tf.name_scope("loss"):
            #adversarial loss
            self.loss_dis_r_forD = - tf.reduce_mean(tf.log(tf.clip_by_value(1. - self.sigmoid_r, 1e-10, 1.0)), name='Loss_dis_refine') #loss related to real for Discriminator
            self.loss_dis_f_forD = - tf.reduce_mean(tf.log(tf.clip_by_value(self.sigmoid_f, 1e-10, 1.0)), name='Loss_dis_real') #loss related to refined for Discriminator
            self.loss_dis_f_forR = - tf.reduce_mean(tf.log(tf.clip_by_value(1. - self.sigmoid_f, 1e-10, 1.0)), name='Loss_dis_refine') #loss related to refined for Refiner
            #regression loss
            self.loss_reconst = tf.reduce_mean(tf.abs(self.rgb_out - self.anno))

            #total loss
            self.loss_dis_total = self.loss_dis_f_forD + self.loss_dis_r_forD
            self.loss_ref_total = self.loss_dis_f_forR + self.reconst_lambda * self.loss_reconst
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
        tf.summary.scalar('self.loss_dis_r_forD', self.loss_dis_r_forD)
        tf.summary.scalar('self.loss_dis_f_forD', self.loss_dis_f_forD)
        tf.summary.scalar('self.loss_dis_f_forR', self.loss_dis_f_forR)
        tf.summary.scalar('self.loss_regression', self.loss_reconst)
        tf.summary.scalar('self.loss_dis_total', self.loss_dis_total)
        tf.summary.scalar('self.loss_ref_total', self.loss_ref_total)
        # tf.summary.scalar('loss_enc_total', loss_enc_total)
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
            ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="refiner")
            # enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
            dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
            # update_ops_dec = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="decoder")
            # update_ops_enc = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="encoder")
            # update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.name_scope("train"):
            # optimizer_dis = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5)
            # with tf.control_dependencies(update_ops_dis):
            #     train_dis = optimizer_dis.minimize(loss_dis_total, var_list=dis_vars, name='Adam_dis')
            # optimizer_dec = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
            # with tf.control_dependencies(update_ops_dec):
            #     train_dec = optimizer_dec.minimize(loss_dec_total, var_list=dec_vars, name='Adam_dec')
            # # train_dec = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_dec_total, var_list=dec_vars
            # #                                                                             , name='Adam_dec')
            # optimizer_enc = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
            # with tf.control_dependencies(update_ops_enc):
            #     train_enc = optimizer_enc.minimize(loss_enc_total, var_list=enc_vars, name='Adam_enc')
            # # train_enc = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss_enc_total, var_list=enc_vars
            # #                                                                             , name='Adam_enc')
            with tf.control_dependencies(update_ops):
                self.train_dis = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(self.loss_dis_total, var_list=dis_vars, name='Adam_dis')
                self.train_ref = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss_ref_total, var_list=ref_vars, name='Adam_ref')
                # train_enc = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss_enc_total, var_list=enc_vars, name='Adam_enc')
            # train_enc = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss_enc_total, var_list=enc_vars
            #                                                                             , name='Adam_enc')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        self.summary_writer = tf.summary.FileWriter(self.board_dir_name, self.sess.graph)
        # saver
        self.saver = tf.train.Saver()

        if self.restore_model_name != '':
            self.saver.restore(self.sess, self.restore_model_name)
            print("model ", self.restore_model_name, " is restored.")


# elif COMPUTE_SCORE_FLAG: # compute score to decide threshold
#     score_A_np = np.zeros((0), dtype=np.float32)
#     for i in range(0, len(make_datasets.test_ng_file_tar_list), BATCH_SIZE):
#         img_batch, _ = make_datasets.get_valid_data_for_1_batch(i, BATCH_SIZE, only_ng_flag=True)
#         score_A_ = sess.run(score_A, feed_dict={x_: img_batch, is_training_: False})
#         print("score_A_.shape, ", score_A_.shape)
#         print("score_A_, ", score_A_)
#         score_A_np = np.concatenate((score_A_np, score_A_))
#     print("score_A_np.shape, ", score_A_np.shape)
#     print("np.min(score_A_np), ", np.min(score_A_np))
#     print("score_A_np, ", score_A_np)
#     print("average score_A_np, ", np.mean(score_A_np))


    def PREDICT_WITH_THRESHOLD_FLAG(self):
        score_A_np = np.zeros((0, 2), dtype=np.float32)
        score_A_pred = np.zeros((0, 1), dtype=np.float32)
        score_A_tar = np.zeros((0, 1), dtype=np.float32)
        val_data_num = len(make_datasets.test_file_tar_list)
        for i in range(0, val_data_num, BATCH_SIZE):
            img_batch, tars_batch = make_datasets.get_valid_data_for_1_batch(i, BATCH_SIZE)
            score_A_ = sess.run(score_A, feed_dict={x_: img_batch, is_training_: False})
            score_A_re = np.reshape(score_A_, (-1, 1))
            tars_batch_re = np.reshape(tars_batch, (-1, 1))
            score_A_np_tmp = np.concatenate((score_A_re, tars_batch_re), axis=1)
            score_A_np = np.concatenate((score_A_np, score_A_np_tmp), axis=0)  # score_A_np = [[1.8, 0.], [0.8, 1.], ...]
            score_A_pred = np.concatenate((score_A_pred, score_A_re), axis=0)
            score_A_tar = np.concatenate((score_A_tar, tars_batch_re), axis=0)
        score_A_pred_max = np.max(score_A_pred)
        score_A_pred_min = np.min(score_A_pred)
        score_A_pred_norm = (score_A_pred - score_A_pred_min) / ((score_A_pred_max - score_A_pred_min + 1e-8))
        # auc_, update_op_auc_ = sess.run([auc, update_op_auc], feed_dict={score_A_pred_: score_A_pred_norm,
        #                                                                  score_A_tar_: score_A_tar, is_training_: False})
        # print("auc_, ", auc_)
        # print("update_op_auc_, ", update_op_auc_)
        tp, fp, tn, fn, precision, recall, f1, recall_normal, precision_normal, f1_normal = Utility.compute_precision_recall_with_threshold(
            score_A_np, SCORE_A_THRESHOLD)
        auc_log = Utility.make_ROC_graph(score_A_np, 'out_graph/' + LOGFILE_NAME, 0)
        # BEFORE_BREAK_EVEN_POINTS = (recall_BEP, precision_BEP, f1_BEP)
        Utility.save_histogram_of_norm_abnorm_score(score_A_np, LOGFILE_NAME, 0, standardize_flag=STANDARDIZE_FLAG)
        # print("precision:{:.4f}".format(precision_BEP))
        print("at threshold:{}, tp:{}, fp:{}, tn:{}, fn:{}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}, AUC:{:.4f},"
            .format(SCORE_A_THRESHOLD, tp, fp, tn, fn, precision, recall, f1, auc_log))
        print("precision_normal:{:.4f}, recall_normal:{:.4f}, f1_normal:{:.4f}"
            .format(precision_normal, recall_normal, f1_normal))
        log_list = [['tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1', 'threshold', 'AUC', 'precision_normal', 'recall_normal', 'f1_normal']]
        log_list.append([tp, fp, tn, fn, precision, recall, f1, SCORE_A_THRESHOLD, auc_log, precision_normal, recall_normal, f1_normal])
        # print("log_list, ", log_list)
        Utility.save_list_to_csv(log_list, 'log/' + LOGFILE_NAME + '_metrics.csv')


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

            len_data = self.make_datasets.make_data_for_1_epoch()

            for i in range(0, len_data, self.batch_size):
                if i % (self.batch_size * 100) == 0:
                    print("i = ", i)
                back_batch, anno_batch, real_img_batch = self.make_datasets.get_data_for_1_batch(i, self.batch_size)
                # z = self.make_datasets.make_random_z_with_norm(self.NOISE_MEAN, NOISE_STDDEV, len(img_batch), NOISE_UNIT_NUM)
                # tar_1 = self.make_datasets.make_target_1_0(1.0, len(back_batch)) #1 ->
                # tar_0 = self.make_datasets.make_target_1_0(0.0, len(back_batch)) #0 ->
                #train discriminator
                self.sess.run(self.train_dis, feed_dict={self.back:back_batch, self.anno:anno_batch,
                                self.real_img:real_img_batch, self.is_training:True, self.keep_prob:self.keep_prob_rate})
                self.sess.run(self.train_ref, feed_dict={self.back:back_batch, self.anno:anno_batch,
                                self.is_training:True, self.keep_prob:self.keep_prob_rate})
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
        parser.add_argument('--base_channel', '-bc', type=int, default=8, help='number of base channel')
        parser.add_argument('--noise_unit_num', '-nn', type=int, default=500, help='number of noise unit')
        parser.add_argument('--train_dir_name', '-dn', type=str, default='../../Efficient-GAN/train_png_data',
                            help='file name of training data')
        # parser.add_argument('--additional_train_dir_name', '-adn', type=str, default='',
        #                     help='file name of training data2')
        # parser.add_argument('--test_ok_dir_name', '-to', type=str, default='../../Efficient-GAN/test_png_data/ok',
        #                     help='file name of test ok data')
        # parser.add_argument('--additional_test_ok_dir_name', '-ato', type=str, default='',
        #                     help='file name of test ok data2')
        parser.add_argument('--test_dir_name', '-tn', type=str, default='../../Efficient-GAN/test_png_data/ng',
                            help='file name of test data')
        parser.add_argument('--valid_span', '-vs', type=int, default=10, help='validation span')
        # parser.add_argument('--train_flag', '-tf', action="store_false", default=True, help='train or not')
        # parser.add_argument('--export_flag', '-ef', action="store_true", default=False, help='export phase')
        # parser.add_argument('--compute_score_flag', '-cf', action="store_true", default=False,
        #                     help='compute_score_phase')
        # parser.add_argument('--predict_with_threshold_flag', '-pf', action="store_true", default=False,
        #                     help='predict_with_threshold_phase')
        parser.add_argument('--restore_model_name', '-rmn', type=str, default='', help='restored model name')
        parser.add_argument('--save_model_span', '-ss', type=int, default=1000, help='span of saving model')
        # parser.add_argument('--save_model_iterate_span', '-sis', type=int, default=10000,
        #                     help='span of saving model by iteration')

        # parser.add_argument('--ok_ng_same_folder_flag', '-sf', action="store_true", default=False,
        #                     help='the case that ok and ng images are in a same folder')
        # parser.add_argument('--export_output_dir', '-eod', type=str, default='./export', help='export output directory')
        # parser.add_argument('--standardize_flag', '-sdf', action="store_false", default=True, help='in make histogram, '
        #                                                                                            'do standardizing or not')
        # parser.add_argument('--score_A_threshold', '-sat', type=float, default=0.36164582, help='threshold of score A')

        return parser.parse_args()
    args = parser()

    main_process = MainProcess(batch_size=args.batch_size, log_file_name=args.log_file_name, epoch=args.epoch,
                               base_channel=args.base_channel, noise_unit_num=args.noise_unit_num,
                               train_dir_name=args.train_dir_name, test_dir_name=args.test_dir_name, valid_span=args.valid_span,
                               restored_model_name=args.restored_model_name,
                 save_model_span=args.save_model_span)
                               # predict_img_span=args.predict_img_span,
                 #               restart_epoch=args.restart_epoch, predict_image=args.predict_image,
                 # predict_mask_image=args.predict_mask_image, restore_mode=args.restore_mode, save_mode=args.save_mode,
                 # predict_by_spec_thr=args.predict_by_spec_thr, img_size_h=args.img_size_h, img_size_w=args.img_size_w,
                 #               img_width_be_crop=args.img_size_be_crop_w, img_height_be_crop=args.img_size_be_crop_h,
                 # crop_flag=True, val_num=5, flip_flag=True, rotate_flag=False,
                 # mixup_flag=False, mixup_rate=1.0, mixup_alpha=0.4, random_erasing_flag=True, image_2_size_magn=16,
                 # prob_verif_flag=False, learning_rate=args.learning_rate, recall_thr_num=args.recall_thr_num,
                 #               evaluate_files_phase=args.evaluate_files_phase)

    if args.predict_no_anno_phase:
        main_process.predict_no_anno_Model()

    # elif args.predict_phase:
    #     main_process.predict()

    elif args.predict_no_anno_2size_phase:
        main_process.predict_no_anno_2size()

    elif args.evaluate_files_phase:
        main_process.evaluate_specified_files()

    else:
        main_process.trainModel()


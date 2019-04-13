import numpy as np
# import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()
import sklearn.metrics as sm
import csv

def compute_precision_recall(score_A_np, before_even_points):
    # array_ok = np.where(score_A_np[:, 1] == 0.0)
    # array_ng = np.where(score_A_np[:, 1] == 1.0)
    # # print("array_ok, ", array_ok)
    # # print("array_ng, ", array_ng)
    #
    # mean_ok = np.mean((score_A_np[array_ok])[:, 0])
    # mean_ng = np.mean((score_A_np[array_ng])[:, 0])
    # threshold = (mean_ok + mean_ng) / 2.0
    # print("mean_ok, ", mean_ok)
    # print("mean_ng, ", mean_ng)
    # print("threshold, ", threshold)
    argsort = np.argsort(score_A_np, axis=0)[:, 0]
    score_A_np_sort = score_A_np[argsort][::-1]
    value_1_0 = (np.where(score_A_np_sort[:, 1] == 1., 1., 0.)).astype(np.float32)
    sum_1 = np.sum(value_1_0)
    len_s = len(score_A_np)
    sum_0 = len_s - sum_1
    tp = np.cumsum(value_1_0).astype(np.float32)
    index = np.arange(1, len_s + 1, 1).astype(np.float32)
    fp = index - tp
    fn = sum_1 - tp
    tn = sum_0 - fp
    # print("fp, ", fp)
    # print("fn, ", fn)
    # print("tn, ", tn)
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    pre_rec_rate = precision / (recall + 1e-8)
    pre_rec_rate_m1 = pre_rec_rate - 1.
    pre_rec_rate_m1_devi1 = np.concatenate((pre_rec_rate_m1[1:], np.array([0], dtype=pre_rec_rate.dtype)), axis=0)
    discriminator_m_p = pre_rec_rate_m1 * pre_rec_rate_m1_devi1
    break_even_points = np.where(discriminator_m_p < 1e-5)
    # print("break_even_points, ", break_even_points)
    # print("break_even_points[0], ", break_even_points[0][0])

    if len(break_even_points[0]) != 0:
        recall_BEP = recall[break_even_points[0][0]]
        precision_BEP = precision[break_even_points[0][0]]
        f1_BEP = (2 * recall_BEP * precision_BEP) / (recall_BEP + precision_BEP + 1e-8)
        tp_BEP = tp[break_even_points[0][0]]
        fp_BEP = fp[break_even_points[0][0]]
        tn_BEP = tn[break_even_points[0][0]]
        fn_BEP = fn[break_even_points[0][0]]
        score_A_BEP = score_A_np_sort[break_even_points[0][0]][0]
    else:
        recall_BEP = before_even_points[0]
        precision_BEP = before_even_points[1]
        f1_BEP = before_even_points[2]
        tp_BEP = before_even_points[3]
        fp_BEP = before_even_points[4]
        tn_BEP = before_even_points[5]
        fn_BEP = before_even_points[6]
        score_A_BEP = before_even_points[7]
    # array_upper = score_A_np[:, 0] >= threshold
    # array_lower = score_A_np[:, 0] < threshold
    # # print("array_upper, ", array_upper)
    # # print("array_lower, ", array_lower)
    # # print("np.sum(array_upper.astype(np.float32)), ", np.sum(array_upper.astype(np.float32)))
    # # print("np.sum(array_lower.astype(np.float32)), ", np.sum(array_lower.astype(np.float32)))
    # array_ok_tf = score_A_np[:, 1] == 0.0
    # array_ng_tf = score_A_np[:, 1] == 1.0
    # # print("np.sum(array_ok_tf.astype(np.float32)), ", np.sum(array_ok_tf.astype(np.float32)))
    # # print("np.sum(array_ng_tf.astype(np.float32)), ", np.sum(array_ng_tf.astype(np.float32)))
    #
    # tn = np.sum(np.equal(array_lower, array_ok_tf).astype(np.int32))
    # tp = np.sum(np.equal(array_upper, array_ng_tf).astype(np.int32))
    # fp = np.sum(np.equal(array_upper, array_ok_tf).astype(np.int32))
    # fn = np.sum(np.equal(array_lower, array_ng_tf).astype(np.int32))
    # print("tp, tn, fp, fn, ", tp, tn, fp, fn)
    # print("precision_BEP.shape, ", precision_BEP.shape)
    # print("tp_BEP, ", tp_BEP)
    return tp_BEP, fp_BEP, tn_BEP, fn_BEP, precision_BEP, recall_BEP, f1_BEP, score_A_BEP


def compute_precision_recall_with_threshold(score_A_np, threshold):
    argsort = np.argsort(score_A_np, axis=0)[:, 0]
    score_A_np_sort = score_A_np[argsort][::-1]
    # print("score_A_np_sort, ", score_A_np_sort)
    positive_num = np.sum(score_A_np_sort[:,0] > threshold)
    # print("positive_num, ", positive_num)
    positive_np = score_A_np_sort[:positive_num]
    # print("positive_np, ", positive_np)
    negative_np = score_A_np_sort[positive_num:]
    # print("negative_np, ", negative_np)
    tp = np.sum(positive_np[:,1])
    # print("tp, ", tp)
    fp = len(positive_np) - tp
    # print("fp, ", fp)
    fn = np.sum(negative_np[:, 1])
    # print("fn, ", fn)
    tn = len(negative_np) - fn
    # print("tn, ", tn)
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * (recall * precision) / (recall + precision + 1e-8)
    recall_normal = tn / (tn + fp + 1e-8)
    precision_normal = tn / (tn + fn + 1e-8)
    f1_normal = 2 * (recall_normal * precision_normal) / (recall_normal + precision_normal)

    return tp, fp, tn, fn, precision, recall, f1, recall_normal, precision_normal, f1_normal



def save_graph(x, y, filename, epoch):
    plt.plot(x, y)
    plt.title('ROC curve ' + filename + ' epoch:' + str(epoch))
    # x axis label
    plt.xlabel("FP / (FP + TN)")
    # y axis label
    plt.ylabel("TP / (TP + FN)")
    # save
    plt.savefig(filename + '_ROC_curve_epoch' + str(epoch) +'.png')
    plt.close()


def make_ROC_graph(score_A_np, filename, epoch):
    argsort = np.argsort(score_A_np, axis=0)[:, 0]
    # print("argsort, ", argsort)
    score_A_np_sort = score_A_np[argsort][::-1]
    # print("score_A_np_sort, ", score_A_np_sort)
    value_1_0 = (np.where(score_A_np_sort[:, 1] == 1., 1., 0.)).astype(np.float32)
    # print("value_1_0, ", value_1_0)
    # score_A_np_sort_0_1 = np.concatenate((score_A_np_sort, value_1_0), axis=1)
    sum_1 = np.sum(value_1_0)
    # print("sum_1, ", sum_1)
    len_s = len(score_A_np)
    # print("len_s, ", len_s)
    sum_0 = len_s - sum_1
    # print("sum_0, ", sum_0)
    tp = np.cumsum(value_1_0).astype(np.float32)
    # print("tp, ", tp)
    index = np.arange(1, len_s + 1, 1).astype(np.float32)
    # print("index, ", index)
    fp = index - tp
    fn = sum_1 - tp
    tn = sum_0 - fp
    # print("fp, ", fp)
    # print("fn, ", fn)
    # print("tn, ", tn)
    tp_ratio = tp / (tp + fn + 1e-8)
    fp_ratio = fp / (fp + tn + 1e-8)
    # print("tp_ratio, ", tp_ratio)
    # print("fp_ratio, ", fp_ratio)
    save_graph(fp_ratio, tp_ratio, filename, epoch)
    auc = sm.auc(fp_ratio, tp_ratio)

    return auc


def unnorm_img(img_np):
    img_np_255 = (img_np + 1.0) * 127.5
    img_np_255_mod1 = np.maximum(img_np_255, 0)
    img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
    img_np_uint8 = img_np_255_mod1.astype(np.uint8)
    return img_np_uint8


def convert_np2pil(images_01):
    list_images_PIL = []
    for num, images_01_1 in enumerate(images_01):
        # img_255_tile = np.tile(images_255_1, (1, 1, 3))
        # print("images_255.shape, ", images_255.shape)
        images_255_1 = (images_01_1 * 255.).astype(np.uint8)
        images_255_1 = np.maximum(images_255_1, 0)
        images_255_1 = np.minimum(images_255_1, 255)

        image_1_PIL = Image.fromarray(images_255_1)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL
    # 5->ng, 7->ok

def make_output_img(img_batch_ng, img_batch_ok, x_z_x_ng, x_z_x_ok, epoch, log_file_name, out_img_dir):
    (data_num, img1_h, img1_w, _) = img_batch_ng.shape
    # img_batch_5_unn = np.tile(unnorm_img(img_batch_5), (1, 1, 3))
    # img_batch_7_unn = np.tile(unnorm_img(img_batch_7), (1, 1, 3))
    # x_z_x_5_unn = np.tile(unnorm_img(x_z_x_5), (1, 1, 3))
    # x_z_x_7_unn = np.tile(unnorm_img(x_z_x_7), (1, 1, 3))

    img_batch_ng_01 = (img_batch_ng + 1.) / 2.
    img_batch_ok_01 = (img_batch_ok + 1.) / 2.
    x_z_x_ng_01 = (x_z_x_ng + 1.) / 2.
    x_z_x_ok_01 = (x_z_x_ok + 1.) / 2.

    diff_ng = img_batch_ng_01 - x_z_x_ng_01
    # diff_ng_r = (2.0 * np.maximum(diff_ng, 0.0)) - 1.0 #(0.0, 1.0) -> (-1.0, 1.0)
    # diff_ng_b = (2.0 * np.abs(np.minimum(diff_ng, 0.0))) - 1.0 #(-1.0, 0.0) -> (1.0, 0.0) -> (1.0, -1.0)
    # diff_ng_g = diff_ng_b * 0.0 - 1.0
    # diff_ng_r_unnorm = unnorm_img(diff_ng_r)
    # diff_ng_b_unnorm = unnorm_img(diff_ng_b)
    # diff_ng_g_unnorm = unnorm_img(diff_ng_g)
    # diff_ng_np = np.concatenate((diff_ng_r_unnorm, diff_ng_g_unnorm, diff_ng_b_unnorm), axis=3)
    diff_ng_np = diff_ng / 2.
    
    diff_ok = img_batch_ok_01 - x_z_x_ok_01
    # diff_ok_r = (2.0 * np.maximum(diff_ok, 0.0)) - 1.0 #(0.0, 1.0) -> (-1.0, 1.0)
    # diff_ok_b = (2.0 * np.abs(np.minimum(diff_ok, 0.0))) - 1.0 #(-1.0, 0.0) -> (1.0, 0.0) -> (1.0, -1.0)
    # diff_ok_g = diff_ok_b * 0.0 - 1.0
    # diff_ok_r_unnorm = unnorm_img(diff_ok_r)
    # diff_ok_b_unnorm = unnorm_img(diff_ok_b)
    # diff_ok_g_unnorm = unnorm_img(diff_ok_g)
    # diff_ok_np = np.concatenate((diff_ok_r_unnorm, diff_ok_g_unnorm, diff_ok_b_unnorm), axis=3)
    diff_ok_np = diff_ok / 2.

    img_batch_ng_PIL = convert_np2pil(img_batch_ng_01)
    img_batch_ok_PIL = convert_np2pil(img_batch_ok_01)
    x_z_x_ng_PIL = convert_np2pil(x_z_x_ng_01)
    x_z_x_ok_PIL = convert_np2pil(x_z_x_ok_01)
    diff_ng_PIL = convert_np2pil(diff_ng_np)
    diff_ok_PIL = convert_np2pil(diff_ok_np)

    wide_image_np = np.ones(((img1_h + 1) * data_num - 1, (img1_w + 1) * 6 - 1, 3), dtype=np.uint8) * 255
    wide_image_PIL = Image.fromarray(wide_image_np)
    for num, (ori_ng, ori_ok, xzx5, xzx7, diff5, diff7) in enumerate(zip(img_batch_ng_PIL, img_batch_ok_PIL ,x_z_x_ng_PIL, x_z_x_ok_PIL, diff_ng_PIL, diff_ok_PIL)):
        wide_image_PIL.paste(ori_ng, (0, num * (img1_h + 1)))
        wide_image_PIL.paste(xzx5, (img1_w + 1, num * (img1_h + 1)))
        wide_image_PIL.paste(diff5, ((img1_w + 1) * 2, num * (img1_h + 1)))
        wide_image_PIL.paste(ori_ok, ((img1_w + 1) * 3, num * (img1_h + 1)))
        wide_image_PIL.paste(xzx7, ((img1_w + 1) * 4, num * (img1_h + 1)))
        wide_image_PIL.paste(diff7, ((img1_w + 1) * 5, num * (img1_h + 1)))

    wide_image_PIL.save(out_img_dir + "/resultImage_"+ log_file_name + '_' + str(epoch) + ".png")
#test

def save_list_to_csv(list, filename):
    f = open(filename, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(list)
    f.close()

def save_1row_to_csv(list, filename):
    f = open(filename, 'a')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(list)
    f.close()

def save_histogram_of_norm_abnorm_score(score_A_np, filename, epoch, division_num=100, standardize_flag=True):
    if standardize_flag:
        #standardize
        score_A_np_0 = (score_A_np[:,0])
        score_A_0_max = np.max(score_A_np_0)
        score_A_0_min = np.min(score_A_np_0)
        score_A_0_stand = (score_A_np_0 - score_A_0_min) / (score_A_0_max - score_A_0_min + 1e-8)
    else:
        score_A_0_stand = (score_A_np[:, 0])

    tar = score_A_np[:,1]
    tar_rev = 1. - tar

    score_abnorm = score_A_0_stand[np.nonzero(score_A_0_stand * tar)]
    score_norm = score_A_0_stand[np.nonzero(score_A_0_stand * tar_rev)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.hist(score_abnorm, bins=division_num, alpha=0.3, histtype='stepfilled', color='r', label='Abnormal Scores')
    plt.hist(score_norm, bins=division_num, alpha=0.3, histtype='stepfilled', color='b', label='Normal Scores')

    # ax.hist(x, bins=50)
    ax.set_title('Histogram of the Normal and Abnormal Scores for the test data')
    ax.set_xlabel('Scores')
    ax.set_ylabel('Freq')
    ax.legend()

    plt.savefig('out_histogram/histogram_socores_' + filename + '_' + str(epoch) + '.png')
    plt.close()






if __name__ == '__main__':
    pred = np.array([6.1, -2.2, 3.3, 1.2, 2.3, 0.4, 0.2, -1.4, -3.2, -5.1, -3.3], dtype=np.float32).reshape(-1, 1)
    tar = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], dtype=np.float32).reshape(-1, 1)
    score_A_np = np.concatenate([pred, tar], axis=1)
    filename = 'tmp_histogram1.png'
    epoch = 3
    # auc, recall_BEP, precision_BEP, f1_BEP = make_ROC_graph(score_A_np, '', 1)
    # tp, fp, tn, fn, precision, recall, medium = compute_precision_recall(score_A_np)
    compute_precision_recall_with_threshold(score_A_np, threshold=0.5)
    # save_histogram_of_norm_abnorm_score(score_A_np, filename, epoch, 4)
    # print("tp, fp, tn, fn, precision, recall, medium", tp, fp, tn, fn, precision, recall, medium)










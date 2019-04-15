import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image

class Make_dataset():

    def __init__(self, train_dir_name, test_dir_name, img_width, img_height, img_width_be_crop, img_height_be_crop,
                 seed=1234, crop_flag=False):
        # train_dir_name = train_dir_name
        self.img_width = img_width
        self.img_height = img_height
        self.img_w_be_crop = img_width_be_crop
        self.img_h_be_crop = img_height_be_crop
        self.seed = seed
        self.crop_flag = crop_flag
        self.train_dir_name = train_dir_name
        self.test_dir_name = test_dir_name
        self.crop_flag = crop_flag
        # self.ok_ng_same_folda_flag = ok_ng_same_folder_flag
        np.random.seed(self.seed)
        # self.ok_prefix = "OK"
        # self.ng_prefix = "NG"
        # self.ok_test_num = ok_test_num
        print("self.train_dir_name, ", self.train_dir_name)
        print("self.test_dir_name, ", self.test_dir_name)
        print("self.img_width, ", self.img_width)
        print("self.img_height, ", self.img_height)
        print("self.img_w_be_crop, ", self.img_w_be_crop)
        print("self.img_h_be_crop, ", self.img_w_be_crop)
        print("self.seed, ", self.seed)
        print("self.crop_flag, ", crop_flag)
        # print("self.ok_prefix, ", self.ok_prefix)
        # print("self.ng_prefix, ", self.ng_prefix)
        # print("self.ok_test_num, ", self.ok_test_num)
        # if train_flag:
        if ok_ng_same_folder_flag: # distinguish by filename prefix
            self.train_test_dir_name = train_dir_name
            # print("self.train_test_dir_name, ", self.train_test_dir_name)
            file_train_test_list = self.get_file_names(self.train_test_dir_name)
            file_train_test_list = self.select_only_png(file_train_test_list)
            # print("len(file_train_test_list), ", len(file_train_test_list))
            file_train_list, file_test_ok_list, file_test_ng_list = self.divide_to_train_testOk_testNg(
                file_train_test_list, self.ok_prefix, self.ng_prefix, self.ok_test_num)
            # print("len(file_train_list), ", len(file_train_list))
            # print("len(file_test_ok_list), ", len(file_test_ok_list))
            # print("len(file_test_ng_list), ", len(file_test_ng_list))

            self.train_file_tar_list = self.add_target_to_list(file_train_list, 0)
            self.test_ok_file_tar_list = self.add_target_to_list(file_test_ok_list, 0)
            self.test_ng_file_tar_list = self.add_target_to_list(file_test_ng_list, 1)
            print("len(train_file_tar_list), ", len(self.train_file_tar_list))
            print("len(self.test_ok_file_tar_list), ", len(self.test_ok_file_tar_list))
            print("len(self.test_ng_file_tar_list), ", len(self.test_ng_file_tar_list))
            self.test_file_tar_list = self.test_ok_file_tar_list + self.test_ng_file_tar_list
            print("len(self.test_file_tar_list), ", len(self.test_file_tar_list))
        else:
            self.train_dir_name = train_dir_name
            print("self.train_dir, ", self.train_dir_name)
            file_train_list = self.get_file_names(self.train_dir_name)
            file_train_list = self.select_only_png(file_train_list)

            if add_train_dir_name != '':
                self.add_train_dir_name = add_train_dir_name
                print("self.add_train_dir, ", self.add_train_dir_name)
                file_train_list2 = self.get_file_names(self.add_train_dir_name)
                file_train_list2 = self.select_only_png(file_train_list2)
                file_train_list = file_train_list + file_train_list2

            self.train_file_tar_list = self.add_target_to_list(file_train_list, 0)
            random.shuffle(self.train_file_tar_list)
            print("len(train_file_tar_list), ", len(self.train_file_tar_list))

            self.test_ok_dir_name = test_ok_dir_name
            print("self.test_ok_dir_name, ", self.test_ok_dir_name)
            file_test_ok_list = self.get_file_names(self.test_ok_dir_name)
            file_test_ok_list = self.select_only_png(file_test_ok_list)
            file_test_ok_list = self.delete_destroyed_file(file_test_ok_list)
            
            if add_test_ok_dir_name != '':
                self.add_test_ok_dir_name = add_test_ok_dir_name
                print("self.add_test_ok_dir_name, ", self.add_test_ok_dir_name)
                file_test_ok_list2 = self.get_file_names(self.add_test_ok_dir_name)
                file_test_ok_list2 = self.select_only_png(file_test_ok_list2)
                file_test_ok_list2 = self.delete_destroyed_file(file_test_ok_list2)
                file_test_ok_list = file_test_ok_list + file_test_ok_list2
            
            self.test_ok_file_tar_list = self.add_target_to_list(file_test_ok_list, 0)
            print("len(self.test_ok_file_tar_list), ", len(self.test_ok_file_tar_list))

            self.test_ng_dir_name = test_ng_dir_name
            print("self.test_ng_dir_name, ", self.test_ng_dir_name)
            file_test_ng_list = self.get_file_names(self.test_ng_dir_name)
            file_test_ng_list = self.select_only_png(file_test_ng_list)
            self.test_ng_file_tar_list = self.add_target_to_list(file_test_ng_list, 1)
            print("len(self.test_ng_file_tar_list), ", len(self.test_ng_file_tar_list))

            self.test_file_tar_list = self.test_ok_file_tar_list + self.test_ng_file_tar_list
            print("len(self.test_file_tar_list), ", len(self.test_file_tar_list))
            # print("self.test_file_tar_list[-1]['tar'], ", self.test_file_tar_list[-1]['tar'])


    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)
        return target_files


    def select_only_png(self, list):
        list_mod = []
        for y in list:
            file_name, extent = y.rsplit(".", 1)
            if (extent == 'png'):  # only .png
                list_mod.append(y)
        return list_mod

    def delete_destroyed_file(self, list):
        list_mod = []
        print("before delete, len is ", len(list))
        for y in list:
            dir_name, filename = y.rsplit("/", 1)
            if (filename == '_976_8493841.png') or (filename == '_1943_8271175.png'):  # destroyed file
                continue
            list_mod.append(y)
        print("after delete, len is ", len(list_mod))
        return list_mod

    def divide_to_train_testOk_testNg(self, file_list, ok_prefix, ng_prefix, ok_test_num):
        ok_files = []
        ng_files = []
        for num, file_list1 in enumerate(file_list):
            dir_name, file_name = file_list1.rsplit("/", 1)
            ok_ng, else_name = file_name.split("_", 1)
            if ok_ng == ok_prefix:
                ok_files.append(file_list1)
            elif ok_ng == ng_prefix:
                ng_files.append(file_list1)
        print("len(file_list), ", len(file_list))
        print("len(ok_files), ", len(ok_files))
        print("len(ng_files), ", len(ng_files))
        random.shuffle(ok_files)
        ok_test = ok_files[:ok_test_num]
        ok_train = ok_files[ok_test_num:]
        return ok_train, ok_test, ng_files



    def add_target_to_list(self, file_list, tar_num):
        file_name_tar_list = []
        for file_name1 in file_list:
            name_tar1 = {'image': file_name1, 'tar': tar_num}
            file_name_tar_list.append(name_tar1)
        return file_name_tar_list


    def read_data(self, file_tar_list, width, height, width_be_crop, height_be_crop, crop_flag=False):
        tars = []
        images = []
        for num, file_tar1 in enumerate(file_tar_list):
            image = Image.open(file_tar1['image'])
            image = image.resize((width, height))
            image = np.asarray(image, dtype=np.float32)
            images.append(image)
            tar = file_tar1['tar']
            tar = np.asarray(tar, dtype=np.float32)
            # tar = self.convert_int_to_oneHot(tar)
            tars.append(tar)
        images_np = np.asarray(images, dtype=np.float32)
        tar_np = np.asarray(tars, dtype=np.float32)
        return images_np, tar_np

    # def crop_img(self, ori_img, output_img_W, output_img_H, margin_W, margin_H):
    #     cropped_img = ori_img.crop((margin_W, margin_H, margin_W + output_img_W, margin_H + output_img_H))
    #     return cropped_img


    def normalize_data(self, data):
        data0_2 = data / 127.5
        data_norm = data0_2 - 1.0
        # data_norm = (data * 2.0) - 1.0 #applied for tanh
        return data_norm

    def make_data_for_1_epoch(self):
        self.file_list_1_epoch = self.train_file_tar_list
        random.shuffle(self.file_list_1_epoch)
        return len(self.file_list_1_epoch)

    def get_data_for_1_batch(self, i, batchsize):
        filename_batch = self.file_list_1_epoch[i:i + batchsize]
        images, _ = self.read_data(filename_batch, self.img_width, self.img_height, self.img_w_be_crop, self.img_h_be_crop, False)
        images_n = self.normalize_data(images)
        return images_n

    def get_valid_data_for_1_batch(self, i, batchsize, only_ng_flag=False):
        if only_ng_flag:
            filename_batch = self.test_ng_file_tar_list[i:i + batchsize]
        else:
            filename_batch = self.test_file_tar_list[i:i + batchsize]
        images, tars = self.read_data(filename_batch, self.img_width, self.img_height, self.img_w_be_crop, self.img_h_be_crop, False)
        images_n = self.normalize_data(images)
        return images_n, tars

    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        norms = np.random.normal(mean, stddev, (data_num, unit_num))
        # tars = np.zeros((data_num, 1), dtype=np.float32)
        return norms


    def make_target_1_0(self, value, data_num):
        if value == 0.0:
            target = np.zeros((data_num, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, 1), dtype=np.float32)
        else:
            print("target value error")
            target = None
        return target

    def convert_int_to_oneHot(self, tar_int):
        if tar_int == 0:
            tar_oneHot = np.array([1., 0.], dtype=np.float32)
        else: #tar_int == 1
            tar_oneHot = np.array([0., 1.], dtype=np.float32)
        return tar_oneHot


    def read_tfrecord(self, filename, img_h, img_w, img_c, class_num):
        filename1 = tf.placeholder(tf.string)
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.float32)
        label = tf.decode_raw(features['label'], tf.float64)
        # print("image.get_shape().as_list(), ", image.get_shape().as_list())
        # print("label.get_shape().as_list(), ", label.get_shape().as_list())


        # image = tf.reshape(image, [img_h, img_w, img_c])
        # label = tf.reshape(label, [class_num])
        #
        # image, label = tf.train.batch([image, label],
        #                               batch_size=16,
        #                               capacity=500)

        sess_data = tf.Session()
        sess_data.run(tf.local_variables_initializer())
        image_, label_ = sess_data.run([image, label], feed_dict={filename1:filename})
        print("image_.shape, ", image_.shape)
        print("label_.shape, ", label_.shape)

        # return image, label

def check_SYNTHIA_RAND_CITYSCAPES(filename):
    example = next(tf.python_io.tf_record_iterator(filename))
    tf_sample = tf.train.Example.FromString(example)
    print("type(tf_sample), ", type(tf_sample))
    print("tf_sample, ", tf_sample)


def debug_specify_cannot_open_img(dir_name):
    target_files = []
    for root, dirs, files in os.walk(dir_name):
        targets = [os.path.join(root, f) for f in files]
        target_files.extend(targets)
    list_mod = []
    for y in target_files:
        file_name, extent = y.rsplit(".", 1)
        if (extent == 'png'):  # only .png
            list_mod.append(y)
    print("len(list_mod), ", len(list_mod))
    for file1 in list_mod:
        try:
            img = Image.open(file1)
        except:
            print("cannot open file, ", file1)
    return



if __name__ == '__main__':
    #debug
    # FILE_NAME = '../../Efficient-GAN/sample_png_data/'
    FILE_NAME = '../../Efficient-GAN/tmp_debug/'

    img_width = 128
    img_height = 128
    img_width_be_crop = 128
    img_height_be_crop = 128
    dir_name = '../../../hiroki_shimada/master_data/ok/test/20181218_aug'
    debug_specify_cannot_open_img(dir_name)

import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image
import cv2

class Make_dataset():

    def __init__(self, syn_dir_name, real_train_dir_name, real_val_dir_name, syn_seg_dir_name, real_seg_dir_name, depth_dir_name,
                 img_width, img_height, img_width_be_crop_syn, img_width_be_crop_real, img_height_be_crop,
                 seed=1234, crop_flag=True, output_img_num=5):
        '''
        Parsed_CityScape---train---:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,     15, 17, 19, 21,    255]
        GT---parsed_LABELS------:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 19, 21, 22]
        '''
        self.syn_dir_name = syn_dir_name
        self.real_train_dir_name = real_train_dir_name
        self.real_val_dir_name = real_val_dir_name
        self.syn_seg_dir_name = syn_seg_dir_name
        self.real_seg_dir_name = real_seg_dir_name
        self.depth_dir_name = depth_dir_name
        self.img_width = img_width
        self.img_height = img_height
        self.img_w_be_crop_syn = img_width_be_crop_syn
        self.img_w_be_crop_real = img_width_be_crop_real
        self.img_h_be_crop = img_height_be_crop
        self.seed = seed
        self.crop_flag = crop_flag
        self.output_img_num = output_img_num
        np.random.seed(self.seed)
        print("self.syn_dir_name, ", self.syn_dir_name)
        print("self.real_train_dir_name, ", self.real_train_dir_name)
        print("self.real_val_dir_name, ", self.real_val_dir_name)
        print("self.syn_seg_dir_name, ", self.syn_seg_dir_name)
        print("self.real_seg_dir_name, ", self.real_seg_dir_name)
        print("self.depth_dir_name, ", self.depth_dir_name)
        print("self.img_width, ", self.img_width)
        print("self.img_height, ", self.img_height)
        print("self.img_w_be_crop_syn, ", self.img_w_be_crop_syn)
        print("self.img_w_be_crop_real, ", self.img_w_be_crop_real)
        print("self.img_h_be_crop, ", self.img_h_be_crop)
        print("self.seed, ", self.seed)
        print("self.crop_flag, ", crop_flag)

        file_syn_list = self.get_file_names(self.syn_dir_name)
        self.file_syn_list = self.select_only_png(file_syn_list)
        self.file_syn_list_num = len(self.file_syn_list)

        file_real_train_list = self.get_file_names(self.real_train_dir_name)
        self.file_real_train_list = self.select_only_png(file_real_train_list)
        self.file_real_train_list_num = len(self.file_real_train_list)
        
        file_real_val_list = self.get_file_names(self.real_val_dir_name)
        self.file_real_val_list = self.select_only_png(file_real_val_list)
        self.file_real_val_list_num = len(self.file_real_val_list)

        print("self.file_syn_list_num, ", self.file_syn_list_num)
        print("self.file_real_train_list_num, ", self.file_real_train_list_num)
        print("self.file_real_val_list_num, ", self.file_real_val_list_num)

        #for validation
        self.file_syn_list_val = random.sample(self.file_syn_list, self.output_img_num)
        self.file_real_val_list_selected = random.sample(self.file_real_val_list, self.output_img_num)


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

    # def delete_destroyed_file(self, list):
    #     list_mod = []
    #     print("before delete, len is ", len(list))
    #     for y in list:
    #         dir_name, filename = y.rsplit("/", 1)
    #         if (filename == '_976_8493841.png') or (filename == '_1943_8271175.png'):  # destroyed file
    #             continue
    #         list_mod.append(y)
    #     print("after delete, len is ", len(list_mod))
    #     return list_mod

    # def divide_to_train_testOk_testNg(self, file_list, ok_prefix, ng_prefix, ok_test_num):
    #     ok_files = []
    #     ng_files = []
    #     for num, file_list1 in enumerate(file_list):
    #         dir_name, file_name = file_list1.rsplit("/", 1)
    #         ok_ng, else_name = file_name.split("_", 1)
    #         if ok_ng == ok_prefix:
    #             ok_files.append(file_list1)
    #         elif ok_ng == ng_prefix:
    #             ng_files.append(file_list1)
    #     print("len(file_list), ", len(file_list))
    #     print("len(ok_files), ", len(ok_files))
    #     print("len(ng_files), ", len(ng_files))
    #     random.shuffle(ok_files)
    #     ok_test = ok_files[:ok_test_num]
    #     ok_train = ok_files[ok_test_num:]
    #     return ok_train, ok_test, ng_files

    def convert_int(self, seg_np):
        '''
        0:other
        1:sky
        2:building
        3:road
        4:sidewalk
        5:
        '''
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 19, 21, 22] ->
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 16, 18, 13]
        seg_np_mod = np.where(seg_np == 19, 16, seg_np)
        seg_np_mod = np.where(seg_np_mod == 21, 18, seg_np_mod)
        seg_np_mod = np.where(seg_np_mod == 22, 13, seg_np_mod)
        return seg_np_mod

    def convert_int_for_real(self, seg_np):
        #    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,     15, 17, 19, 21,    255] ->

        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,     15, 17, 16, 18,    13]
        seg_np_mod = np.where(seg_np == 19, 16, seg_np)
        seg_np_mod = np.where(seg_np_mod == 21, 18, seg_np_mod)
        seg_np_mod = np.where(seg_np_mod == 255, 13, seg_np_mod)
        return seg_np_mod



    def add_target_to_list(self, file_list, tar_num):
        file_name_tar_list = []
        for file_name1 in file_list:
            name_tar1 = {'image': file_name1, 'tar': tar_num}
            file_name_tar_list.append(name_tar1)
        return file_name_tar_list


    def read_data(self, file_syn_list, file_real_list, width, height, width_be_crop_syn, width_be_crop_real, 
                  height_be_crop, seg_dir, depth_dir, crop_flag=True, real_val_flag=False, real_val_dir=''):
        # print("width_be_crop_syn, ", width_be_crop_syn)
        # print("width_be_crop_real, ", width_be_crop_real)
        # print("height_be_crop, ", height_be_crop)
        # print("width, ", width)
        # print("height, ", height)
        syns, reals, segs, depths, real_segs = [], [], [], [], []
        for num, (file_syn1, file_real1) in enumerate(zip(file_syn_list, file_real_list)):
            syn = Image.open(file_syn1)                        #RGB 760h  x 1280w x 3c -> 380h x 640w x 3c
            syn_np = np.asarray(syn)
            syn_np = cv2.resize(syn_np, (width_be_crop_syn, height_be_crop))

            syn_dir_name, syn_file_name_only = file_syn1.rsplit('/', 1)
            seg = Image.open(seg_dir + syn_file_name_only)  # L   760  x 1280w......19classes -> 380h x 640w
            seg_np = np.asarray(seg)
            seg_np = cv2.resize(seg_np, (width_be_crop_syn, height_be_crop), interpolation=cv2.INTER_NEAREST)

            depth = Image.open(depth_dir + syn_file_name_only)  # RGB 760h  x 1280w x 3c -> 380h x 640w x 3c
            depth_np = np.asarray(depth)
            depth_np = cv2.resize(depth_np, (width_be_crop_syn, height_be_crop), interpolation=cv2.INTER_NEAREST)

            real = Image.open(file_real1)  # RGB 1024h x 2048w x 3c -> 380h x 760w x 3c
            real_np = np.asarray(real)
            real_np = cv2.resize(real_np, (width_be_crop_real, height_be_crop))

            if real_val_flag:
                real_dir_name, real_file_name_only = file_real1.rsplit('/', 1)
                real_seg = Image.open(real_val_dir + real_file_name_only)
                real_seg_np = np.asarray(real_seg)
                real_seg_np = cv2.resize(real_seg_np, (width_be_crop_syn, height_be_crop), interpolation=cv2.INTER_NEAREST)
            else:
                real_seg_np = None

            if crop_flag:
                w_margin_s = np.random.randint(0, width_be_crop_syn - width + 1)
                w_margin_r = np.random.randint(0, width_be_crop_real - width + 1)
                h_margin = np.random.randint(0, height_be_crop - height + 1)

                syn_np = syn_np[h_margin:h_margin + height, w_margin_s:w_margin_s + width, :]
                seg_np = seg_np[h_margin:h_margin + height, w_margin_s:w_margin_s + width]
                depth_np = depth_np[h_margin:h_margin + height, w_margin_s:w_margin_s + width, :]
                real_np = real_np[h_margin:h_margin + height, w_margin_r:w_margin_r + width, :]
            else:
                syn_np = cv2.resize(syn_np, (width, height))
                seg_np = cv2.resize(seg_np, (width, height), interpolation=cv2.INTER_NEAREST)
                depth_np = cv2.resize(depth_np, (width, height), interpolation=cv2.INTER_NEAREST)
                real_np = cv2.resize(real_np, (width, height))

            syn_np = syn_np.astype(np.float32) / 255.
            seg_np = (self.convert_int(seg_np)).astype(np.int32)
            depth_np, _ = np.split(depth_np, [1], axis=2)
            depth_np = depth_np.astype(np.float32) / 255.
            real_np = real_np.astype(np.float32) / 255.

            syns.append(syn_np)
            segs.append(seg_np)
            depths.append(depth_np)
            reals.append(real_np)
            if real_val_flag:
                real_seg_np = (self.convert_int_for_real(real_seg_np)).astype(np.int32)
                real_segs.append(real_seg_np)

        if real_val_flag:
            syns_np = np.asarray(syns, dtype=np.float32)
            segs_np = np.asarray(segs, dtype=np.int32)
            depths_np = np.asarray(depths, dtype=np.float32)
            reals_np = np.asarray(reals, dtype=np.float32)
            real_segs_np = np.asarray(real_segs, dtype=np.int32)
            return syns_np, segs_np, depths_np, reals_np, real_segs_np
        else:
            syns_np = np.asarray(syns, dtype=np.float32)
            segs_np = np.asarray(segs, dtype=np.int32)
            depths_np = np.asarray(depths, dtype=np.float32)
            reals_np = np.asarray(reals, dtype=np.float32)
            return syns_np, segs_np, depths_np, reals_np


    def normalize_data(self, data):
        data0_2 = data / 127.5
        data_norm = data0_2 - 1.0
        # data_norm = (data * 2.0) - 1.0 #applied for tanh
        return data_norm

    def make_data_for_1_epoch(self):
        self.file_syn_list_1_epoch = self.file_syn_list
        random.shuffle(self.file_syn_list_1_epoch)
        # self.file_real_train_list_1_epoch = self.file_real_train_list
        # random.shuffle(self.file_real_train_list_1_epoch)
        return len(self.file_syn_list_1_epoch)

    def get_data_for_1_batch(self, i, batchsize):
        filename_syn_batch = self.file_syn_list_1_epoch[i:i + batchsize]
        # i_real = i % self.file_real_train_list_num
        filename_real_batch = random.sample(self.file_real_train_list, len(filename_syn_batch))
        syns_np, segs_np, depths_np, reals_np = self.read_data(filename_syn_batch, filename_real_batch, self.img_width, self.img_height,
                                   self.img_w_be_crop_syn, self.img_w_be_crop_real, self.img_h_be_crop, self.syn_seg_dir_name,
                                   self.depth_dir_name, crop_flag=True)
        # images_n = self.normalize_data(images)
        return syns_np, segs_np, depths_np, reals_np

    def get_valid_data_for_1_batch(self, i, batchsize):
        filename_syn_batch = self.file_syn_list_val
        # i_real = i % self.file_real_train_list_num
        filename_real_batch = random.sample(self.file_real_train_list, len(filename_syn_batch))
        syns_np, segs_np, depths_np, reals_np, real_segs_np = self.read_data(filename_syn_batch, filename_real_batch,
                                                                             self.img_width, self.img_height,
                                                               self.img_w_be_crop_syn, self.img_w_be_crop_real,
                                                               self.img_h_be_crop, self.syn_seg_dir_name,
                                                               self.depth_dir_name, crop_flag=False, real_val_flag=True,
                                                               real_val_dir=self.real_val_dir_name)
        # images_n = self.normalize_data(images)
        return syns_np, segs_np, depths_np, reals_np, real_segs_np

    def get_data_for_1_batch_for_output(self):
        filename_syn_batch = self.file_syn_list_val
        # i_real = i % self.file_real_train_list_num
        filename_real_batch = self.file_real_val_list_selected
        syns_np, segs_np, depths_np, reals_np, real_segs_np = self.read_data(filename_syn_batch, filename_real_batch, self.img_width,
                                                               self.img_height,
                                                               self.img_w_be_crop_syn, self.img_w_be_crop_real,
                                                               self.img_h_be_crop, self.syn_seg_dir_name,
                                                               self.depth_dir_name, crop_flag=False, real_val_flag=True,
                                                               real_val_dir=self.real_val_dir_name)
        # images_n = self.normalize_data(images)
        return syns_np, segs_np, depths_np, reals_np, real_segs_np


    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        norms = np.random.normal(mean, stddev, (data_num, unit_num))
        # tars = np.zeros((data_num, 1), dtype=np.float32)
        return norms


    def make_target_1_0(self, value, data_num, width, height):
        if value == 0.0:
            target = np.zeros((data_num, height, width, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, height, width, 1), dtype=np.float32)
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


    # def read_tfrecord(self, filename, img_h, img_w, img_c, class_num):
    #     filename1 = tf.placeholder(tf.string)
    #     filename_queue = tf.train.string_input_producer([filename])
    #     reader = tf.TFRecordReader()
    #     _, serialized_example = reader.read(filename_queue)
    #
    #     features = tf.parse_single_example(
    #         serialized_example,
    #         features={
    #             'image': tf.FixedLenFeature([], tf.string),
    #             'label': tf.FixedLenFeature([], tf.string)
    #         })
    #
    #     image = tf.decode_raw(features['image'], tf.float32)
    #     label = tf.decode_raw(features['label'], tf.float64)
    #
    #     sess_data = tf.Session()
    #     sess_data.run(tf.local_variables_initializer())
    #     image_, label_ = sess_data.run([image, label], feed_dict={filename1:filename})
    #     print("image_.shape, ", image_.shape)
    #     print("label_.shape, ", label_.shape)
    #
    #     # return image, label

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
    # FILE_NAME = '../../Efficient-GAN/tmp_debug/'

    img_width = 320
    img_height = 320
    img_width_be_crop_syn = 640
    img_width_be_crop_real = 760
    img_height_be_crop = 380
    syn_dir_name = '/media/webfarmer/HDCZ-UT/dataset/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/RGB/'
    real_train_dir_name = '/media/webfarmer/HDCZ-UT/dataset/cityScape/data/leftImg8bit/train/'
    real_val_dir_name = '/media/webfarmer/HDCZ-UT/dataset/cityScape/data/leftImg8bit/val/'
    depth_dir_name = '/media/webfarmer/HDCZ-UT/dataset/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/Depth/Depth/'
    real_seg_dir_name = '/media/webfarmer/HDCZ-UT/dataset/SYNTHIA_RAND_CITYSCAPES/segmentation_annotation/Parsed_CityScape/val/'
    syn_seg_dir_name = '/media/webfarmer/HDCZ-UT/dataset/SYNTHIA_RAND_CITYSCAPES/segmentation_annotation/SYNTHIA/GT/parsed_LABELS/'
    make_dataset = Make_dataset(syn_dir_name, real_train_dir_name, real_val_dir_name, syn_seg_dir_name, real_seg_dir_name, depth_dir_name,
                 img_width, img_height, img_width_be_crop_syn, img_width_be_crop_real, img_height_be_crop)
    len_syn, len_real_tr = make_dataset.make_data_for_1_epoch()
    print("len_syn, ", len_syn)
    print("len_real_tr, ", len_real_tr)
    syns_np, segs_np, depths_np, reals_np = make_dataset.get_data_for_1_batch(0, 10)
    print("syns_np.shape, ", syns_np.shape)
    print("segs_np.shape, ", segs_np.shape)
    print("depths_np.shape, ", depths_np.shape)
    print("reals_np.shape, ", reals_np.shape)
    
    print("syns_np.dtype, ", syns_np.dtype)
    print("segs_np.dtype, ", segs_np.dtype)
    print("depths_np.dtype, ", depths_np.dtype)
    print("reals_np.dtype, ", reals_np.dtype)
    
    print("np.max(syns_np), ", np.max(syns_np))
    print("np.max(segs_np), ", np.max(segs_np))
    print("np.max(depths_np), ", np.max(depths_np))
    print("np.max(reals_np), ", np.max(reals_np))
    
    print("np.min(syns_np), ", np.min(syns_np))
    print("np.min(segs_np), ", np.min(segs_np))
    print("np.min(depths_np), ", np.min(depths_np))
    print("np.min(reals_np), ", np.min(reals_np))

    # debug_specify_cannot_open_img(dir_name)

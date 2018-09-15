# coding: utf-8

import numpy as np
import cv2
import os
import random
from tensorflow.core.framework import summary_pb2


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def random_start(img_cube, image_size):
    h, w = img_cube.shape[1], img_cube.shape[2]
    h_start = np.random.randint(0, h - image_size)
    w_start = np.random.randint(0, w - image_size)
    return h_start, w_start


def get_data_func_train(txt_line, image_size):
    vid_dir, label = txt_line[0], int(txt_line[1])
    pic_list = sorted(os.listdir(vid_dir), key=lambda x: int(x.split('.')[0]))
    pic_path = [os.path.join(vid_dir, path) for path in pic_list]
    imgs = []
    for path in pic_path:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32)
    starts = random_start(imgs, image_size)
    imgs = imgs[:, starts[0]: starts[0] + image_size, starts[1]: starts[1] + image_size, :]
    imgs = imgs / 255. * 2 - 1
    return imgs, label

def get_data_func_val(txt_line, image_size):
    vid_dir, label = txt_line[0], int(txt_line[1])
    pic_list = sorted(os.listdir(vid_dir), key=lambda x: int(x.split('.')[0]))
    pic_path = [os.path.join(vid_dir, path) for path in pic_list]
    imgs = []
    for i in range(0, len(pic_path)):
        img = cv2.imread(pic_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32)
    h, w = img.shape[1], img.shape[2]
    h_start = (h - image_size) / 2
    w_start = (w - image_size) / 2
    imgs = imgs[:, h_start: h_start + image_size, w_start: w_start + image_size, :]
    imgs = imgs / 255. * 2 - 1
    return imgs, label

def shuffle_and_overwrite(file_name):
    content = open(file_name, 'r').readlines()
    random.shuffle(content)
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)

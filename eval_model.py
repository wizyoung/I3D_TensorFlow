# coding: utf-8
import time
import logging
import tensorflow as tf
import sonnet as snt
import numpy as np
import os
import sys
from tqdm import tqdm, trange
from pprint import pprint

import i3d

from utils import get_data_func_val

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_CLASS = 50
EVAL_BATCH_SIZE = 5
NUM_THREADS = 8  # threads of preprocessing
PREFETCH_BUFFER = 5
IMAGE_SIZE = 224
TXT_DATA_DIR = './train&val_txt_folder'
VAL_FILE = 'val.txt'
VAL_LEN = len(open(os.path.join(TXT_DATA_DIR, VAL_FILE), 'r').readlines())

MODEL_PATH = './save/single_gpu/model_step_94000_lr_0.0005062982'

with tf.device('/cpu:0'):
    dataset = tf.data.Dataset.from_tensor_slices([os.path.join(TXT_DATA_DIR, VAL_FILE)])
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        lambda x: tf.data.TextLineDataset(x).map(lambda x: tf.string_split([x], delimiter=' ').values),
        cycle_length=NUM_THREADS, block_length=1))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        lambda x: tuple(tf.py_func(get_data_func_val, [x, IMAGE_SIZE], [tf.float32, tf.int64])), batch_size=EVAL_BATCH_SIZE,
        num_parallel_batches=NUM_THREADS))
    dataset.prefetch(PREFETCH_BUFFER)

    dataset_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    init_op = dataset_iterator.make_initializer(dataset)
    # dataset_iterator = dataset.make_one_shot_iterator()

    batch_vid, batch_label = dataset_iterator.get_next()
    batch_vid.set_shape([None, None, IMAGE_SIZE, IMAGE_SIZE, 3])

with tf.variable_scope('RGB'):
    model = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
    logits, _ = model(inputs=batch_vid, is_training=False, dropout_keep_prob=1.0)
    out = tf.layers.dense(logits, NUM_CLASS, activation=None, use_bias=True)

is_in_top_K = tf.nn.in_top_k(predictions=out, targets=batch_label, k=1)
loss_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_label, logits=out, name='cross_entropy'))

saver_to_restore = tf.train.Saver(reshape=True)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver_to_restore.restore(sess, MODEL_PATH)
    sess.run(init_op)
    #
    iter_num = int(np.ceil(float(VAL_LEN) / EVAL_BATCH_SIZE))

    correct_cnt = 0
    loss_cnt = 0
    wrong_index_list = []
    for i in trange(iter_num):
        _is_in_top_K, _loss_cross_entropy = sess.run([is_in_top_K, loss_cross_entropy])
        correct_cnt += np.sum(_is_in_top_K)
        loss_cnt += _loss_cross_entropy
        wrong_idx = list(np.where(_is_in_top_K == False)[0] + i * EVAL_BATCH_SIZE)
        wrong_index_list.extend(wrong_idx)
    acc = float(correct_cnt) / VAL_LEN
    loss = float(loss_cnt) / iter_num

    sys.stdout.write('\nval_acc:{}\nval_loss:{}\n'.format(acc, loss))
    sys.stdout.flush()

## analyse wrong samples
# import collections
# wrong_dict = collections.OrderedDict()
# val_file = open(os.path.join(TXT_DATA_DIR, VAL_FILE), 'r').readlines()
# for i in wrong_index_list:
#     item = val_file[i].strip().split(' ')
#     key = item[-1]
#     if key not in wrong_dict:
#         wrong_dict[key] = 1
#     else:
#         wrong_dict[key] += 1

# wrong_dict = collections.OrderedDict(sorted(wrong_dict.items(), key=lambda t: int(t[0])))
# for key, value in wrong_dict.items():
#     sys.stdout.write('{}:{}\n'.format(key, value))
#     sys.stdout.flush()


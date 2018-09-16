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

from utils import get_data_func_train, make_summary, shuffle_and_overwrite

#######################################
##### define some constants here #####
#######################################
N_GPU = 4
BATCH_SIZE = 6
EPOCH_NUM = 100
NUM_CLASS = 50
NUM_THREADS = 8  # threads of preprocessing
PREFETCH_BUFFER = 5
OPT_EPSILON = 1.0
TOP_K = 1
L2_PARAM = 1e-7
DROPOUT_KEEP_PRAM = 0.5
IMAGE_SIZE = 224
GLOBAL_STEP_INIT = 0
OPTIMIZER = 'momentum'
LEARNING_RATE_TYPE = 'exponential'
LEARNING_RATE_INIT = 1e-3
LEARNING_RATE_LOWER_BOUND = 1e-5
# only for exponential decay(following two)
LEARNING_DECAY_FACTOR = 0.95
LEARNING_DECAY_FREQ = 4000
# only for piecewise mode(following two)
BOUNDARIES = [10000, 20000, 30000, 40000, 50000]
VALUES = [LEARNING_RATE_INIT, 0.0008, 0.0005, 0.0003, 0.0001, 5e-5]

TXT_DATA_DIR = './train&val_txt_folder'
TRAIN_FILE = 'train.txt'
TRAIN_LEN = len(open(os.path.join(TXT_DATA_DIR, TRAIN_FILE), 'r').readlines())

TENSORBOARD_LOG_DIR = './logs/multi_gpu'
PROGROESS_LOG_FILE = './logs/multi_gpu.log'
SAVE_DIR = './save/multi_gpu'
SHOW_TRAIN_INFO_FREQ = 100
SAVE_FREQ = 1000

if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=PROGROESS_LOG_FILE, filemode='a')

CHECKPOINT_PATH = './checkpoints/rgb_imagenet/model.ckpt'

def config_learning_rate(global_step, batch_num):
    if LEARNING_RATE_TYPE == 'exponential':
        lr_tmp = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step, LEARNING_DECAY_FREQ, LEARNING_DECAY_FACTOR,
                                            staircase=True, name='exponential_learning_rate')
        return tf.maximum(lr_tmp, LEARNING_RATE_LOWER_BOUND)
    elif LEARNING_RATE_TYPE == 'fixed':
        return tf.convert_to_tensor(LEARNING_RATE_INIT, name='fixed_learning_rate')
    elif LEARNING_RATE_TYPE == 'piecewise':
        return tf.train.piecewise_constant(global_step, boundaries=BOUNDARIES, values=VALUES,
                                           name='piecewise_learning_rate')
    else:
        raise ValueError('Unsupported learning rate type!')


def config_optimizer(optimizer_name, learning_rate, epsilon, decay=0.9, momentum=0.9):
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, epsilon=epsilon)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate, epsilon=epsilon)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')


def get_input(num_class):
    '''
        dataset loading using tf.data module
    '''
    train_dataset = tf.data.Dataset.from_tensor_slices([os.path.join(TXT_DATA_DIR, TRAIN_FILE)])
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.apply(tf.contrib.data.parallel_interleave(
        lambda x: tf.data.TextLineDataset(x).map(lambda x: tf.string_split([x], delimiter=' ').values),
        cycle_length=NUM_THREADS, block_length=1))
    train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(
        lambda x: tuple(tf.py_func(get_data_func_train, [x, IMAGE_SIZE], [tf.float32, tf.int64])), batch_size=BATCH_SIZE,
        num_parallel_batches=NUM_THREADS))
    train_dataset.prefetch(PREFETCH_BUFFER)

    dataset_iterator = train_dataset.make_initializable_iterator()
    dataset_init_op = dataset_iterator.initializer
    batch_vid, batch_label = dataset_iterator.get_next()
    batch_vid.set_shape([None, None, IMAGE_SIZE, IMAGE_SIZE, 3])

    return batch_vid, batch_label, dataset_init_op


def get_model_loss(input_x, input_y, train_flag, dropout_flag, scope, reuse=None, num_class=NUM_CLASS, top_k=TOP_K):
    with tf.variable_scope('RGB', reuse=reuse):
        model = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        logits, _ = model(inputs=input_x, is_training=train_flag, dropout_keep_prob=dropout_flag)
        logits_dropout = tf.nn.dropout(logits, keep_prob=dropout_flag)
        out = tf.layers.dense(logits_dropout, num_class, activation=None, use_bias=True)

    is_in_top_K = tf.cast(tf.nn.in_top_k(predictions=out, targets=input_y, k=top_k), tf.float32)

    # prepare l2 loss
    regularization_loss = 0.
    for var in tf.global_variables():
        var_name_type = var.name.split('/')[-1][:-2]
        if var_name_type == 'w' or var_name_type == 'kernel':
            regularization_loss += tf.nn.l2_loss(var)
    regularization_loss = tf.identity(regularization_loss, name='regularization_loss')

    loss_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_y, logits=out, name='cross_entropy'))
    total_loss = tf.add(loss_cross_entropy, L2_PARAM * regularization_loss, name='total_loss')

    tf.summary.scalar('{}/loss_ratio'.format(scope), regularization_loss / loss_cross_entropy)

    return total_loss, loss_cross_entropy, is_in_top_K


def average_gradients(tower_grads):
    average_grads = []

    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]  # var
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main():
    with tf.device('/cpu:0'):
        batch_x, batch_y, _dataset_init_op = get_input(NUM_CLASS)
        train_flag = tf.placeholder(dtype=tf.bool, name='train_flag')
        dropout_flag = tf.placeholder(dtype=tf.float32, name='dropout_flag')

        batch_num = TRAIN_LEN / (BATCH_SIZE * N_GPU)

        global_step = tf.Variable(GLOBAL_STEP_INIT, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        learning_rate = config_learning_rate(global_step, batch_num)
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = config_optimizer(OPTIMIZER, learning_rate, OPT_EPSILON)

        cross_entropy_list = []
        in_top_K_list = []

        reuse_flag = False
        tower_grads = []
        for i in range(N_GPU):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('GPU_{}'.format(i)) as scope:
                    current_loss, _current_cross_entropy, current_in_top_K = get_model_loss(
                        batch_x, batch_y, train_flag, dropout_flag, scope, reuse_flag)
                    reuse_flag = True
                    grads = optimizer.compute_gradients(current_loss)
                    tower_grads.append(grads)
                    cross_entropy_list.append(_current_cross_entropy)
                    in_top_K_list.append(current_in_top_K)

                    # retain the bn ops only from the last tower
                    # as suggested by: https://github.com/tensorflow/models/blob/master/research/inception/inception/inception_train.py#L249
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, name='train_op')

        # maintain a variable map to restore from the ckpt
        variable_map = {}
        for var in tf.global_variables():
            var_name_split = var.name.split('/')
            if var_name_split[1] == 'inception_i3d' and 'dense' not in var.name:
                variable_map[var.name[:-2]] = var

        saver_to_restore = tf.train.Saver(var_list=variable_map, reshape=True)
        saver_to_save = tf.train.Saver(max_to_keep=50)

        # NOTE: optional: check variable names
        # for var in tf.global_variables():
        #     print(var.name)

        # TODO: may var moving average here?

        average_cross_entropy = tf.reduce_mean(cross_entropy_list)
        average_in_top_K = tf.reduce_sum(in_top_K_list)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            shuffle_and_overwrite(os.path.join(TXT_DATA_DIR, TRAIN_FILE))
            sess.run(_dataset_init_op)
            saver_to_restore.restore(sess, CHECKPOINT_PATH)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(TENSORBOARD_LOG_DIR, 'train'), sess.graph)

            sys.stdout.write('\n----------- start to train -----------\n')
            sys.stdout.flush()

            intermediate_train_info = [0., 0.]
            for epoch in range(EPOCH_NUM):

                pbar = tqdm(total=batch_num, desc='Epoch {}'.format(epoch),
                            unit=' batch (batch_size: {} * {} GPUs)'.format(BATCH_SIZE, N_GPU))
                epoch_acc, epoch_loss = 0., 0.
                for i in range(batch_num):
                    _, _loss_cross_entropy, _is_in_top_K, summary, _global_step, lr = sess.run(
                        [train_op, average_cross_entropy, average_in_top_K, merged, global_step, learning_rate],
                        feed_dict={train_flag: True, dropout_flag: DROPOUT_KEEP_PRAM})
                    train_writer.add_summary(summary, global_step=_global_step)

                    intermediate_train_info[0] += _is_in_top_K
                    epoch_acc += _is_in_top_K
                    intermediate_train_info[1] += _loss_cross_entropy
                    epoch_loss += _loss_cross_entropy

                    # intermediate evaluation for the training dataset
                    if _global_step % SHOW_TRAIN_INFO_FREQ == 0:
                        intermediate_train_acc = float(intermediate_train_info[0]) / (SHOW_TRAIN_INFO_FREQ * BATCH_SIZE * N_GPU)
                        intermediate_train_loss = intermediate_train_info[1] / (SHOW_TRAIN_INFO_FREQ)

                        step_log_info = 'global_step:{}, step_train_acc:{:.4f}, step_train_loss:{:4f}, lr:{:.4g}'.format(
                            _global_step, intermediate_train_acc, intermediate_train_loss, lr)
                        sys.stdout.write('\n' + step_log_info + '\n')
                        sys.stdout.flush()
                        logging.info(step_log_info)
                        train_writer.add_summary(
                            make_summary('accumulated_statistics/train_acc', intermediate_train_acc),
                            global_step=_global_step)
                        train_writer.add_summary(
                            make_summary('accumulated_statistics/train_loss', intermediate_train_loss),
                            global_step=_global_step)
                        intermediate_train_info = [0., 0.]

                    # start to save
                    if _global_step % SAVE_FREQ == 0:
                        saver_to_save.save(sess, SAVE_DIR + '/model_step_{}_train_acc_{:.4f}_lr_{:.4g}'.format(
                            _global_step, intermediate_train_acc, lr))

                    pbar.update(1)
                pbar.close()

                epoch_acc = float(epoch_acc) / TRAIN_LEN
                epoch_loss = float(epoch_loss) / batch_num
                log_info = '=====Epoch:{}, whole_train_acc:{:.4f}, whole_train_loss:{:4f}, lr:{:.7g}====='.format(
                    epoch, epoch_acc, epoch_loss, lr)
                logging.info(log_info)
                sys.stdout.write('\n' + log_info + '\n')
                sys.stdout.flush()

                shuffle_and_overwrite(os.path.join(TXT_DATA_DIR, TRAIN_FILE))
                sess.run(_dataset_init_op)
            train_writer.close()

if __name__ == '__main__':
    main()
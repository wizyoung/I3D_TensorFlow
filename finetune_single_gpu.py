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

from utils import get_data_func_train, get_data_func_val, make_summary, shuffle_and_overwrite

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#######################################
##### define some constants here #####
#######################################
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
LEARNING_DECAY_FACTOR = 0.94
LEARNING_DECAY_FREQ = 8000
# only for piecewise mode(following two)
BOUNDARIES = [10000, 20000, 30000, 40000, 50000]
VALUES = [LEARNING_RATE_INIT, 0.0008, 0.0005, 0.0003, 0.0001, 5e-5]

TXT_DATA_DIR = './train&val_txt_folder'
TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'
TRAIN_LEN = len(open(os.path.join(TXT_DATA_DIR, TRAIN_FILE), 'r').readlines())
VAL_LEN = len(open(os.path.join(TXT_DATA_DIR, VAL_FILE), 'r').readlines())

TENSORBOARD_LOG_DIR = './logs/single_gpu' 
PROGROESS_LOG_FILE = './logs/single_gpu.log'
SAVE_DIR = './save/single_gpu'

SHOW_TRAIN_INFO_FREQ = 200
SAVE_FREQ = 1000

if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=PROGROESS_LOG_FILE, filemode='w')

CHECKPOINT_PATH = './checkpoints/rgb_imagenet/model.ckpt'

def config_learning_rate(global_step, batch_num):
    if LEARNING_RATE_TYPE == 'exponential':
        lr_tmp = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step, LEARNING_DECAY_FREQ,
                                            LEARNING_DECAY_FACTOR, staircase=True, name='exponential_learning_rate')
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


def main():
    # manually shuffle the train txt file because tf.data.shuffle is soooo slow!
    shuffle_and_overwrite(os.path.join(TXT_DATA_DIR, TRAIN_FILE))
    # dataset loading using tf.data module
    with tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices([os.path.join(TXT_DATA_DIR, TRAIN_FILE)])
        train_dataset = train_dataset.apply(tf.contrib.data.parallel_interleave(
            lambda x: tf.data.TextLineDataset(x).map(lambda x: tf.string_split([x], delimiter=' ').values),
            cycle_length=NUM_THREADS, block_length=1))
        train_dataset = train_dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda x: tuple(tf.py_func(get_data_func_train, [x, IMAGE_SIZE], [tf.float32, tf.int64])),
                batch_size=BATCH_SIZE, num_parallel_batches=NUM_THREADS))
        train_dataset.prefetch(PREFETCH_BUFFER)

        val_dataset = tf.data.Dataset.from_tensor_slices([os.path.join(TXT_DATA_DIR, VAL_FILE)])
        val_dataset = val_dataset.shuffle(VAL_LEN)
        val_dataset = val_dataset.apply(tf.contrib.data.parallel_interleave(
            lambda x: tf.data.TextLineDataset(x).map(lambda x: tf.string_split([x], delimiter=' ').values),
            cycle_length=NUM_THREADS, block_length=1))
        val_dataset = val_dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda x: tuple(tf.py_func(get_data_func_val, [x, IMAGE_SIZE], [tf.float32, tf.int64])),
                batch_size=BATCH_SIZE, num_parallel_batches=NUM_THREADS))
        val_dataset.prefetch(PREFETCH_BUFFER)

        train_iterator = train_dataset.make_initializable_iterator()
        val_iterator = val_dataset.make_initializable_iterator()

        train_handle = train_iterator.string_handle()
        val_handle = val_iterator.string_handle()
        handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
        dataset_iterator = tf.data.Iterator.from_string_handle(handle_flag, train_dataset.output_types,
                                                               train_dataset.output_shapes)

        batch_vid, batch_label = dataset_iterator.get_next()
        batch_vid.set_shape([None, None, IMAGE_SIZE, IMAGE_SIZE, 3])

    train_flag = tf.placeholder(dtype=tf.bool, name='train_flag')
    dropout_flag = tf.placeholder(dtype=tf.float32, name='dropout_flag')

    # define model here
    with tf.variable_scope('RGB'):
        model = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        logits, _ = model(inputs=batch_vid, is_training=train_flag, dropout_keep_prob=dropout_flag)
        logits_dropout = tf.nn.dropout(logits, keep_prob=dropout_flag)
        out = tf.layers.dense(logits_dropout, NUM_CLASS, activation=None, use_bias=True)

        is_in_top_K = tf.nn.in_top_k(predictions=out, targets=batch_label, k=TOP_K)

        # maintain a variable map to restore from the ckpt
        variable_map = {}
        for var in tf.global_variables():
            var_name_split = var.name.split('/')
            if var_name_split[1] == 'inception_i3d' and 'dense' not in var_name_split[1]:
                variable_map[var.name[:-2]] = var
            if var_name_split[-1][:-2] == 'w' or var_name_split[-1][:-2] == 'kernel':
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(var))

        # optional: print to check the variable names
        # pprint(variable_map)

        regularization_loss = tf.losses.get_regularization_loss(name='regularization_loss')  # sum of l2 loss
        loss_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_label, logits=out, name='cross_entropy'))
        total_loss = tf.add(loss_cross_entropy, L2_PARAM * regularization_loss, name='total_loss')
        tf.summary.scalar('batch_statistics/total_loss', total_loss)
        tf.summary.scalar('batch_statistics/cross_entropy_loss', loss_cross_entropy)
        tf.summary.scalar('batch_statistics/l2_loss', regularization_loss)
        tf.summary.scalar('batch_statistics/loss_ratio', regularization_loss / loss_cross_entropy)

        saver_to_restore = tf.train.Saver(var_list=variable_map, reshape=True)

        batch_num = TRAIN_LEN / BATCH_SIZE

        global_step = tf.Variable(GLOBAL_STEP_INIT, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        learning_rate = config_learning_rate(global_step, batch_num)
        tf.summary.scalar('learning_rate', learning_rate)

        # set dependencies for BN ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = config_optimizer(OPTIMIZER, learning_rate, OPT_EPSILON)
            train_op = optimizer.minimize(total_loss, global_step=global_step)

        # NOTE: if you don't want to save the params of the optimizer into the checkpoint,
        # you can place this line before the `update_ops` line
        saver_to_save = tf.train.Saver(max_to_keep=40)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            train_handle_value, val_handle_value = sess.run([train_handle, val_handle])
            sess.run(train_iterator.initializer)
            saver_to_restore.restore(sess, CHECKPOINT_PATH)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(TENSORBOARD_LOG_DIR, 'train'), sess.graph)

            sys.stdout.write('\n----------- start to train -----------\n')

            intermediate_train_info = [0., 0.]
            for epoch in range(EPOCH_NUM):
                epoch_acc, epoch_loss = 0., 0.
                pbar = tqdm(total=batch_num, desc='Epoch {}'.format(epoch),
                            unit=' batch (batch_size: {})'.format(BATCH_SIZE))
                for i in range(batch_num):
                    _, _loss_cross_entropy, _is_in_top_K, summary, _global_step, lr = sess.run(
                        [train_op, loss_cross_entropy, is_in_top_K, merged, global_step, learning_rate],
                        feed_dict={train_flag: True, dropout_flag: DROPOUT_KEEP_PRAM, handle_flag: train_handle_value})
                    train_writer.add_summary(summary, global_step=_global_step)

                    intermediate_train_info[0] += np.sum(_is_in_top_K)
                    intermediate_train_info[1] += _loss_cross_entropy
                    epoch_acc += np.sum(_is_in_top_K)
                    epoch_loss += _loss_cross_entropy

                    # intermediate evaluation for the training dataset
                    if _global_step % SHOW_TRAIN_INFO_FREQ == 0:
                        intermediate_train_acc = float(intermediate_train_info[0]) / (SHOW_TRAIN_INFO_FREQ * BATCH_SIZE)
                        intermediate_train_loss = intermediate_train_info[1] / SHOW_TRAIN_INFO_FREQ

                        step_log_info = 'Epoch:{}, global_step:{}, step_train_acc:{:.4f}, step_train_loss:{:4f}, lr:{:.7g}'.format(
                            epoch, _global_step, intermediate_train_acc, intermediate_train_loss, lr)
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

                    # start to evaluate
                    if _global_step % SAVE_FREQ == 0:
                        if intermediate_train_acc >= 0.8:
                            saver_to_save.save(sess, SAVE_DIR + '/model_step_{}_lr_{:.7g}'.format(_global_step, lr))

                    pbar.update(1)
                pbar.close()

                # start to validata on the validation dataset
                sess.run(val_iterator.initializer)
                iter_num = int(np.ceil(float(VAL_LEN) / BATCH_SIZE))
                correct_cnt, loss_cnt = 0, 0
                pbar = tqdm(total=iter_num, desc='EVAL train_epoch:{}'.format(epoch),
                            unit=' batch(batch_size={})'.format(BATCH_SIZE))
                for _ in range(iter_num):
                    _is_in_top_K, _loss_cross_entropy = sess.run([is_in_top_K, loss_cross_entropy],
                                                                    feed_dict={handle_flag: val_handle_value,
                                                                            train_flag: False, dropout_flag: 1.0})
                    correct_cnt += np.sum(_is_in_top_K)
                    loss_cnt += _loss_cross_entropy
                    pbar.update(1)
                pbar.close()
                val_acc = float(correct_cnt) / VAL_LEN
                val_loss = float(loss_cnt) / iter_num

                log_info = '==>> Epoch:{}, global_step:{}, val_acc:{:.4f}, val_loss:{:4f}, lr:{:.7g}'.format(
                    epoch, _global_step, val_acc, val_loss, lr)
                logging.info(log_info)
                sys.stdout.write('\n' + log_info + '\n')
                sys.stdout.flush()

                # manually shuffle the data with python for better performance
                shuffle_and_overwrite(os.path.join(TXT_DATA_DIR, TRAIN_FILE))
                sess.run(train_iterator.initializer)

                epoch_acc = float(epoch_acc) / TRAIN_LEN
                epoch_loss = float(epoch_loss) / batch_num
                log_info = '==========Epoch:{}, whole_train_acc:{:.4f}, whole_train_loss:{:4f}, lr:{:.7g}=========='.format(
                    epoch, epoch_acc, epoch_loss, lr)
                logging.info(log_info)
                sys.stdout.write('\n' + log_info + '\n')
                sys.stdout.flush()

        train_writer.close()


if __name__ == '__main__':
    main()

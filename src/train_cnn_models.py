"""Training lenet style CNN on mnist of fashion mnist."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

import numpy as np
from argparse import ArgumentParser

import tensorflow as tf
from utils import get_nist_data


# some functions for building a convnet.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def shuffle_batch(X, y, batch_size):
    """This function is from the ageron(@github handle) tutorials."""
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        x_batch, y_batch = X[batch_idx], y[batch_idx]
        yield x_batch, y_batch


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--data',
                        choices=['mnist', 'fmnist'],
                        dest='tdata',
                        required=True)
    parser.add_argument('--savemodelpath',
                        dest='savemodelpath',
                        default="../models/")
    parser.add_argument('--randlabels',
                        dest='randlabels',
                        action='store_true')
    parser.add_argument('--reg',
                        dest='reg',
                        action='store_true')
    parser.add_argument('--log',
                        dest='logging',
                        action='store_true')
    parser.add_argument('--lr',
                        type=float,
                        dest='lr',
                        default=1e-4)
    parser.add_argument('--batchsize',
                        type=int,
                        dest='bsize',
                        default=50)
    parser.add_argument('--nepochs',
                        type=int,
                        dest='nepochs',
                        default=50)
    return parser


if __name__ == '__main__':
    # get the command line parameters
    parser = build_parser()
    options = parser.parse_args()
    logging = options.logging
    tdata = options.tdata
    randlabels = options.randlabels
    n_epochs = options.nepochs
    batch_size = options.bsize
    lr = options.lr
    reg = options.reg
    modelpath = options.savemodelpath
    regstrength = 0.001

    # some general mnist/fmnist properties
    nclasses = 10
    validation = True  # whether to use validation set
    if options.tdata == "mnist":
        mdata = True
    else:
        mdata = False

    xtuple, ytuple = get_nist_data(mst=mdata,
                                   validation=validation,
                                   norm_divisor=255.0,
                                   label_to_categorical=True,
                                   num_classes=nclasses)
    x_train, x_valid, x_test = xtuple
    y_train, y_valid, y_test = ytuple
    if randlabels:
        y_train_true, y_random_labels = y_train, np.random.permutation(y_train)
        y_train = y_random_labels
        n_epochs = 100  # requires more epochs to fit random labels.
    if logging:
        print("x_valid shape: ", x_valid.shape, "x_train shape: ",
              x_train.shape, "x_test shape: ", x_test.shape)
        print("y_valid shape: ", y_valid.shape, "x_train shape: ",
              y_train.shape, "y_test shape: ", y_test.shape)
    inpshp = x_train[0].shape
    # get working directory
    wd = os.getcwd()
    # modeldirname = "cnn_" + options.tdata + "/"
    if randlabels:
        modeldirname = "cnn_" + options.tdata + "_randlabels/"
    else:
        modeldirname = "cnn_" + options.tdata + "/"
    modelpath = os.path.join(wd, modelpath, modeldirname)
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
    if logging:
        print("Model Path: ", modelpath)
    # building the tf graph
    tf.reset_default_graph()
    with tf.name_scope("input_manipulation"):
        x = tf.placeholder(tf.float32, shape=[None,
                                              inpshp[0],
                                              inpshp[1]])
        x_image = tf.reshape(x, [-1,
                                 inpshp[0],
                                 inpshp[1], 1])
        y_ = tf.placeholder(tf.float32, shape=[None, nclasses])
        tf.add_to_collection("x", x)
        tf.add_to_collection("y_", y_)
        tf.add_to_collection("x_input_reshaped", x_image)
    with tf.name_scope("hidden1"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1_pre = conv2d(x_image, W_conv1) + b_conv1
        h_conv1 = tf.nn.relu(h_conv1_pre)
        h_pool1 = max_pool_2x2(h_conv1)
        tf.add_to_collection("hidden1_hconv_pre", h_conv1_pre)
        tf.add_to_collection("hidden1_hconv_act", h_conv1)
        tf.add_to_collection("hidden1_hpool", h_pool1)
    with tf.name_scope("hidden2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2_pre = conv2d(h_pool1, W_conv2) + b_conv2
        h_conv2 = tf.nn.relu(h_conv2_pre)
        h_pool2 = max_pool_2x2(h_conv2)
        tf.add_to_collection("hidden2_hconv_pre", h_conv2_pre)
        tf.add_to_collection("hidden2_hconv_act", h_conv2)
        tf.add_to_collection("hidden2_hpool", h_pool2)
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        tf.add_to_collection('fc1', h_fc1)
    with tf.name_scope('softmax_linear'):
        W_fc2 = weight_variable([1024, nclasses])
        b_fc2 = bias_variable([nclasses])
        y_logits = tf.matmul(h_fc1, W_fc2) + b_fc2
        tf.add_to_collection('logits', y_logits)
    with tf.name_scope('loss_and_training'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,
                                                       logits=y_logits))
        reg_loss_l2 = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) \
            + tf.nn.l2_loss(W_fc1)
        if reg:
            total_loss = cross_entropy + regstrength*reg_loss_l2
        else:
            total_loss = cross_entropy
        train_step = tf.train.AdamOptimizer(lr).minimize(total_loss)
        tf.add_to_collection('training_step', train_step)
        tf.add_to_collection('cross_entropy', cross_entropy)
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.add_to_collection("correct_prediction", correct_prediction)
        tf.add_to_collection("accuracy", accuracy)
    saver0 = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            for x_batch, y_batch in shuffle_batch(x_train,
                                                  y_train,
                                                  batch_size):
                sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})
            acc_batch = accuracy.eval(feed_dict={x: x_batch, y_: y_batch})
            acc_valid = accuracy.eval(feed_dict={x: x_valid, y_: y_valid})
            loss_train = total_loss.eval(feed_dict={x: x_batch, y_: y_batch})
            if logging:
                print(epoch, "Batch accuracy (Train) :", acc_batch,
                      "Validation accuracy:", acc_valid, "Train (batch) Loss:",
                      loss_train)
        saver0.save(sess, modelpath+'final_cnn_model_'+options.tdata)

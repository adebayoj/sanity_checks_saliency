"""Training lenet style CNN on mnist of fashion mnist."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

import numpy as np
import pickle
from argparse import ArgumentParser

import tensorflow as tf
from utils import get_nist_data


n_inputs = 28*28
n_hidden1 = 2500
n_hidden2 = 1500
n_hidden3 = 500
n_outputs = 10


def neuron_layer(X, n_neurons, name, activation=None, add_to_collection=True):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        z = tf.matmul(X, W) + b
        if add_to_collection:
            tf.add_to_collection(name+"_weights", W)
        if activation:
            return activation(z)
        else:
            return z


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
    # reshape inputs for mlp
    x_valid = x_valid.reshape(-1, 28*28)
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
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
    # modeldirname = "mlp_" + options.tdata + "/"
    if randlabels:
        modeldirname = "mlp_" + options.tdata + "_randlabels/"
    else:
        modeldirname = "mlp_" + options.tdata + "/"
    modelpath = os.path.join(wd, modelpath, modeldirname)
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
    if logging:
        print("Model Path: ", modelpath)
    demobatch = x_test[:20]
    demo_batch_labels = y_test[:20]
    # now let's save demo input to a particular path
    with open(modelpath+"demo_input_output_tuple.pkl", "wb") as fileobj:
        pickle.dump((demobatch, demo_batch_labels), fileobj)
    if randlabels:
        # now let's save demo input to a particular path
        with open(modelpath+"sample_train_inpopt_rand_labels_tuple.pkl",
                  "wb") as fileobj:
            pickle.dump((x_train[:20],
                         y_train_true[:20],
                         y_random_labels[:20]), fileobj)
    # building the tf graph
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, n_inputs], name="x")
    y = tf.placeholder(tf.float32, shape=[None, n_outputs], name="y")
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    with tf.name_scope("mlp"):
        hidden1 = neuron_layer(x,
                               n_hidden1,
                               name="hidden1", activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1,
                               n_hidden2,
                               name="hidden2", activation=tf.nn.relu)
        hidden3 = neuron_layer(hidden2, n_hidden3,
                               name="hidden3", activation=tf.nn.relu)
        logits = neuron_layer(hidden3, n_outputs, name="logits")
        tf.add_to_collection("hidden_activations_1", hidden1)
        tf.add_to_collection("hidden_activations_2", hidden2)
        tf.add_to_collection("hidden_activations_3", hidden3)
        tf.add_to_collection("logits", logits)

    with tf.name_scope("loss_and_training"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                       logits=logits),
            name="loss")
        train_step = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(cross_entropy)
        # TO DO: put in regularization again.
        tf.add_to_collection('training_step', train_step)
        tf.add_to_collection('cross_entropy', cross_entropy)
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.add_to_collection("correct_prediction", correct_prediction)
        tf.add_to_collection("accuracy", accuracy)
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            for x_batch, y_batch in shuffle_batch(x_train,
                                                  y_train,
                                                  batch_size):
                sess.run(train_step, feed_dict={x: x_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={x: x_valid, y: y_valid})
            if logging:
                print(epoch, "Batch accuracy:", acc_batch,
                      "Validation accuracy:", acc_valid)
            saver.save(sess, modelpath+'mlp_model_'+options.tdata,
                       global_step=epoch)
        saver.save(sess, modelpath+'final_mlp_model_'+options.tdata)
        meta_graph_def = tf.compat.v1.train.export_meta_graph(
            modelpath+'simple_mlp_'+options.tdata+'.meta')

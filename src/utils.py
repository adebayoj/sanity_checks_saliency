"""Contains utility functions for analysis."""
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import saliency
import keras
from keras.datasets import mnist, fashion_mnist
import PIL.Image
import json
import pickle

from inception_v3 import inception_v3
from inception_v3 import inception_v3_arg_scope

slim = tf.contrib.slim


# some utils
def abs_grayscale_norm(img):
    """Returns absolute value normalized image 2D."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"
    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        img = np.absolute(img)
        img = img/float(img.max())
    else:
        img = saliency.VisualizeImageGrayscale(img)
    return img


def diverging_norm(img):
    """Returns image with positive and negative values."""

    assert isinstance(img, np.ndarray), "img should be a numpy array"
    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        imgmax = np.absolute(img).max()
        img = img/float(imgmax)
    else:
        img = saliency.VisualizeImageDiverging(img)
    return img


def LoadImage(file_path, resize=True, sztple=(299, 299)):
    with tf.gfile.GFile(file_path, 'rb') as file:
        img = PIL.Image.open(file).convert('RGB')
    if resize:
        img = img.resize(sztple, PIL.Image.ANTIALIAS)
    img = np.asarray(img)
    return img / 127.5 - 1.0


def plot_single_img(img,
                    ax=False,
                    norm=diverging_norm,
                    show_axis=False,
                    grayscale=False,
                    cmap='gray',
                    title='',
                    fig_size=(4, 4)):
    """Function to plot a single image."""

    plt.figure(figsize=fig_size)
    if norm:
        img = norm(img)
    if not show_axis:
        plt.axis('off')
    plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.show()


def inception_block_names():
    layer_randomization_order = ['InceptionV3/Logits',
                                 'InceptionV3/Mixed_7c',
                                 'InceptionV3/Mixed_7b',
                                 'InceptionV3/Mixed_7a',
                                 'InceptionV3/Mixed_6e',
                                 'InceptionV3/Mixed_6d',
                                 'InceptionV3/Mixed_6c',
                                 'InceptionV3/Mixed_6b',
                                 'InceptionV3/Mixed_6a',
                                 'InceptionV3/Mixed_5d',
                                 'InceptionV3/Mixed_5c',
                                 'InceptionV3/Mixed_5b',
                                 'InceptionV3/Conv2d_4a_3x3',
                                 'InceptionV3/Conv2d_3b_1x1',
                                 'InceptionV3/Conv2d_2b_3x3',
                                 'InceptionV3/Conv2d_2a_3x3',
                                 'InceptionV3/Conv2d_1a_3x3']
    return layer_randomization_order


def save_pickle_file(dt, absfp, platform='linux_workstation'):
    """function to help to save pickle files."""
    platform_options = ['linux_workstation', 'cloudbucket', 'remote_server']
    if platform not in platform_options:
        raise ValueError("Platform option not availaible.")

    if platform == "cloudbucket":
        with tf.gfile.Open(absfp, "wb") as fileobj:  # this is writing bytes
            fileobj.write(pickle.dumps(dt))
        return True
    else:
        with open(absfp, "wb") as fileobj:  # this is writing bytes
            fileobj.write(pickle.dumps(dt))
        return True


def load_pickle_file(absfp, platform='linux_workstation'):
    """function to help to save pickle files."""
    platform_options = ['linux_workstation',
                        'cloudbucket',
                        'remote_server']
    if platform not in platform_options:
        raise ValueError("Platform option not availaible.")

    if platform == "cloudbucket":
        with tf.gfile.Open(absfp, "rb") as infile:
            dt = pickle.load(infile)
        return dt
    else:
        with open(absfp, "rb") as infile:
            dt = pickle.load(infile)
        return dt


def get_saliency_constructors(model_graph,
                              model_session,
                              logit_tensor,
                              input_tensor,
                              gradcam=False,
                              conv_layer_gradcam=None):
    """Initialize mask functions in saliency package.

    Args:
        model_graph: tf graph of the model.
        model_session: tf session with trained model loaded.
        logit_tensor: tensor corresponding to the model logit output.
        input_tensor: tensor coresponding to the input data.
        gradcam: boolean to indicate whether to include gradcam.
        conv_layer_gradcam: tensor corresponding to activations
                            from a conv layer, from the trained model.
                            Authors recommend last layer.
    Returns:
        saliency_constructor: dictionary (name of method, and value is
                              function to each saliency method.
        neuron_selector: tensor of specific output to explain.
    """

    assert (type(tf.Graph()) == type(model_graph)),\
        ("Model graph should be of type {}".format(type(tf.Graph())))

    if gradcam and conv_layer_gradcam is None:
        raise ValueError("If gradcam is True, then conv_layer_gradcam"
                         "is be provided.")
    with model_graph.as_default():
        with tf.name_scope("saliency"):
            neuron_selector = tf.placeholder(tf.int32)
            y_salient = logit_tensor[neuron_selector]
    gradient_saliency = saliency.GradientSaliency(model_graph,
                                                  model_session,
                                                  y_salient,
                                                  input_tensor)
    guided_backprop = saliency.GuidedBackprop(model_graph,
                                              model_session,
                                              y_salient,
                                              input_tensor)
    integrated_gradients = saliency.IntegratedGradients(model_graph,
                                                        model_session,
                                                        y_salient,
                                                        input_tensor)
    saliency_constructor = {'vng': gradient_saliency,
                            'gbp': guided_backprop,
                            'ig': integrated_gradients}
    if gradcam:
        gradcam = saliency.GradCam(model_graph,
                                   model_session,
                                   y_salient,
                                   input_tensor,
                                   conv_layer_gradcam)
        saliency_constructor['gc'] = gradcam
    return saliency_constructor, neuron_selector


def vargrad(img, salfunc, feed_dict, noise_std=0.15, nruns=25):
    """This function computes vargrad for arbitrary saliency function.

    ARGS:
        img: input ndarray.
        salfunc: saliency function to be used.
        kwargs: kwargs if any for this function.
    Returs:
        vargrad: vector of vargrad map.
    """

    # check the input
    assert isinstance(img, np.ndarray), "Not a numpy array."
    grad_list = []
    stdev = noise_std * (np.max(img) - np.min(img))
    for i in range(nruns):
        noise = np.random.normal(0, stdev, img.shape)
        img_noise = img + noise
        mask = salfunc(img_noise, feed_dict=feed_dict)
        grad_list.append(mask)
    vargrad = np.var(np.array(grad_list), axis=0)
    assert len(vargrad.shape) == len(img.shape), ("vargrad output not "
                                                  " same as input")
    return vargrad


def get_nist_data(mst=True,
                  validation=True,
                  norm_divisor=255.0,
                  label_to_categorical=True,
                  num_classes=10):
    if mst:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if norm_divisor:
        x_train = x_train/norm_divisor
        x_test = x_test/norm_divisor
    if validation:
        x_valid, x_train = x_train[:5000], x_train[5000:]
        y_valid, y_train = y_train[:5000], y_train[5000:]
    if label_to_categorical:
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_train = keras.utils.to_categorical(y_train, num_classes)
    if validation:
        y_valid = keras.utils.to_categorical(y_valid, num_classes)
    return (x_train, x_valid, x_test), (y_train, y_valid, y_test)


class Inceptionv3_Wrapper(object):
    def __init__(
        self,
        chkpointpath='../models/inceptionv3/inception_v3.ckpt',
        lblmetadatapath='../models/inceptionv3/imagenet_class_index.json'
    ):

        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.chkpntpath = chkpointpath
        self.labelmetadatapath = lblmetadatapath
        self.num_classes = 1001  # 0 is background or null class
        self.label_dict = {}
        if not tf.io.gfile.exists(self.chkpntpath):
            raise ValueError("There is no checkpoint at the input path")
        with self.graph.as_default():
            self.input_batch = tf.placeholder(tf.float32,
                                              shape=(None, 299, 299, 3))
            with slim.arg_scope(inception_v3_arg_scope()):
                _, self.end_points = inception_v3(
                  self.input_batch,
                  is_training=False,
                  num_classes=self.num_classes)
                self.session = tf.Session(graph=self.graph)
                self.saver = tf.train.Saver()
                self.saver.restore(self.session, self.chkpntpath)

            self.logits = self.graph.get_tensor_by_name(
                'InceptionV3/Logits/SpatialSqueeze:0')
            self.trainable_variables = tf.trainable_variables()

        if not tf.io.gfile.exists(self.labelmetadatapath):
            raise ValueError("There is no label file at the input path.")

        # process labels in appropriate dictionary
        with open(self.labelmetadatapath) as json_file:
            data = json.load(json_file)
            shift = 0
            if self.num_classes == 1001:
                self.label_dict = {0: ["background", "background"]}
                shift = 1
            for key in data:
                self.label_dict[int(key)+shift] = data[key]

    def predict(self, batch):
        """predict on batch data."""
        if not isinstance(batch, (np.ndarray)):
            raise ValueError("input should be a numpy array!")

        if len(batch.shape) < 4:
            raise ValueError(
              "Shape should be (nsamples, height, width, channels)")

        feed_dict = {self.input_batch: batch}
        logits = self.session.run(self.logits,
                                  feed_dict=feed_dict)
        return logits

    def reinitlayerlist(self, blocklist, independent=False):
        """Reinitialize Tensors with these names."""
        if independent:
            self.__init__()
        tensors = []
        for op in self.trainable_variables:
            for blockname in blocklist:
                if blockname in op.name:
                    tensors.append(op)

        # now reinitialize
        with self.graph.as_default():
            to_ini = tf.initialize_variables(tensors)
            _ = self.session.run(to_ini)
        return True

    def indextoclsnames(self, arr, topk=5):
        """Given a numpy vector, get label names for topk."""
        names = []
        topk_indices = arr.argsort()[::-1][:topk]
        for val in topk_indices:
            names.append(self.label_dict[val][1])
        return names


if __name__ == '__main__':
    pass

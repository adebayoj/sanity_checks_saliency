"""Cascading/independent block randomization for InceptionV3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from argparse import ArgumentParser

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # don't use gpu for this.

import tensorflow as tf
from utils import Inceptionv3_Wrapper
from utils import get_saliency_constructors
from utils import save_pickle_file
from utils import inception_block_names
from utils import LoadImage


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--rand',
        choices=['cascading', 'independent'],
        dest='randtype',
        required=True)
    parser.add_argument(
        '--chkpntpath',
        dest='chkpntpath',
        default=('/home/julius/research/test_sanity_checks/'
                 'models/inceptionv3/inception_v3.ckpt'))
    parser.add_argument(
        '--labeldatapath',
        dest='labeldatapath',
        default=('/home/julius/research/test_sanity_checks/'
                 'models/inceptionv3/inception_v3.ckpt'))
    parser.add_argument(
        '--inputimgfolderpath',
        dest='inputimgfolderpath',
        default=('/home/julius/research/sanity_checks_saliency/'
                 'data/demo_images/'))
    parser.add_argument(
        '--normaloutputpath',
        dest='normaloutputpath',
        default=('/home/julius/research/test_sanity_checks/data/models/'
                 'inceptionv3/saliency_independent_rand'))
    parser.add_argument(
        '--randoutputpath',
        dest='randoutputpath',
        default=('/home/julius/research/test_sanity_checks/data/models/'
                 'inceptionv3/saliency_independent_rand'))
    parser.add_argument(
        '--log',
        dest='logging',
        action='store_true')
    parser.add_argument(
        '--normal_masks',
        dest='normal_masks',
        action='store_true')
    return parser


if __name__ == '__main__':
    # get the command line parameters
    parser = build_parser()
    options = parser.parse_args()
    logging = options.logging
    randtype = options.randtype.lower()
    chkpntpath = options.chkpntpath
    labeldatapath = options.labeldatapath
    inputimgfolderpath = options.inputimgfolderpath
    normaloutputpath = options.normaloutputpath
    randoutputpath = options.randoutputpath
    computnormalmasks = options.normal_masks

    # default parameters
    # pair saliency automatically computes smoothgrad squared
    smoothgradsq = False  # turn off smoothgradsquared
    nsamples_sg = 50  # number of noisy samples to compute
    xsteps_ig = 50 # interpolation steps for integrated gradients
    stdev_spread_sg = 0.15  # std for smoothgrad noisy samples
    gradcam_three_dims = True  # gradcam should be 3 channels

    # assemble a list of these images
    listfiles = tf.io.gfile.listdir(inputimgfolderpath)
    demo_batch = []
    for fl in listfiles:
        demo_batch.append(LoadImage(inputimgfolderpath+fl, resize=True))
    demo_batch = np.array(demo_batch)
    if logging:
        print(demo_batch.shape)

    layer_randomization_order = inception_block_names()

    # compute normal saliency masks.
    if computnormalmasks:
        # load of inception model
        inception_model = Inceptionv3_Wrapper(
           chkpointpath=chkpntpath,
           lblmetadatapath=labeldatapath)

        # specify necessary saliency setup functions.
        saliency_dict, n_selector = get_saliency_constructors(
            inception_model.graph,
            inception_model.session,
            inception_model.logits[0],
            inception_model.input_batch,
            gradcam=True,
            conv_layer_gradcam=inception_model.end_points['Mixed_7c'])

        # dictionary of methods that we'll compute.
        saliency_methods = {
            'Gradient': saliency_dict['vng'].GetMask,
            'SmoothGrad': saliency_dict['vng'].GetSmoothedMask,
            'Guided\nBackProp': saliency_dict['gbp'].GetMask,
            'Integrated\nGradients': saliency_dict['ig'].GetMask,
            'IG\nSmoothGrad': saliency_dict['ig'].GetSmoothedMask,
            'GradCAM': saliency_dict['gc'].GetMask}

        for k, current_image in enumerate(demo_batch):
            if logging:
                print("On input {} in the demo batch".format(k))
            # all black baseline for integrated gradients.
            baseline = np.zeros(current_image.shape)

            # get model prediction on this input.
            imglogits = inception_model.session.run(
                [inception_model.logits],
                feed_dict={inception_model.input_batch:
                           np.expand_dims(current_image, 0)})[0]

            # set up input to the saliency functions.
            prediction_class = imglogits.argmax()  # output to explain.
            gen_feed_dict = {n_selector: prediction_class}

            # set up params for each saliency method.
            saliency_params = {
                'Gradient': {"feed_dict": gen_feed_dict},
                'SmoothGrad': {"feed_dict": gen_feed_dict,
                               "stdev_spread": stdev_spread_sg,
                               "nsamples": nsamples_sg,
                               "magnitude": smoothgradsq},
                'Guided\nBackProp': {"feed_dict": gen_feed_dict},
                'Integrated\nGradients': {"feed_dict": gen_feed_dict,
                                          "x_steps": xsteps_ig,
                                          "x_baseline": baseline},
                'IG\nSmoothGrad': {"feed_dict": gen_feed_dict,
                                   "x_steps": xsteps_ig,
                                   "nsamples": nsamples_sg,
                                   "stdev_spread": stdev_spread_sg,
                                   "x_baseline": baseline,
                                   "magnitude": smoothgradsq},
                'GradCAM': {"feed_dict": gen_feed_dict,
                            "three_dims": gradcam_three_dims}}

            # store masks in dictionaries and pickle the output
            output_masks = {}
            for key in saliency_methods:
                if logging:
                    print("On Method: {}".format(key))
                params = saliency_params[key]
                output_masks[key] = saliency_methods[key](current_image,
                                                          **params)
            # compute input-gradient and guided-gradcam
            output_masks["Input-Grad"] = np.multiply(output_masks['Gradient'],
                                                     current_image)

            output_masks["GBP-GC"] = np.multiply(
                output_masks['Guided\nBackProp'],
                output_masks['GradCAM'])

            # now serialize the result (pickle) to storage
            filename = listfiles[k].split(".")[0] + "++normalmasks.pickle"
            save_pickle_file(output_masks,
                             normaloutputpath + filename,
                             platform='cloudbucket')

    if logging:
        print("Done with computing masks with normal model.")

    # cascading or independent randomization now.
    for i, layer_name in enumerate(layer_randomization_order):
        if randtype == "cascading":
            if logging:
                print("Cascading randomization of layer {}".format(
                    layer_name))
            inception_model = Inceptionv3_Wrapper(
                chkpointpath=chkpntpath,
                lblmetadatapath=labeldatapath)
            layer_list = layer_randomization_order[:i+1]
            inception_model.reinitlayerlist(layer_list,
                                            independent=False)
        elif randtype == "independent":
            if logging:
                print("Independent randomization of layer {}".format(
                    layer_name))
            # Randomize all trainable ops in that layer/block
            inception_model = Inceptionv3_Wrapper(
                chkpointpath=chkpntpath,
                lblmetadatapath=labeldatapath)
            layer_list = [layer_name]
            inception_model.reinitlayerlist(layer_list,
                                            independent=True)

        # We need to redefine these parameters for
        # randomization since we load a new model from scratch.
        # specify necessary saliency setup functions.
        saliency_dict, n_selector = get_saliency_constructors(
            inception_model.graph,
            inception_model.session,
            inception_model.logits[0],
            inception_model.input_batch,
            gradcam=True,
            conv_layer_gradcam=inception_model.end_points['Mixed_7c'])

        # dictionary of methods that we'll compute.
        saliency_methods = {
            'Gradient': saliency_dict['vng'].GetMask,
            'SmoothGrad': saliency_dict['vng'].GetSmoothedMask,
            'Guided\nBackProp': saliency_dict['gbp'].GetMask,
            'Integrated\nGradients': saliency_dict['ig'].GetMask,
            'IG\nSmoothGrad': saliency_dict['ig'].GetSmoothedMask,
            'GradCAM': saliency_dict['gc'].GetMask}

        # list to store collection of images
        list_of_random_mask_per_layer = []
        for j, current_image in enumerate(demo_batch):
            if logging:
                print("On input {} in the demo batch".format(j))
            # all black baseline for integrated gradients.
            baseline = np.zeros(current_image.shape)

            # get model prediction on this input.
            imglogits = inception_model.session.run(
                [inception_model.logits],
                feed_dict={inception_model.input_batch:
                           np.expand_dims(current_image, 0)})[0]

            # set up input to the saliency functions.
            prediction_class = imglogits.argmax()  # output to explain.
            gen_feed_dict = {n_selector: prediction_class}

            # set up params for each saliency method.
            # set up params for each saliency method.
            saliency_params = {
                'Gradient': {"feed_dict": gen_feed_dict},
                'SmoothGrad': {"feed_dict": gen_feed_dict,
                               "stdev_spread": stdev_spread_sg,
                               "nsamples": nsamples_sg,
                               "magnitude": smoothgradsq},
                'Guided\nBackProp': {"feed_dict": gen_feed_dict},
                'Integrated\nGradients': {"feed_dict": gen_feed_dict,
                                          "x_steps": xsteps_ig,
                                          "x_baseline": baseline},
                'IG\nSmoothGrad': {"feed_dict": gen_feed_dict,
                                   "x_steps": xsteps_ig,
                                   "nsamples": nsamples_sg,
                                   "stdev_spread": stdev_spread_sg,
                                   "x_baseline": baseline,
                                   "magnitude": smoothgradsq},
                'GradCAM': {"feed_dict": gen_feed_dict,
                            "three_dims": gradcam_three_dims}}

            output_masks = {}
            for key in saliency_methods:
                if logging:
                    print("On Method: {}".format(key))
                params = saliency_params[key]
                output_masks[key] = saliency_methods[key](current_image,
                                                          **params)
            # compute input-gradient and guided-gradcam
            # compute input-gradient and guided-gradcam
            output_masks["Input-Grad"] = np.multiply(output_masks['Gradient'],
                                                     current_image)

            output_masks["GBP-GC"] = np.multiply(
                output_masks['Guided\nBackProp'],
                output_masks['GradCAM'])

            # now serialize the result (pickle) to storage
            if "/" in layer_name:
                layer_save_name = layer_name.split("/")[-1]  # last item
            filename = listfiles[j].split(".")[0] + "++" + randtype + \
                "++" + layer_save_name + ".pickle"
            save_pickle_file(output_masks,
                             randoutputpath + filename, platform='cloudbucket')

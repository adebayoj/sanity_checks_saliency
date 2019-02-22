
Sanity Checks for Saliency Maps
=====================
This repository will be updated with code to replicate the paper
**Sanity Checks for Saliency Maps** by<br/>
*Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, & Been Kim*.

#### This repository will be updated with code by Monday March 4th. 

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/saliency_methods_and_edge_detector.png" width="700">


### Overview

Saliency methods have emerged as a popular tool to highlight
features in an input deemed relevant for the prediction of a 
learned model. Several saliency methods have been proposed, often 
guided by visual appeal on image data. In this work, we propose 
an actionable methodology to evaluate what kinds of explanations 
a given method can and cannot provide. We find that reliance, 
solely, on visual assessment can be misleading. Through extensive
experiments we show that some existing saliency methods are 
independent both of the model and of the data generating process.
Consequently, methods that fail the proposed tests are 
inadequate for tasks that are sensitive to either data or model,
such as, finding outliers in the data, explaining the 
relationship between inputs and outputs that the model learned,
or debugging the model. We interpret our findings through an 
analogy with edge detection in images, a technique that requires 
neither training data nor model. Theory in the case of a 
linear model and a single-layer convolutional neural network
supports our experimental findings.

#### Model Randomization Test 

For the model randomization test, we randomize the weights of a 
model starting from the top layer, successively, all the way to 
the bottom layer. This procedure destroys the learned
weights from the top layers to the bottom ones. We compare the resulting explanation from a network with random weights to the one obtained with the model’s original weights. Below we show the
evolution of saliency masks from different methods for a demo image from the ImageNet dataset and the Inception v3 model.

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/bird_img_cascading_demo_diverging_visualization.png" width="700">


#### Data Randomization Test

In our data randomization test, we permute the training labels
and train a model on the randomized training data. A model 
achieving high training accuracy on the randomized training data 
is forced to memorize the randomized labels without being able to
exploit the original structure in the data. We now compare 
saliency masks for a model trained on random labels and one
trained true labels. We present examples below on MNIST and Fashion MNIST.

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/mnist_cnn_random_labels_test.png" width="700">

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/mnist_mlp_random_labels_test.png" width="700">

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/fmnist_cnn_random_labels_test.png" width="700">

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/fmnist_mlp_random_labels_test.png" width="700">

### Data

See /doc/data/ for the exact ImageNet images used in this 
work.  

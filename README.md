Sanity Checks for Saliency Maps
=====================
This repository will be updated with code to replicate the paper
**Sanity Checks for Saliency Maps** by *Julius Adebayo, Justin 
Gimer, Ian Goodfellow, Moritz Hardt, & Been Kim*.

We will be updating this repository with code replicate our 
experiments. 

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

#### Model Randomization Test Inception v3 on ImageNet

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/bird_img_cascading_demo.png" width="700">

#### Data Randomization Test

<img src="https://raw.githubusercontent.com/adebayoj/sanity_checks_saliency/master/doc/figures/mnist_digit_zero_random_labels_test.png" width="700">

### Data

An initial version of the demo data used can be obtained at: 
/doc/data/. We will add all of the data used for the ImageNet
experiments. 

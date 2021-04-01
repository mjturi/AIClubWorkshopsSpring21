---
layout: page
title: Generative Adversial Networks
show_in_menu: false
disable_anchors: true
---

# Generative Adversial Networks(GANs)
Generative Aversarial Networks are a type of neural network that can be used to generate fake images that are near identical to real images. These networks were first introduced by Goodfellow et al. in their 2014 paper, [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)[1]. 

## How are the images generated?
With GANs we use two neural networks for training. Firstly, we have a ```generator``` G(x). This network recieves an input of randomly generated noise. It is trained to output imitation images that eventually become near-identical images to their ground-truth originals. We also have a ```discriminator```(adversary) D(x), that determines whether a given image is real or fake.

Both of these models play an adversarial game where the generator will try to fool the discriminator by generating data that is similar to those in the training set [2]. As mentioned, the generator will attempt to generate images from random noise. The discriminator, which is learning to determine whether images are real or fake, will guide the generator by determining if the generators images are real or fake. During training, the generator will progressively become better at creating iamges that look real, while the discriminator will become better at telling them apart.

The process reaches equilibrium when the discriminator can no longer distinguish real images from fake[3].

<p align="center">
  <img width="460" height="300" src="https://www.tensorflow.org/tutorials/generative/images/gan2.png">
</p>

Here we can take a look at a GAN during it's training:

<p align="center">
  <img width="460" height="300" src="https://github.com/adityabingi/DCGAN-TF2.0/raw/master/results/dcgan_training.gif">
</p>

The author for the results above can be found [here](https://github.com/adityabingi/DCGAN-TF2.0).

# References
The following section include links to various references used throughout this tutorial
[1] [GANs with Keras and TensorFlow](https://www.pyimagesearch.com/2020/11/16/gans-with-keras-and-tensorflow/)
[2] [GANs — A Brief Introduction to Generative Adversarial Networks](https://medium.com/analytics-vidhya/gans-a-brief-introduction-to-generative-adversarial-networks-f06216c7200e#:~:text=How%20does%20it%20work%3F,fake%20data%20from%20real%20data.) 
[3] [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)
[4] []()
[5] []()
[6] []()
[7] []()
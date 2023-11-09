# Generative_Models_Introduction

## Generative Adversarial Networks (GAN): 

The model has two subnetworks, a discriminator and a generator: (Here we discuss Deep Convolutional GAN or DCGAN)

### Discriminator: 

It tries to distinguish between images belonging to the dataset and those produced by the generator. It does so by giving $D_\phi$, which lies between 0 and 1. The more $D_\phi$ is closed to one, the more the discriminator has detected that the input images are from the dataset. 

![img00](./images/c5.JPG)

### Generator: 

The generator tries to fool the discriminator in a way that it always gives $D_\phi$ close to one. In other words, the generator tries to produce images that are as similar as possible to the training set images, making the discriminator confused.

![img00](./images/c6.JPG)


As a result, the final loss function can be written as mini-max equation, with $\theta$ as the parameters of the generator network and $\phi$ as the parameters of the discriminator. The discriminator wants to maxmimize the second term, where as the generator tries to minimize the second term be forcing $D_{\phi} \sim 1$

$$argmin_{G_{\theta}} argmax_{P_{\phi}} V(G_{\theta}, P_{\phi}) = E_{x \sim P_{data}} \[ log D_{\phi(x)} \] + E_{x \sim P_{G_{\theta}}} \[ log (1 - D_{\phi(x)}) \]$$

After both the generator and the discriminator are trained, the discriminator is put aside and the generator starts from random noises. 

## Diffusion-Based: 

Diffusion-based models have many variations, among which Denoising Diffusion Probablistic Models (DDPM) are discussed here. In DDPM, only one network is trained. We add a random noise to $x_0$, which is an image of the dataset, and obtain $x_t$ based on this formula:

$$x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1-\bar{\alpha_t}} \epsilon$$


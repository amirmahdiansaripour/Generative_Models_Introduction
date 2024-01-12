# Generative_Models_Introduction

## Generative Adversarial Networks (GAN): 

The model has two subnetworks, a discriminator and a generator: (Here we discuss Deep Convolutional GAN or DCGAN)

### Discriminator: 

It tries to distinguish between images belonging to the dataset and those produced by the generator. It does so by giving $D_\phi$, which lies between 0 and 1. The more $D_\phi$ is closed to one, the more the discriminator has detected that the input images are from the dataset. 

![img00](./images/c5.JPG)

### Generator: 

The generator tries to fool the discriminator in a way that it always gives $D_\phi$ close to one. In other words, the generator tries to produce images that are as similar as possible to the training set images, making the discriminator confused.

![img01](./images/c6.JPG)


As a result, the final loss function can be written as mini-max equation, with $\theta$ as the parameters of the generator network and $\phi$ as the parameters of the discriminator. The discriminator wants to maxmimize the second term, where as the generator tries to minimize the second term be forcing $D_{\phi} \sim 1$

$$argmin_{G_{\theta}} argmax_{P_{\phi}} V(G_{\theta}, P_{\phi}) = E_{x \sim P_{data}} \[ log D_{\phi(x)} \] + E_{x \sim P_{G_{\theta}}} \[ log (1 - D_{\phi(x)}) \]$$

After both the generator and the discriminator are trained, the discriminator is put aside and the generator starts from random noises. 

## Diffusion-Based: 

Diffusion-based models have many variations, among which Denoising Diffusion Probablistic Models (DDPM) are discussed here. In DDPM, only one network is trained. We add a random noise ($\epsilon$) to $x_0$, which is an image of the dataset, and obtain $x_t$ based on this formula:

$$x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1-\bar{\alpha_t}} \epsilon$$

We give $x_t$ along with $t$ (the current step) to a network, generally CNN and U-Net, to predict the noise which has initially been added ($\epsilon$). Hence, the network being trained in DDPM works based on the following loss function: ($\theta*$ is the parameter refering to the optimized weights of the network)


$${\theta}^* = argmin_{\theta} || \epsilon - \epsilon_{\theta}(\sqrt{\bar{\alpha_t}} x_0 + \sqrt{1-\bar{\alpha_t}} \epsilon, t) ||$$

![img02](./images/c7.JPG)

We repeat the above minimization until the network converges: ($T$ is a hyper parameter showing the maximum step in the forward trajectory.)

![img02](./images/c1.JPG)

When the network has learned how to reduce the proper amount of noise from $x_t$ when the step $t$ is given, the sampling step begins. In sampling, we start from a random noise and denoise it step-by-step until we reach an image belonging to the same distribution as the training set.

![img03](./images/c2.JPG)

## Variational Auto-Encoders: 

The idea behind VAE is to transfer our complex distribution to a prior and known distribution by passing it through an encoder. Subsequently, we sample from the latent space (encoder output) by a decoder symmetric to the encoder. Therefore, a general schema of a VAE looks like:

![img12](./images/vae_1_2.png)

For training a VAE, we should pay attention to two errors: **reconstruntion error**, which is how similar the generated images are to the initial ones, and the **regularization error**, which is how similar the latent distribution is to prior distribution ($N(0, I)$).

## Comparison: 

### Training time: 

In order to have a fair comparison, the number of epochs and batch sizes should be equal for both models. However, in order to make sure that $\epsilon_{\theta}(x_t, t)$ has seen random noise during its training, $T$ should be set to large values ($\sim 1000$). 

As we can see, the training of GAN takes more time. The reason is GAN there are two networks to be trained, namely $D_{\phi}$ and $G_{\theta}$. In DDPM, only $\epsilon_{\theta}(x_t, t)$ is trained.

| (batch size, n_epoch) | GAN | training time (second) |
| --- | --- | --- |
| (128, 20) | GAN | 574.85 |
| (128, 20) |  DDPM | 508.57 |


### Inference time: 

Taking an average over 50 times of generating images: 

For GAN:

![img05](./images/c4.JPG)

For DDPM: 

![img06](./images/c3.JPG)

| model | inference time average (second) |
| --- | --- |
| GAN | 0.01 |
| DDPM | 7.2 |

As we can see, there is a significant difference between the inference time of a GAN and a DDPM. The reason is the iterative loop in the sampling phase of DDPM, which runs for $T$ times, whereas GAN generates in only one iteraion.


### Does Reducing T in DDPM Solve the High Inference Time?

Viewing the high inference time of DDPM (7 sec), one may think that reducing the number of backward steps, $T$, which has been set to 1000, can reduce the delay. Let's try with $T = 100$ and $T = 500$:

1. When $T = 100$:

![img10](./images/T_100_exe_time.JPG) | ![img11](./images/T_100.JPG)

Obviously, the inference time has reduced (from 7 to 1 on average), but the quality of images is undesirable. The resoan is that when the number of steps is remarkablt reduced, the DDPM does not see pure noise in its training step. Therefore, it performs awfully in the inference step starting from $x_T = N(0, I)$ 

### Output Images Examples: 

1. GAN output for (n_epochs = 20, batch_size = 128)

![img07](./images/c9.JPG)

2. GAN output for (n_epochs = 60, batch_size = 100)

![img08](./images/c8.JPG)

3. DDPM output for (n_epochs = 20, batch_size = 128)

![img09](./images/c10.JPG)


## References: 

1. Jonathan Ho, Ajay Jain, Pieter Abbeel. $\textbf{Denoising Diffusion Probabilistic Models}$. arXiv preprint: 2006.11239, 2020.

2. Ian J.Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. $\textbf{Generative Adversarial Nets}$. arXiv prepeint: 1406.2661, 2014.

3. Olaf Ronneberger, Philipp Fischer, Thomas Brox. $\textbf{U-Net: Convolutional Networks for Biomedical Image Segmentation}$. arXiv preprint: 1505.04597.

4. [Medium link for DCGAN](https://towardsdatascience.com/image-generation-in-10-minutes-with-generative-adversarial-networks-c2afc56bfa3b)

5. [Medium link for DDPM](https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1)  



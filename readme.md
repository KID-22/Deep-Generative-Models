# Deep Generative Models
In this project, we implemented VAE, GAN, WGAN, WGAN-GP on two synthetic datasets with Pytorch.

## Folder Structure
The following shows basic folder structure.
```
├── fig # figures
├── data
│   ├── circle_1 # circle_1 data
│   ├── circle_1 # circle_1 data
│   └── generate_data.ipynb # generate synthetic dataset
├── code
│   ├── model # include GAN.py VAE.py WGAN.py WGAN_GP.py
│   ├── main_vae.py # gateway for VAE
│   ├── ...
│   ├── ...
│   ├── main_wgan_gp.py # gateway for WGAN_GP
│   └── utils.py # utils
├── ckpt # model checkpoints to be saved here
├── result # generation sample results and loss to be saved here
└── log # training log to be saved here
```

## Dataset
+ **Circle 1**: This dataset consists of 28 components, where each component is isotropic and generated from a mixture of 2-dimensional standard normal Gaussian. The 28 components are uniformly distributed on a circle (radius=70).

+ **Circle 2**: Similar to Circle 1 dataset, Circle 2 dataset also consists of 28 isotropic standard normal Gaussian components. Differently, the 28 components are uniformly distributed over 2 circles, 8 components on the small circle (radius=25) and 20 components on the large circle (radius=50).

## Reproducing Results
For a easy way to reproduce any results in this project, you can run the command as follows.
```
sh run.sh
```
To run the model on a different dataset, or with different hyper-parameters, etc, simply modify the config in the main_*.py file.


## Results
### Sample Quality
#### Circle 1
![](./fig/circle_1/vae.jpg)
![](./fig/circle_1/gan.jpg)
![](./fig/circle_1/wgan.jpg)
![](./fig/circle_1/wgan_gp.jpg)


#### Circle 2
![](./fig/circle_2/vae.jpg)
![](./fig/circle_2/gan.jpg)
![](./fig/circle_2/wgan.jpg)
![](./fig/circle_2/wgan_gp.jpg)

### Visualization of Loss
#### Circle 1
![](./fig/circle_1/vae_loss.jpg)
![](./fig/circle_1/gan_loss.jpg)
![](./fig/circle_1/wgan_loss.jpg)
![](./fig/circle_1/wgan_gp_loss.jpg)

#### Circle 2
![](./fig/circle_2/vae_loss.jpg)
![](./fig/circle_2/gan_loss.jpg)
![](./fig/circle_2/wgan_loss.jpg)
![](./fig/circle_2/wgan_gp_loss.jpg)

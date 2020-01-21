# SemiSupervisedPytorchGAN
A semi supervised GAN for image classification implemented in Pytorch

______________________

**21.01.2020**

[SemiSupervisedGAN.ipynb](SemiSupervisedGAN.ipynb) is a simple semi supervised GAN inspired by [\[1\]](https://arxiv.org/abs/1606.03498) and [\[2\]](https://arxiv.org/abs/1511.06390), where the Generator is a ([DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)-like) deCNN and the Discriminator is, in fact, a ResNet classifier. For more information on this project and the topic in general please refer to the [meetup](https://www.meetup.com/Paris-Women-in-Machine-Learning-Data-Science/events/267059218/) [slides](220120_meetup_slides.pdf).
 
The dataset used for the dog vs. cat classification is available on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). For every experiment, I set *L* labeled images aside and used the rest as unlabeled data. Since *L* is small, the results strongly depend on the choice of the labeled images, and should be averaged over a few different labeled subsets. In the table below, `Supervised accuracy` refers to the accuracy of a ResNet18 model trained on *L* labeled images, whereas `Semi-supervised accuracy` comes from a semi-supervised GAN that uses ResNet18 as its Discriminator/Classifier, with `SSGAN epochs` being the number of training epochs used to get the best result. 

| *L*  | Set #  | Supervised accuracy (%) | Semi-supervised accuracy (%) | SSGAN epochs  | Increase in accuracy (%) |
|---|---|---|---|---|---|
| 100  | 1  | 60.2  | 83.2  | 40  | 23  |
| 100  | 2  | 56.7  |   |   |   |
| 100  | 3  | 62.1  |   |   |   |


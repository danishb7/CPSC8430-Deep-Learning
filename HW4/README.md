Problem Statement:-
Propose simple GAN model that generates images for a set of input images which is a complex model based on CNNs architecture.
The input is images that are provided to the network.
The output will also be fake generated image given by the learnings from the input images. The above is made possible by 
discriminator and generator networks in the GAN.

Dataset:-
The homework given that was to include a sample of resources and inspection records uses the samples as training data for 
the model. The CIFAR10 dataset encompasses 50000 training images and 10000 test images. The data is in their respective png files.

Method
The models in this homework are DCGAN and WGAN models which take image inputs and output images. The only thing that is changed 
in these models is the loss function. For DCGAN they are called BCE loss and for WGAN they are called W loss. It progresses through 
processing stages through CNN (In this case, we feed pixels), weight initialization, declaration of hyperparameters, declaration 
of error and scores, training, testing and even representation of the learning curves.

Analysis
While training WGAN is slow as compared to DCGAN, but it is better in terms of providing the variety of outcomes. When it comes to 
the reconstruction of the class label through the discriminator, ACGAN is better. Though DCGAN was trained on a variety of images, it 
sometimes only generates one type of image. And also, it is right to set all the parameters before training so that the loss 
decreases as the epoch increases.

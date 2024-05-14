'''
Marked exercises after Lecture 10 on GANs

This notebook contains the marked exercises with instructions and explanations.
Work through the cells below in sequential order, executing each cell as you progress. Throughout the exercise, you will encounter 
instructions marked with the words YOUR CODE HERE followed by raise NotImplementedError(). 
You will have to substitute raise NotImplementedError() with your own code. Follow the instructions and write the code to complete the tasks.
Along the way, you may also find questions. Try to reflect on the questions before/after running the code.
This notebook was developed at the Idiap Research Institute by Alina Elena Baia, Darya Baranouskaya and Olena Hrynenko (equal contribution).

Note: This notebook serves as the main file for the marked exercise and contains the code for initialising the models, initialising the dataset, 
setting the hyperparameters for training, and training the model. No code implementation is required in this notebook file. 
You are allowed to change hyperparameters.
Make sure to upload the required source files. You will be asked to modify these files in order to complete the tasks.


You are asked to complete the following tasks related to completing the implementation of a GAN model and its training function, training
WGAN [40 points]
2.10.1 [3 points] Implement DiscriminatorBlock in model.py.
2.10.2 [5 points] Implement GeneratorBlock in model.py.
2.10.3 [5 points] Compete Generator in model.py: specify kernel_sizes, stride_sizes, padding_sizes in Generator def __init__.
2.10.4 [3 points] Implement weight clamping in trainer.py: def clamp_weights(self)
2.10.5 [9 points] Implement Discriminator update step in trainer.py: def disc_step(self, z, real_images).
2.10.6 [9 points] Implement Generator update step in trainer.py: def gen_step(self).
2.10.7 [6 points] Complete the call of disc_step() and gen_step in train_epoch(). Train the GAN for at least 5000 iterations and 5 epochs, 
and report the generated images.

Further instructions and explanations are provided in the accompanying README.md and files.

IMPORTANT REMINDER: Make sure to also complete the following tasks, which do no require code implementation. 
Follow the instructions provided in the Marked_exercises_submission2_lecture10.pdf file.

Questions [20 points] – Make sure to reference any sources used
2.10.8 [10 points] Challenges with training GANs (200 words max)
Discuss mode collapse as a challenge with training GANs and choose a second challenge yourself. 
Outline methods that address these challenges. Additionally, state the cause for these challenges. 
Support your claims with formulas, plots, and graphs. One plot = 50 words.

2.10.9 [10 points] Evaluation metrics for GANs (200 words max)
Discuss two metrics for evaluating the performance of GANs. Describe the advantages and
disadvantages of each metric. Support your claims with formulas, plots, and graphs. One
plot = 50 words.
'''

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Generator, Discriminator, weights_init
from trainer import WGANTrainer

torch.manual_seed(42)


def load_fashion_mnist_dataset(batch_size):
    transforms_fashion_mnist = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True,
                                     transform=transforms_fashion_mnist)
    # For testing your implementation we recommend to use a subset of the dataset to save images more often,
    # although that would be reflected in the quality of the generated images
    from torch.utils.data import Subset
    dataset = Subset(dataset, range(40000))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # You are allowed to change hyperparameters
    # Although, if you implemented everything correctly this hyperparmeters should work
    lr_gen = 0.00007
    lr_disc = 0.00005
    batch_size = 64
    weight_cliping_limit = 0.01
    n_critic_steps = 2
    n_image_channels = 1  #3 if RGB and 1 if Greyscale

    gen = Generator(in_channels=100, out_channels=n_image_channels)
    disc = Discriminator(in_channels=n_image_channels)

    print(gen.apply(weights_init))
    print(disc.apply(weights_init))

    gen.to(device)
    disc.to(device)

    # assert for you to check the correctness of the sizes of models outputs
    z = torch.rand((4, 100, 1, 1), device=device)
    fake_image = gen(z)
    assert list(fake_image.shape) == [4, n_image_channels, 32, 32]
    assert list(disc(fake_image).shape) == [4, 1, 1, 1]


    # WGAN with gradient clipping uses RMSprop instead of ADAM
    optimizer_gen = torch.optim.RMSprop(gen.parameters(), lr=lr_gen)
    optimizer_disc = torch.optim.RMSprop(disc.parameters(), lr=lr_disc)

    train_loader = load_fashion_mnist_dataset(batch_size)

    #initialise the trainer
    trainer = WGANTrainer(model_gen=gen, model_disc=disc,
                          optimizer_gen=optimizer_gen, optimizer_disc=optimizer_disc,
                          n_disc_steps=n_critic_steps, weight_cliping=weight_cliping_limit, device=device)

    #train WGAN
    trainer.train(n_epoches=30, train_loader=train_loader)



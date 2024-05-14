import torch
import numpy as np
import os
from torcheval.metrics import FrechetInceptionDistance as FID
from model import Generator, Discriminator, weights_init
from trainer import plot_generated_images


if __name__ == '__main__':
    models_weight_path = 'model_weights/'
    report_images_path = 'report_images/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_image_channels = 1

    gen = Generator(in_channels=100, out_channels=n_image_channels)
    try:
        gen.load_state_dict(torch.load(models_weight_path + 'generator.pt'))
    except:
        raise Exception(f'File with Generator weights generator.pt is not found in {models_weight_path}')

    disc = Discriminator(in_channels=n_image_channels)
    try:
        disc.load_state_dict(torch.load(models_weight_path + 'discriminator.pt'))
    except:
        raise Exception(f'File with Discriminator weights discriminator.pt is not found in {models_weight_path}')

    gen.to(device)
    disc.to(device)

    # we check the performance of your model on a fixed vector
    fixed_z_for_check = torch.load('fixed_20_z_for_check.pt').to(device)

    # assert to check the correctness of model implementation
    z = torch.rand((4, 100, 1, 1), device=device)
    fake_image = gen(z)
    assert list(fake_image.shape) == [4, n_image_channels, 32, 32]
    assert list(disc(fake_image).shape) == [4, 1, 1, 1]


    fake_images = gen.generate_images(fixed_z_for_check)
    plot_generated_images(fake_images,
                          results_path=report_images_path + f'ckeck_output_fixed_z_for_check.jpg', show=False)


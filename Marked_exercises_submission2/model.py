import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                batchnorm=True, activation=True):
        super().__init__()
        '''
        Initialise a class nested from a torch.Module (nn.Sequential). 
        This block is going to be used in Generator.
        Forward should sequentially call:
          1) Deconvolution
          2) BatchNorm2d if batchnorm is True
          3) ReLU activation if activation is True
        For deconvolution you can use ConvTranspose2d or Upsample + Conv2d
        Read about those operations.
        Information about the difference between them can be found here https://discuss.pytorch.org/t/upsample-conv2d-vs-convtranspose2d/138081/2
        '''
        layers = []
        ### BEGIN SOLUTION
        raise NotImplementedError()
        ### END SOLUTION
        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_block(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        '''
        in_channels = 100
        out_channels = number of output channels - 3 for RGB image and 1 for Greyscale
        
        Generator is constructed by stacking multiple GeneratorBlocks with different parameters, and Tanh activation before the final output. 
        
        You need to 
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels


        # compute what should be the size of kernels, strides and padding
        # kernel_sizes = [...]
        # stride_sizes = [...]
        # padding_sizes = [...]
        ### BEGIN SOLUTION
        raise NotImplementedError()
        ### END SOLUTION

        # filters: [1024, 512, 256]
        # You need to use four GeneratorBlock with provided in_channels, out_channels, batchnorm, activation
        # For the kernel_size, stride and padding you need to find appropriate values.
        # so that the final image output has a size (self.out_channels x 32 x 32)
        self.layers = nn.Sequential(
            # Z - input latent vector of dimensionality in_channels
            GeneratorBlock(in_channels=in_channels, out_channels=1024,
                           kernel_size=kernel_sizes[0], stride=stride_sizes[0], padding=padding_sizes[0],
                           batchnorm=True, activation=True),
            GeneratorBlock(in_channels=1024, out_channels=512,
                           kernel_size=kernel_sizes[1], stride=stride_sizes[1], padding=padding_sizes[1],
                           batchnorm=True, activation=True),
            GeneratorBlock(in_channels=512, out_channels=256,
                           kernel_size=kernel_sizes[2], stride=stride_sizes[2], padding=padding_sizes[2],
                           batchnorm=True, activation=True),
            GeneratorBlock(in_channels=256, out_channels=out_channels,
                           kernel_size=kernel_sizes[3], stride=stride_sizes[3], padding=padding_sizes[3],
                           batchnorm=False, activation=False)
            # output of self.layers --> Image (self.out_channels x 32 x 32)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layers(x)
        return self.tanh(x)

    def generate_images(self, x):
        fake_images = self.forward(x).detach().cpu().numpy()
        transformed_fake_images = []
        for img in fake_images:
            if self.out_channels == 3:
                transformed_fake_images.append(img.reshape(self.out_channels, 32, 32))
            else:
                transformed_fake_images.append(img.reshape(32, 32))
        return transformed_fake_images


class DiscriminatorBlock(nn.Module):
    '''

    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, instancenorm=True, activation=True):
        '''
        Inintialise a class nested from a torch.Module (nn.Sequential).
        Forward should sequentially call:
          1) Convolution
          2) InstanceNorm2d with learnable affine parameters (affine=True) if instancenorm is True
          3) LeakyReLU with 0.2 slope activation if activation is True. For LeakyRelU Set parameter inplace=True, this might speed up the training
        '''
        super().__init__()
        layers = []
        ### BEGIN SOLUTION
        raise NotImplementedError()
        ### END SOLUTION
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        '''
        in_channels: number of input channels - 3 for RGB image and 1 for Greyscale

        Discriminator is constructed by stacking multiple GeneratorBlocks with different parameters, and Tanh activation before the final output.

        '''
        super().__init__()
        self.in_channels = in_channels

        # out_channels = 1 - for real of fake classification
        # filters: [256, 512, 1024]
        self.layers = nn.Sequential(

            # Image (Cx32x32)
            DiscriminatorBlock(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1,
                               instancenorm=True, activation=True),
            # State (256x16x16)
            DiscriminatorBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, instancenorm=True,
                               activation=True),
            # State (512x8x8)
            DiscriminatorBlock(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1,
                               instancenorm=True, activation=True)
            # output of self.layers --> State (1024x4x4)
        )

        # Implemented the self.cls layer of the discriminator to compress the image into a suitable output for computing Wasserstein loss
        # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
        self.cls = DiscriminatorBlock(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0,
                                      instancenorm=False, activation=False)

    def forward(self, x):
        # input image of dimensionality (Cx64x64)
        x = self.layers(x)
        return self.cls(x)

import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm


'''
You need to implement simple 'Wasserstein GAN' https://arxiv.org/abs/1701.07875
'''

def set_model_require_grad(model, require_grad):
    '''
    require_grad: bool - specifies if all the parameters of the model should be set to require_grad=True or require_grad=False mode
    In GAN training you would have to set Discriminator and Generator parameters
    into requires_grad=True and requires_grad=False mode.
    A function that goes through every parameter of a model and sets them to parameter.requires_grad=requires_grad mode
    '''
    for param in model.parameters():
        param.requires_grad = require_grad #fixed to param.requires_grad = ... which specifies if the gradients should be computed for the Tensor,
        #previously was a typo param.require_grad = ...


def plot_generated_images(images, results_path=None, show=True):
    '''
    plot 5 generated images in a row

    :param images:  images to plot
    :param results_path: if not None saves images to results_path
    :param show: if True shows the images with plt.show()
    '''

    y_dim = 5
    x_dim = int(np.ceil(len(images) / y_dim))

    fig, axes = plt.subplots(x_dim, y_dim, figsize=(x_dim * 6, y_dim * 6))
    axes = axes.flatten()

    for i in range(len(images)):
        image = images[i]
        #mean and std are taken from the dataset
        mean, std = np.array([0.5]), np.array([0.5])
        image = image * np.expand_dims(std, axis=(1, 2))
        image = image + np.expand_dims(mean, axis=(1, 2))

        axes[i].imshow(np.transpose(image, (1, 2, 0)), cmap='gray')
        axes[i].axis('off')
    if results_path is not None:
        plt.savefig(results_path)
    if show:
        plt.show()
    else:
        plt.clf()



class WGANTrainer():

    def __init__(self, model_gen, model_disc, optimizer_gen, optimizer_disc,
                 n_disc_steps=1, weight_cliping=0.001, device=torch.device('cpu')):
        self.model_gen, self.model_disc = model_gen, model_disc
        self.optimizer_gen, self.optimizer_disc = optimizer_gen, optimizer_disc

        self.weight_cliping = weight_cliping # parameter for weights clamping in clamp_weights

        '''
        GAN training implies mini-max game. To balance the power/optimization of generator and
        and discriminator you might want to update generator every n_disc_steps or discriminator, so that 
        When the discriminator is near optimality, the ratio between densities (𝑝_𝑑𝑎𝑡𝑎 and 𝑝_𝑚𝑜𝑑𝑒𝑙) is more accurate,
         generating better gradients to update the generator. 
        If you choose to do the other way around, you would be updating the generator successively with poor gradients.
        https://ai.stackexchange.com/questions/35768/why-do-we-train-the-discriminators-k-times-but-train-the-generator-only-1-time-i
        '''
        self.n_disc_steps = n_disc_steps

        self.device = device

        self.disc_loss_log = [] # variables to log train losses
        self.gen_loss_log = []

        # we fix latent vectors so that during training you can see the progress of generator on the same latents
        self.fixed_z_for_check = torch.load('fixed_20_z_for_check.pt').to(self.device)
        self.fixed_z_for_eval = self.fixed_z_for_check[:5]

        self.model_weights_path = 'model_weights/' #path where the models are going to be saved
        if not os.path.exists(self.model_weights_path):
            os.mkdir(self.model_weights_path)

        self.report_images_path = 'report_images/' #path where the report images are going to be saved
        if not os.path.exists(self.report_images_path):
            os.mkdir(self.report_images_path)

    def save_models(self, epoch=None):
        #if you want to save the weights for every epoch separately you should pass the number of epoch in the function
        # if you set epoch=None the new weights will override previous weights and will be saved in 'generator.pt' and 'discriminator.pt'
        # make sure to save the final weights into 'generator.pt' and 'discriminator.pt' files
        if epoch is None:
            torch.save(self.model_gen.state_dict(), self.model_weights_path + 'generator.pt')
            torch.save(self.model_disc.state_dict(), self.model_weights_path + 'discriminator.pt')
        else:
            torch.save(self.model_gen.state_dict(), self.model_weights_path + f'generator_ep{epoch}.pt')
            torch.save(self.model_disc.state_dict(), self.model_weights_path + f'discriminator_ep{epoch}.pt')

    def plot_loss(self):
        #ploting the whole loss starting from the beginning of training
        plt.figure(figsize=(5, 5))
        plt.plot(self.disc_loss_log, c='blue', label='discriminator')
        plt.scatter(np.arange(len(self.gen_loss_log)) * self.n_disc_steps, self.gen_loss_log,
                    marker='+', linewidths=2, c='orange', label='generator')
        plt.title('training loss')

    def clamp_weights(self):
        '''
        In Theorem 3 in Wasserstain GAN paper https://arxiv.org/pdf/1701.07875 to ensure
        that all the functions in critic will be K-Lipschitz and the space is compact
        weights should be clamped to a fixed box (W = [−0.01, 0.01]^l) after each gradient update
        '''
        ### BEGIN SOLUTION
        for param in self.model_disc.parameters():
            param.data.clamp_(-self.weight_cliping, self.weight_cliping)
        ### END SOLUTION

    def gen_step(self,
                 ### BEGIN SOLUTION
                 noise,
                 ### END SOLUTION
                 ):
        '''
        1) clean generator gradients
        2) compute loss: you will need to call critic from fake images
        3) don't forget to do compute gradients and update weights
        4) already implemented: logging of the loss for printing

        In this function only generator weights should be updated.
        Make sure that your discriminator weights stay the same and don't change
        '''

        # loss_gen = ...
        ### BEGIN SOLUTION
        # 1
        self.optimizer_gen.zero_grad()
        
        # 2
        # not sure about that
        fake_images = self.model_gen(noise)
        #fake_image = self.model_gen.generate_images(noise)
        #fake_images_tensor = torch.tensor(fake_image, device=self.device)
        #fake_images_tensor = fake_images_tensor.unsqueeze(0)
        #fake_images_tensor = fake_images_tensor.permute(1, 0, 2, 3)
        # fake_image = self.model_gen.generate_images(noise) # ????? pas du bruit ??? OU LE BRUIT
        # fake_images_for_discriminator = torch.tensor(transformed_fake_images, dtype=torch.float32, device=device) # do I need to transform the fake_image into a type the discriminator can understand or not ?
        # Evaluate the critic (discriminator) on real and fake images
        # with torch.no_grad(): or not ?
        critic_fake_output = self.model_disc(fake_images)
        # not sure
        loss_gen = - torch.mean(critic_fake_output)
        
        # 3
        loss_gen.backward() # make sure not updating disc weights ?? with torch.no_grad sur l'autre ?  je me dem si pas déjà fait dans train_epoch qd il fait require_grad
        self.optimizer_gen.step()
        # self.clamp_weights() # NOT ASKED IN THE DONNéE HERE ????? Let it ?
        
        # 4
        ### END SOLUTION

        self.gen_loss_log.append(loss_gen.item())

    def disc_step(self,
                  ### BEGIN SOLUTION
                  real_images,
                  noise,
                  ### END SOLUTION
                  ):
        '''
        1) clean the gradients
        2) compute wasserstain loss: you will need to call critic from fake and real images
        3) don't forget to do compute gradients, update and clamp weights
        4) already implemented: log the loss for printing

        In this function only discriminator weights should be updated.
        Make sure that your generator weights stay the same and don't change
        '''

        # loss_disc = ...

        ### BEGIN SOLUTION
        # MAYBE BE CAREFUL TO THE LAST COMM TELLING THAT DISC WEIGHTS SHOULD BE UPDATED BUT NOT GEN WEIGHTS
        # 1
        self.optimizer_disc.zero_grad()
        
        # 2

        # EN FAIT GENERATE_IMAGES RENVOIT 64 IMAGES NON ???? Peut aller dans model.py et faire [0] dans le return transformed image comme ça on gagne du temps sur les itérations juste pour tester
        #fake_image = self.model_gen.generate_images(noise) # ????? z = bruit ???
# 64 size before
        #fake_images_tensor = torch.tensor(fake_image, device=self.device)
        # 64 x 32 x 32 after tensor
        #fake_images_tensor = fake_images_tensor.unsqueeze(0) # needed because generate_images return 32 x 32 if no color and not 1 x 32 x 32. (3 x 32 x 32 is okey i guess with colors. then will require a if statement for unsqueeze if grey scale and not if color)
        # 1 x 64 x 32 x 32 after unsqueeze (maybe won't be needed for colours)
        #fake_images_tensor = fake_images_tensor.permute(1, 0, 2, 3)
        # real_images and fake_images_tensor have 64, 1, 32, 32 size after, before sending to disc. Then is it just 64 images of each ??? Why many at the same time

        # fake_images_for_discriminator = torch.tensor(transformed_fake_images, dtype=torch.float32, device=device) # do I need to transform the fake_image into a type the discriminator can understand or not ?
        
        # Evaluate the critic (discriminator) on real and fake images
        # with torch.no_grad(): or not ?????
        fake_images = self.model_gen(noise)
        
        critic_fake_output = self.model_disc(fake_images)
        critic_real_output = self.model_disc(real_images)
        # not sure
        loss_disc = torch.mean(critic_fake_output) - torch.mean(critic_real_output)
        
        # 3
        loss_disc.backward() # need to ensure to update of weights of gen ? je me dem si pas déjà fait dans train_epoch qd il fait require_grad
        self.optimizer_disc.step()
        self.clamp_weights()
        
        # 4
        ### END SOLUTION
        self.disc_loss_log.append(loss_disc.item())

    def train_epoch(self, train_loader):

        set_model_require_grad(self.model_disc, True)
        set_model_require_grad(self.model_gen, False)
        
        for batch_num, (real_images, _) in tqdm(enumerate(train_loader)):
            real_images = real_images.to(self.device)
            z = torch.rand((real_images.shape[0], 100, 1, 1), device=self.device)
            # print("Working on batch number : ", batch_num+1, "/", len(train_loader)) # JUST FOR ME RN

            # pass the correct attributed into the disc_step
            self.disc_step(
                ### BEGIN SOLUTION
                real_images, # SHOULDNT NOISE Z BE GAUSSIAN = RANDN INSTEAD OF RAND ?
                noise = z,
                ### END SOLUTION
            )
            # optimization generator every n_disc_steps
            if batch_num % self.n_disc_steps == 0:
                # print("Also optimize on generator this time as batch number is pair") # JUST FOR ME RN
                # this should speed up the training, so that redundunt gradients are not computed
                set_model_require_grad(self.model_disc, False)
                set_model_require_grad(self.model_gen, True)

                # pass the correct attributed into the gen_step
                self.gen_step(
                    ### BEGIN SOLUTION
                    noise = z
                    ### END SOLUTION
                )
                set_model_require_grad(self.model_disc, True)
                set_model_require_grad(self.model_gen, False)

    def train(self, n_epoches, train_loader):
        len_loader = len(train_loader)

        for epoch in range(n_epoches):
            self.model_gen.train()
            self.model_disc.train()

            self.train_epoch(train_loader)

            # save the current weights of the models in 'generator.pt' and 'discriminator.pt' files
            self.save_models(epoch=None)

            # print average of losses computed during current epoch
            print(f'Epoch {epoch+1}/{n_epoches}: gen_loss {np.mean(self.gen_loss_log[-len_loader:])},'
                  f' disc_loss {np.mean(self.disc_loss_log[-len_loader:])}')
            # plot and save images of losses that have been computed for every batch since the beginning of training
            self.plot_loss()
            plt.savefig(self.report_images_path + f'losses_ep{epoch}.jpg')
            plt.show()

            #plot images
            self.model_gen.eval()
            self.model_disc.eval()
            fake_images = self.model_gen.generate_images(self.fixed_z_for_eval)
            plot_generated_images(fake_images, results_path=None, show=True)

            fake_images = self.model_gen.generate_images(self.fixed_z_for_check)
            plot_generated_images(fake_images,
                                  results_path=self.report_images_path + f'fixed_z_for_check_ep{epoch}.jpg', show=False)


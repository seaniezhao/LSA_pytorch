import torch
from model.modules import *
from model.MuseGAN import MuseGAN
import torch.autograd as autograd
import time

class MuseGANTrainer:
    def __init__(self, device, z_inter_dim, z_intra_dim, track_dim, lmbda, print_batch = True):
        """ Object to hold data iterators, train the model """

        self.__dict__.update(locals())

        self.museGan = MuseGAN(track_dim, z_inter_dim, z_intra_dim).to(device)
        self.discriminator = BarDiscriminator().to(device)


    def generate_inter_intra(self, batch_size):
        z_inter = torch.normal(torch.zeros(batch_size, self.z_inter_dim), 0.1).to(self.device)
        z_intra = torch.normal(torch.zeros(batch_size, self.z_intra_dim, self.track_dim), 0.1).to(self.device)

        return z_inter, z_intra


    def train(self, train_iter, num_epochs, model_path='', lr=1e-3):


        optimizerG = torch.optim.Adam(self.museGan.parameters(), lr=lr)
        optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        counter = 0
        for epoch in range(num_epochs):

            print('{:-^120}'.format(' Epoch {} Start '.format(epoch)))
            num_batch = len(train_iter)
            for batch_idx, batch in enumerate(train_iter):

                batch = (batch[0].to(self.device), batch[1].to(self.device))

                batch_start_time = time.time()

                num_iters_D = 100 if counter < 25 or counter % 500 == 0 else 5

                for j in range(num_iters_D):

                    optimizerD.zero_grad()
                    d_loss = self.train_D(batch)

                    d_loss.backward()
                    optimizerD.step()

                optimizerG.zero_grad()
                g_loss = self.train_G(batch)

                g_loss.backward()
                optimizerG.step()

                if self.print_batch:
                    print('---{}--- epoch: {:2d} | batch: {:4d}/{:4d} | time: {:6.2f} ' \
                          .format('test', epoch,
                                  batch_idx, num_batch, time.time() - batch_start_time))
                    print('D loss: %6.2f, G loss: %6.2f' % (g_loss.item(), d_loss.item()))

                counter += 1

        print('{:=^120}'.format(' Training End '))

    def gradient_penalty(self, real, fake, layer_conditions):
        batch_size = real.shape[0]

        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real.nelement()/batch_size)).contiguous().\
            view(batch_size, real.shape[1], real.shape[2], real.shape[3]).to(self.device)

        interpolates = alpha * real + ((1 - alpha) * fake).to(self.device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.discriminator(interpolates, layer_conditions)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lmbda

        return gradient_penalty

    def train_D(self, batch):
        for param in self.discriminator.parameters():
            param.requires_grad = True

        images, conditions = batch
        batch_size = images.shape[0]
        z_tuple = self.generate_inter_intra(batch_size)

        G_output, layer_conditions = self.museGan(z_tuple, conditions)
        G_output = G_output.detach()

        DX_score = self.discriminator(images, layer_conditions)
        DG_score = self.discriminator(G_output, layer_conditions)

        gp = self.gradient_penalty(images, G_output, layer_conditions)


        d_loss = torch.mean(DG_score) - torch.mean(DX_score)
        d_loss += gp


        return d_loss

    def train_G(self, batch):
        # to avoid extra computation
        for param in self.discriminator.parameters():
            param.requires_grad = False

        images, conditions = batch
        batch_size = images.shape[0]
        z_tuple = self.generate_inter_intra(batch_size)


        G_output, layer_conditions = self.museGan(z_tuple, conditions)

        DG_score = self.discriminator(G_output, layer_conditions)

        g_loss = -torch.mean(DG_score)

        return g_loss



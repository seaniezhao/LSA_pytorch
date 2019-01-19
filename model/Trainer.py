import torch
from model.modules import *


class MuseGANTrainer:
    def __init__(self, device, z_intra_dim, z_inter_dim, track_dim):
        """ Object to hold data iterators, train the model """

        self.__dict__.update(locals())

        self.cEncoder = BarEncoder().to(device)
        self.generator = SumGenerator().to(device)
        self.discriminator = BarDiscriminator().to(device)


    def generate_inter_intra(self):
        pass

    def train(self, train_iter, val_iter, num_epochs, model_path='', lr=1e-3):

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        #for epoch in range(num_epochs):


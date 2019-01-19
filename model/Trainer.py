import torch
from model.modules import *


class MuseGANTrainer:
    def __init__(self, device):
        """ Object to hold data iterators, train the model """
        self.cEncoder = BarEncoder().to(device)
        self.generator = SumGenerator().to(device)
        self.discriminator = BarDiscriminator().to(device)



    def train(self, train_iter, val_iter, num_epochs, model_path='', lr=1e-3):

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

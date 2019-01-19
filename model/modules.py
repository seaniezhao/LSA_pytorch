import torch.nn as nn
import torch.nn.functional as F
import torch, torchvision
from model.ops import *

class BarEncoder(nn.Module):
    # chroma sequence
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 1), stride=(3, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
        )
    def forward(self, condition):
        #condition shape: (?, 1, 48, 12)
        h0 = condition
        h1 = self.layer1(h0)  #(?, 16, 16, 12)
        h2 = self.layer2(h1)  #(?, 16, 8, 12)
        h3 = self.layer2(h1)  #(?, 16, 4, 12)
        h4 = self.layer2(h1)  #(?, 16, 2, 12)
        h5 = self.layer2(h1)  #(?, 16, 1, 12)

        return [h0, h1, h2, h3, h4, h5]

#单轨单小节生成器
class BarGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 1), stride=(3, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)


        )



    def forward(self, z, condition):
        h0 = z.view(-1, z.size(1), 1, 1)

        return h0

#全部轨单小节生成
class SumGenerator(nn.Module):
    def __init__(self, track_dim):
        super().__init__()

        self.generator = []
        for idx in range(track_dim):
            bg = BarGenerator()
            self.generator.append(bg)

    def forward(self, *input):

        n_track_data = 0
        return n_track_data

class BarDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 128, (1, 7), stride=(1, 7)),
            nn.LeakyReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(129, 128, (3, 1), stride=(3, 1)),
            nn.LeakyReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(144, 128, (2, 1), stride=(2, 1)),
            nn.LeakyReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(144, 128, (2, 1), stride=(2, 1)),
            nn.LeakyReLU(True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(144, 256, (2, 1), stride=(2, 1)),
            nn.LeakyReLU(True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(272, 512, (2, 1), stride=(2, 1)),
            nn.LeakyReLU(True)
        )

        self.linear1 = nn.Linear()

    def forward(self, barData, barCondi):
        # barData shape (?, 48, 84, 5)

        h0 = self.layer1(barData) #(?, 48, 12, 128)

        h1 = concat_prev(h0, barCondi[0] if barCondi else None) #(?, 48, 12, 129)
        h1 = self.layer2(h1)  # (?, 16, 12, 128)

        h2 = concat_prev(h1, barCondi[1] if barCondi else None) #(?, 16, 12, 144)
        h2 = self.layer3(h2)  # (?, 8, 12, 128)

        h3 = concat_prev(h2, barCondi[2] if barCondi else None) #(?, 8, 12, 144)
        h3 = self.layer4(h3)  # (?, 4, 12, 128)

        h4 = concat_prev(h3, barCondi[3] if barCondi else None) #(?, 4, 12, 144)
        h4 = self.layer5(h4)  # (?, 2, 12, 256)

        h5 = concat_prev(h4, barCondi[4] if barCondi else None) #(?, 2, 12, 272)
        h5 = self.layer6(h5)  # (?, 1, 12, 512)

        h6 = concat_prev(h5, barCondi[5] if barCondi else None) #(?, 2, 12, 272)
        h6.view(barData.size(0), -1)
        h6 = self.layer6(h1)  # (?, 1, 12, 512)

        h7 = 0

        return h5, h7
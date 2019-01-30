import torch.nn as nn
import torch.nn.functional as F
import torch, torchvision
from model.ops import *

class BarEncoder_ChromaSeq(nn.Module):
    # chroma sequence
    def __init__(self):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 1), stride=(3, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self, condition):
        # condition shape: (?, 1, 48, 12)
        h0 = condition
        h1 = self.layer0(h0)  # (?, 16, 16, 12)
        h2 = self.layer1(h1)  # (?, 16, 8, 12)
        h3 = self.layer2(h2)  # (?, 16, 4, 12)
        h4 = self.layer3(h3)  # (?, 16, 2, 12)
        h5 = self.layer4(h4)  # (?, 16, 1, 12)

        return [h0, h1, h2, h3, h4, h5]


class BarEncoder_ChordSeq(nn.Module):
    # chroma sequence
    def __init__(self):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 12), stride=(1, 12)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 7), stride=(1, 7)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 1), stride=(3, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, condition):
        # condition shape: (?, 1, 48, 84)
        h0 = self.layer0(condition) # (?, 16, 48, 7)
        h1 = self.layer1(h0)  # (?, 16, 48, 1)
        h2 = self.layer2(h1)  # (?, 16, 16, 1)
        h3 = self.layer3(h2)  # (?, 16, 8, 1)
        h4 = self.layer4(h3)  # (?, 16, 4, 1)
        h5 = self.layer5(h4)  # (?, 16, 2, 1)

        return [h0, h1, h2, h3, h4, h5, condition]


# 单轨单小节生成器
class BarGenerator_ChromaSeq(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (1, 12), stride=(1, 12), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(528, 256, (2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(272, 256, (2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(272, 128, (2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(144, 128, (2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(144, 64, (3, 1), stride=(3, 1), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(65, 1, (1, 7), stride=(1, 7), padding=0),
        )

    def forward(self, z, barCondi):
        h0 = z.view(-1, z.size(1), 1, 1)
        h0 = self.layer0(h0)

        h1 = self.layer1(h0)  # (?, 512, 1, 12)

        h2 = concat_prev(h1, barCondi[5])  # (?, 528, 1, 12)
        h2 = self.layer2(h2)  # (?, 256, 2, 12)

        h3 = concat_prev(h2, barCondi[4])  # (?, 272, 2, 12)
        h3 = self.layer3(h3)  # (?, 256, 4, 12)

        h4 = concat_prev(h3, barCondi[3])  # (?, 272, 4, 12)
        h4 = self.layer4(h4)  # (?, 128, 8, 12)

        h5 = concat_prev(h4, barCondi[2])  # (?, 144, 8, 12)
        h5 = self.layer5(h5)  # (?, 128, 16, 12)

        h6 = concat_prev(h5, barCondi[1])  # (?, 144, 16, 12)
        h6 = self.layer6(h6)  # (?, 64, 48, 12)

        h7 = concat_prev(h6, barCondi[0])  # (?, 65, 48, 12)
        h7 = self.layer7(h7)  # (?, 1, 48, 84)

        return torch.tanh(h7)


class BarGenerator_ChordSeq(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(528, 512, (2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(528, 256, (2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(272, 256, (2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(272, 128, (3, 1), stride=(3, 1), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(144, 64, (1, 7), stride=(1, 7), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(80, 1, (1, 12), stride=(1, 12), padding=0),
        )

    def forward(self, z, barCondi):
        h0 = z.view(-1, z.size(1), 1, 1)  # (?, z_dim, 1, 1)
        h0 = self.layer0(h0)  # (?, 1024, 1, 1)

        h1 = h0.view(-1, 512, 2, 1)  # (?, 512, 2, 1)
        h1 = concat_prev(h1, barCondi[5])  # (?, 528, 2, 1)
        h1 = self.layer1(h1)  # (?, 512, 4, 1)

        h2 = concat_prev(h1, barCondi[4])  # (?, 528, 4, 1)
        h2 = self.layer2(h2)  # (?, 256, 8, 1)

        h3 = concat_prev(h2, barCondi[3])  # (?, 272, 8, 1)
        h3 = self.layer3(h3)  # (?, 256, 16, 1)

        h4 = concat_prev(h3, barCondi[2])  # (?, 272, 16, 1)
        h4 = self.layer4(h4)  # (?, 128, 48, 1)

        h5 = concat_prev(h4, barCondi[1])  # (?, 144, 48, 1)
        h5 = self.layer5(h5)  # (?, 64, 48, 7)

        h6 = concat_prev(h5, barCondi[0])  # (?, 80, 48, 7)
        h6 = self.layer6(h6)  # (?, 1, 48, 84)

        return torch.tanh(h6)



class BarDiscriminator_ChromaSeq(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(5, 128, (1, 7), stride=(1, 7)),
            nn.LeakyReLU(0.2, True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(129, 128, (3, 1), stride=(3, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(144, 128, (2, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(144, 128, (2, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(144, 256, (2, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(272, 512, (2, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.linear0 = nn.Sequential(
            nn.Linear(6336, 1024),
            nn.LeakyReLU(0.2, True)
        )

        self.linear1 = nn.Linear(1024, 1)

    def forward(self, barData, barCondi):
        # barData shape (?, 5, 48, 84)

        h0 = self.layer0(barData)  # (?, 128, 48, 12)

        h1 = concat_prev(h0, barCondi[0])  # (?, 129, 48, 12)
        h1 = self.layer1(h1)  # (?, 128, 16, 12)

        h2 = concat_prev(h1, barCondi[1])  # (?, 144, 16, 12)
        h2 = self.layer2(h2)  # (?, 128, 8, 12)

        h3 = concat_prev(h2, barCondi[2])  # (?, 144, 8, 12)
        h3 = self.layer3(h3)  # (?, 128, 4, 12)

        h4 = concat_prev(h3, barCondi[3])  # (?, 144, 4, 12)
        h4 = self.layer4(h4)  # (?, 256, 2, 12)

        h5 = concat_prev(h4, barCondi[4])  # (?, 272, 2, 12)
        h5 = self.layer5(h5)  # (?, 512, 1, 12)

        h6 = concat_prev(h5, barCondi[5])  # (?, 528, 2, 12)
        h6 = h6.view(barData.size(0), -1)
        h6 = self.linear0(h6)  # (?, 1024)

        h7 = self.linear1(h6)  # (?, 1)
        return h7


class BarDiscriminator_ChordSeq(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(6, 128, (1, 12), stride=(1, 12)),  # (?, 128, 48, 7)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, (1, 7), stride=(1, 7)),  # (?, 128, 48, 1)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, (2, 1), stride=(2, 1)),  # (?, 128, 24, 1)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, (2, 1), stride=(2, 1)),  # (?, 128, 12, 1)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (4, 1), stride=(2, 1)),  # (?, 256, 5, 1)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 1), stride=(2, 1)),  # (?, 512, 2, 1)
            nn.LeakyReLU(0.2, True),
        )

        self.linear0 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, True)
        )

        self.linear1 = nn.Linear(1024, 1)

    def forward(self, barData, barCondi):
        # barData shape (?, 5, 48, 84)


        h0 = concat_prev(barData, barCondi[6])  # (?, 6, 48, 84)

        h1 = self.conv(h0)
        h1 = h1.view(barData.shape[0], -1)
        h2 = self.linear0(h1)  # (?, 1024)

        out = self.linear1(h2)

        return out

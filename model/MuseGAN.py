import torch.nn as nn
from model.modules import *

class MuseGAN(nn.Module):
    # chroma sequence
    def __init__(self, trainer_type, track_dim, z_inter_dim, z_intra_dim):
        super().__init__()

        self.track_dim = track_dim

        if trainer_type == 0:
            self.cEncoder = BarEncoder_ChordSeq()
            self.generators = nn.ModuleList(
                [BarGenerator_ChordSeq(z_inter_dim + z_intra_dim) for _ in range(track_dim)])
        elif trainer_type == 1:
            self.cEncoder = BarEncoder_ChromaSeq()
            self.generators = nn.ModuleList([BarGenerator_ChromaSeq(z_inter_dim + z_intra_dim) for _ in range(track_dim)])


    def forward(self, z_tuple, condition):
        z_inter, z_intra = z_tuple
        layer_conditions = self.cEncoder(condition)

        n_track_data = []
        for idx in range(self.track_dim):
            z = torch.cat((z_inter, z_intra[:, :, idx]), dim=1)
            out_put = self.generators[idx](z, layer_conditions)
            n_track_data.append(out_put)
        rt = torch.cat(n_track_data, dim=1)

        return rt, layer_conditions

class Discrimitor(nn.Module):

    def __init__(self, trainer_type):
        super().__init__()

        if trainer_type == 0:
            self.model = BarDiscriminator_ChordSeq()
        elif trainer_type == 1:
            self.model = BarDiscriminator_ChromaSeq()

    def forward(self, barData, condition):

        score = self.model(barData, condition)

        return score

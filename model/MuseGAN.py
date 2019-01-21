import torch.nn as nn
from model.modules import *

class MuseGAN(nn.Module):
    # chroma sequence
    def __init__(self, track_dim, z_inter_dim, z_intra_dim):
        super().__init__()

        self.cEncoder = BarEncoder()
        self.track_dim = track_dim
        self.generators = nn.ModuleList([BarGenerator(z_inter_dim+z_intra_dim) for _ in range(track_dim)])


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


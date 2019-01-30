from model.Trainer import MuseGANTrainer
from data.dataset import get_my_data
import torch



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0: chord sequence, 1: chroma sequence
    trainer_type = 0

    train_iter, val_iter = get_my_data(64, trainer_type)

    # device, z_intra_dim, z_inter_dim, track_dim, lmbda,
    trainer = MuseGANTrainer(trainer_type, device, 64, 64, 5, 10)

    trainer.train(train_iter, 30)


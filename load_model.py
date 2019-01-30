from model.Trainer import MuseGANTrainer
from data.dataset import get_song_condition
import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0: chord sequence, 1: chroma sequence
    trainer_type = 0

    # device, z_intra_dim, z_inter_dim, track_dim, lmbda,
    trainer = MuseGANTrainer(trainer_type, device, 64, 64, 5, 10)

    trainer.load_model('checkpoints/type0_checkpoint_25.ckpt')

    conditions = get_song_condition('slam_dunk_y.npy')
    conditions = torch.Tensor(conditions).to(device)
    trainer.eval_sampler(conditions)

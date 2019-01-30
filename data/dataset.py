import torch
import os
import torch.utils.data
import numpy as np


class LSADataset(torch.utils.data.Dataset):
    def __init__(self, path):

        self.data_X = np.load(os.path.join(path, 'x_bar_chroma.npy'))
        self.data_y = np.load(os.path.join(path, 'y_bar_chroma.npy'))

        # (?, 48, 84, 5) --> (?, 5, 48, 84)
        self.data_X = np.transpose(self.data_X, [0, 3, 1, 2])
        # (?, 48, 12, 1) --> (?, 1, 48, 12)
        self.data_y = np.transpose(self.data_y, [0, 3, 1, 2])

        pass

    def __getitem__(self, index):


        return (self.data_X[index].astype(np.float32)*2 -1, self.data_y[index].astype(np.float32)*2 -1)

    def __len__(self):

        return len(self.data_y)



def get_my_data(BATCH_SIZE=64, trainer_type=0):
    """ Load music data """

    t_path = '/home/sean/pythonProj/LSArrangement/LSA_pytorch/data/chord_sequence/tra'
    v_path = '/home/sean/pythonProj/LSArrangement/LSA_pytorch/data/chord_sequence/val'

    if trainer_type == 1:  # chroma sequence
        t_path = '/home/sean/pythonProj/LSArrangement/LSA_pytorch/data/chroma_sequence/tra'
        v_path = '/home/sean/pythonProj/LSArrangement/LSA_pytorch/data/chroma_sequence/val'

    train_iter = torch.utils.data.DataLoader(
        LSADataset(t_path),
        batch_size=BATCH_SIZE, shuffle=True,num_workers=10)

    val_iter = torch.utils.data.DataLoader(
        LSADataset(v_path),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    return train_iter, val_iter

def get_song_condition(path):

    data_y = np.load(path)
    #data_y = np.expand_dims(data_y, axis=3)
    data_y = np.transpose(data_y, [0, 3, 1, 2])

    return data_y.astype(np.float32)*2 - 1


if __name__ == '__main__':
    get_my_data()
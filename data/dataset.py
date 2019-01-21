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

    def __getitem__(self, index):


        return (self.data_X[index].astype(np.float32)*2 -1, self.data_y[index].astype(np.float32)*2 -1)

    def __len__(self):

        return len(self.data_y)



def get_my_data(BATCH_SIZE=32):
    """ Load music data """

    train_iter = torch.utils.data.DataLoader(
        LSADataset('/home/sean/pythonProj/LSArrangement/LSA_pytorch/data/chroma_sequence/tra'),
        batch_size=BATCH_SIZE, shuffle=True,num_workers=10)

    val_iter = torch.utils.data.DataLoader(
        LSADataset('/home/sean/pythonProj/LSArrangement/LSA_pytorch/data/chroma_sequence/val'),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=10)


    return train_iter, val_iter


if __name__ == '__main__':
    get_my_data()
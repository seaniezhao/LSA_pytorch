import torch
import os
import torch.utils.data
import numpy as np


class LSADataset(torch.utils.data.Dataset):
    def __init__(self, path):

        self.data_X = np.load(os.path.join(path, 'x_bar_chroma.npy'))
        self.data_y = np.load(os.path.join(path, 'y_bar_chroma.npy'))


    def __getitem__(self, index):



        return (self.data_X[index], self.data_y[index])

    def __len__(self):

        return len(self.data_X)



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
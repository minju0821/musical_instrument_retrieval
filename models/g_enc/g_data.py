import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa


class InstrumentReprDataset(Dataset):
    def __init__(self, data_path='/home/haessun/ai_prod/inst_repr_lib'):
        self.data_path = data_path
        self.data_list = glob.glob(data_path + '/*.pt')
        self.data_list.sort()

        # /home/haessun/ai_prod/inst_repr_lib/1.pt

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        repr_path = self.data_list[idx]
        repr = torch.load(repr_path)

        inst_num = int(repr_path.split('/')[-1][:-3])
        return repr, inst_num
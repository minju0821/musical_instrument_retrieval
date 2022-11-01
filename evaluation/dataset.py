import os
import glob
import random

import torch
import numpy as np
from torch.utils.data import Dataset

random.seed(0)
torch.manual_seed(0)

SAMPLING_RATE = 16000

""" customized for evaluation"""
class RenderedNlakhDataset(Dataset):
    def __init__(self, data_path = None, split = None):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.dir_list = sorted(glob.glob(os.path.join(self.data_path, self.split, "*")))
        self.pad_num = 0.

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        dir_path = self.dir_list[idx]
        fname_list = sorted(glob.glob(os.path.join(dir_path, "*.npy")))

        # emb_list = torch.full((9, 1024), self.pad_num)
        emb_idx_list = torch.full((9,), -1, dtype=torch.int32)
        emb_list_idx = 0
        for fname in fname_list:
            
            if 'mix_audio' in fname or 'summed' in fname:
                continue

            if 'mix' in fname:
                mix = np.load(fname)
                mix = torch.tensor(mix, dtype=torch.float32)
            else:
                # emb_list[emb_list_idx] = torch.from_numpy(np.load(fname))
                emb_idx_list[emb_list_idx] = int(fname.split('/')[-1][:-8])
                emb_list_idx += 1

        emb_list_idx = torch.tensor(emb_list_idx, dtype=torch.int32)
        # return mix, emb_list, emb_idx_list, emb_list_idx
        return mix, emb_idx_list

"""
- for all Nlakh data of single instrument, process the audio with trained single_instrument_encoder and save it in the form of .npy
- this dataset returns the embedding of single instrument of given index for each instrument class
- ex) get_idx = 1 -> returns only 0001.npy in every single instrument class embeddings
"""
class EmbeddingLibraryDataset(Dataset):
    def __init__(self, path = None, split = 'valid', get_idx = None):
        self.dir_list = sorted(glob.glob(f'{path}/{split}/*'))
        self.get_idx = get_idx

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        file_list = sorted(glob.glob(f'{self.dir_list[idx]}/*.npy'))
        emb = np.load(file_list[self.get_idx-1])
        return emb
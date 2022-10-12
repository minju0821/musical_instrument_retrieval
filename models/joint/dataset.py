from json import load
import os
import glob
import numpy as np
import librosa
import torch
import random
from torch.utils.data import Dataset

SAMPLING_RATE = 16000

class EmbeddingDataset_before(Dataset):
    def __init__(self, data_path = "/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3", split = None):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.dir_list = glob.glob(os.path.join(self.data_path, self.split, "*"))
        self.dir_list.sort()

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        dir_path = self.dir_list[idx]
        fname_list = glob.glob(os.path.join(dir_path, "*.npy"))
        embedding_list = []

        for fname in fname_list:
            embedding = np.load(fname)
            embedding = torch.from_numpy(embedding)
            embedding_list.append(embedding)

        embedding_sum = torch.sum(torch.stack(embedding_list), dim=0)
        return embedding_sum, embedding_list
    
    def collate_fn(self, batch):
        embedding_sum = torch.stack([item[0][0] for item in batch])
        
        max_tracks = max([len(item[1]) for item in batch])
        target_embedding = torch.full((len(batch), max_tracks, 1024), 1e8)  # Fill the rest with infinity.
        for i, item in enumerate(batch):
            for j, track in enumerate(item[1]):
                target_embedding[i, j, :] = track
        return embedding_sum, target_embedding

class EmbeddingDataset(Dataset):
    def __init__(self, data_path = "/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb/", split = None, loss = 'MSE', batch_size = 32):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.single_list = glob.glob(os.path.join(self.data_path, f'{self.split}_single', '*'))
        self.single_list.sort()
        self.dir_list = glob.glob(os.path.join(self.data_path, self.split, "*"))
        self.dir_list.sort()

        if loss == 'MSE':
            self.pad_num = 1e8
        elif loss == 'Cosine':
            self.pad_num = 0.
        self.batch_size = batch_size

    def __len__(self):
        # return 320
        return len(self.dir_list)

    def __getitem__(self, idx):
        dir_path = self.dir_list[idx]
        fname_list = glob.glob(os.path.join(dir_path, "*.npy"))
        fname_list.sort()

        emb_list = torch.full((9, 1024), self.pad_num)
        emb_list_len = 0
        emb_idxes = []
        for fname in fname_list:
            if 'mix_audio' in fname:
                mix_audio = np.load(fname)
                mel_spec = librosa.feature.melspectrogram(y=mix_audio, sr=SAMPLING_RATE, win_length=1024, hop_length=512, n_mels=128)
                log_spec = librosa.power_to_db(mel_spec)
                mix_audio = torch.tensor(log_spec, dtype=torch.float32)
            elif 'mix' in fname:
                mix = np.load(fname)
                mix = torch.tensor(mix, dtype=torch.float32)
            elif 'summed' in fname:
                summed = np.load(fname)
                summed = torch.tensor(summed, dtype=torch.float32)
            else:
                emb = np.load(fname)
                emb = torch.from_numpy(emb)
                emb_list[emb_list_len] = emb
                emb_idxes[emb_list_len] = int(fname.split('/')[-1][:-4])
                emb_list_len += 1
        
        return mix_audio, emb_list, emb_list_len, emb_idxes
    
    def collate_fn(self, batch):
        mix_audio = torch.stack([item[0][0] for item in batch])
        
        max_tracks = 9
        target_embedding = torch.full((len(batch), max_tracks, 1024), self.pad_num)  # Fill the rest with infinity.
        for i, item in enumerate(batch):
            for j, track in enumerate(item[1]):
                target_embedding[i, j, :] = track

        emb_len = torch.stack([item[2] for item in batch])
        emb_idxes = torch.stack([item[3] for item in batch])
        return mix_audio, target_embedding, emb_len, emb_idxes

class Single_Mix_Mel_dataset(Dataset):
    def __init__(self, data_path = "/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb", split = None, loss = 'Cosine'):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.single_list = glob.glob(os.path.join(self.data_path, f'{self.split}_single', '*'))
        self.single_list.sort()
        self.dir_list = glob.glob(os.path.join(self.data_path, self.split, "*"))
        self.dir_list.sort()

        if loss == 'MSE':
            self.pad_num = 1e8
        elif loss == 'Cosine':
            self.pad_num = 0.

    def __len__(self):
        # return 320
        return len(self.dir_list)

    def __getitem__(self, idx):
        dir_path = self.dir_list[idx]
        multi_list = glob.glob(os.path.join(dir_path, "*.npy"))
        multi_list.sort()

        for fname in multi_list:
            if 'mix_audio' in fname:
                mix_audio = np.load(fname)
                mel_spec = librosa.feature.melspectrogram(y=mix_audio, sr=SAMPLING_RATE, win_length=1024, hop_length=512, n_mels=128)
                log_spec = librosa.power_to_db(mel_spec)
                mix_audio = torch.tensor(log_spec, dtype=torch.float32)
        
        single_path = self.single_list[idx]
        single_path_list = glob.glob(os.path.join(single_path, "*.npy"))
        single_path_list.sort()

        single_list = torch.full((9, 128, 157), self.pad_num)

        # if self.split == 'train':
        #     single_idxes = np.zeros(953, dtype=np.int)
        # elif self.split =='valid':
        #     single_idxes = np.zeros(53, dtype=np.int)

        single_list_len = 0
        single_idxes = torch.zeros(len(single_path_list), dtype=torch.long)
        for fname in single_path_list:
            single_mel = np.load(fname)
            single_mel = torch.from_numpy(single_mel)
            single_list[single_list_len] = single_mel
            # inst_num = int(fname.split('/')[-1][:-4])
            # single_idxes[inst_num] = 1
            single_idxes[single_list_len] = int(fname.split('/')[-1][:-4])
            single_list_len += 1

        track_len = torch.tensor(single_list_len, dtype=torch.int32)
        return mix_audio, single_list, single_idxes, track_len

    def collate_fn(self, batch):
        mix_audio = torch.stack([item[0][0] for item in batch])

        single_tup = [item[1] for item in batch]
        single_list = torch.cat(single_tup, dim=0)

        single_idxes = [item[2] for item in batch]
        single_idxes = torch.cat(single_idxes, dim=0)

        track_len = torch.stack([item[3] for item in batch])
        return mix_audio, single_list, single_idxes, track_len

class FewShotReprDataset(Dataset):
    def __init__(self, path = "/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb_npy", split = 'valid', num_few = 5):
        self.dir_list = glob.glob(f'{path}/{split}_default_nfft/*')
        self.dir_list.sort()
        self.num_few = num_few

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        file_list = glob.glob(f'{self.dir_list[idx]}/*.npy')
        file_list.sort()
        file_idxes = random.sample(range(1000), self.num_few)
        npy = np.zeros(1024, dtype=np.float32)
        for idx in file_idxes:
            loaded = np.load(file_list[idx]).squeeze()
            npy += loaded
        npy /= self.num_few
        return npy

class OneShotReprDataset(Dataset):
    def __init__(self, path = "/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb_npy", split = 'valid', get_idx = None):
        self.dir_list = glob.glob(f'{path}/{split}_default_nfft/*')
        self.dir_list.sort()
        self.get_idx = get_idx

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        file_list = glob.glob(f'{self.dir_list[idx]}/*.npy')
        file_list.sort()
        npy = np.load(file_list[self.get_idx])
        return npy

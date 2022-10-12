from json import load
import os
import glob
import numpy as np
import librosa
import torch
import pytorch_lightning as pl
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
    def __init__(self, data_path = "/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3", split = None, loss = 'MSE', batch_size = 32):
        super().__init__()
        self.data_path = data_path
        self.split = split
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
        emb_idx_list = np.zeros(len(fname_list)-3, dtype=np.int)
        emb_list_idx = 0
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
                emb_list[emb_list_idx] = emb
                emb_idx_list[emb_list_idx] = int(fname.split('/')[-1][:-4])
                emb_list_idx += 1

        return mix_audio, emb_list, emb_list_idx, emb_idx_list
    
    def collate_fn(self, batch):
        mix_audio = torch.stack([item[0][0] for item in batch])
        
        # max_tracks = max([len(item[1]) for item in batch])
        max_tracks = 9
        target_embedding = torch.full((len(batch), max_tracks, 1024), self.pad_num)  # Fill the rest with infinity.
        for i, item in enumerate(batch):
            for j, track in enumerate(item[1]):
                target_embedding[i, j, :] = track

        emb_len = torch.stack([item[2] for item in batch])
        # emb_len = [item[2] for item in batch]
        return mix_audio, target_embedding, emb_len


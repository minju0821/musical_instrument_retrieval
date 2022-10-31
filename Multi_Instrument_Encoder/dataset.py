import os
import glob
import random
from pathlib import Path

import scipy
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

random.seed(0)
torch.manual_seed(0)

SAMPLING_RATE = 16000

"""
- before training the multi_instrument_encoder
    1. save all the embeddings with trained single_instrument_encoder
    2. save preprocessed mixture audios in the form of mel-spectrogram for faster training
- mix_mel.npy : mel-spectrogram of mixture audio
- {num}_emb.npy : embedding of single instrument audio extracted by trained single_instrument_encoder
"""
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

        emb_list = torch.full((9, 1024), self.pad_num)
        emb_idx_list = torch.full((9,), -1, dtype=torch.int32)
        emb_list_idx = 0
        for fname in fname_list:
            if 'mix' in fname:
                mix = np.load(fname)
                mix = torch.tensor(mix, dtype=torch.float32)
            else:
                emb_list[emb_list_idx] = torch.from_numpy(np.load(fname))
                emb_idx_list[emb_list_idx] = int(fname.split('/')[-1][:-4])
                emb_list_idx += 1

        emb_list_idx = torch.tensor(emb_list_idx, dtype=torch.int32)
        return mix, emb_list, emb_idx_list, emb_list_idx

# randomly mix single instrument tracks to make mixture audio
class RandomMixMultiInstrumentDataset(Dataset):
    def __init__(self, audio_path = None, single_inst_emb_path = None, split = None, min_tracks=2, max_tracks=9):
        self.split = split
        self.audio_path = Path(audio_path) / split
        if self.split == "train":
            self.num_inst = 953
        elif self.split == "valid":
            self.num_inst = 53

        self.single_emb_path = sorted(glob.glob(single_inst_emb_path + "/*/"))
        
        self.min_tracks = min_tracks
        self.max_tracks = max_tracks

    def __len__(self):
        if self.split == 'train':
            return 10000
        elif self.split == 'valid':
            return 1000
        # These numbers are arbitrary. It is just to make the dataloader work.

    def __getitem__(self, idx):
        mix_audio = np.zeros((SAMPLING_RATE*5))
        num_tracks = np.random.randint(self.min_tracks, self.max_tracks+1)
        inst_idx = np.random.randint(1, self.num_inst+1, 9)
        inst_idx[num_tracks:] = -1

        emb_list = torch.zeros((9, 1024)) # pad with zeros (Cosine loss)

        for i in range(num_tracks):
            sample_idx = np.random.randint(1, 1001)
            if self.split == "train":
                audio = scipy.io.wavfile.read(self.audio_path / f"{inst_idx[i]:03d}" / f"{sample_idx:04d}.wav")[1]
            elif self.split == "valid":
                audio = scipy.io.wavfile.read(self.audio_path / f"{inst_idx[i]:02d}" / f"{sample_idx:04d}.wav")[1]
            mix_audio += audio  

            single_emb = np.load(self.single_emb_path[inst_idx[i]-1] + f'{sample_idx:04d}.npy')
            emb_list[i] = torch.from_numpy(single_emb).squeeze()
            
        mix_audio = mix_audio / np.max(np.abs(mix_audio))
        mel_spec = librosa.feature.melspectrogram(y=mix_audio, sr=SAMPLING_RATE, win_length=1024, hop_length=512, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec)
        mix_audio = torch.from_numpy(log_spec)

        # inst_idx : 1~53 -> 0~52
        inst_idx -= 1
        num_tracks = torch.tensor(num_tracks, dtype=torch.int32)
        return mix_audio, emb_list, inst_idx, num_tracks

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
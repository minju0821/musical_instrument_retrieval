import glob
import random
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset

import librosa
import scipy.io.wavfile

random.seed(0)
torch.manual_seed(0)

# pre-rendered wav. data (midi w/ nsynth) : to save data loading time
class RenderedInstrumentDataset(Dataset):
    def __init__(self, split = "train", data_path = "/data4/aiproducer_inst/rendered_single_inst/",
                 num_samples_per_inst=1000):
        assert num_samples_per_inst == 1000, "1000 samples per instrument"
        self.num_samples_per_inst = num_samples_per_inst
        self.split = split
        self.data_path = Path(data_path) / split
        self.inst_dirs = sorted(glob.glob(str(self.data_path) + "/*/"))
        self.length = len(self.inst_dirs) * self.num_samples_per_inst
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        inst_idx = idx % len(self.inst_dirs)
        sample_idx = idx // len(self.inst_dirs)
        inst_dir = self.inst_dirs[inst_idx]
        fname = inst_dir + f"{sample_idx+1:04d}.wav"

        sr, audio = scipy.io.wavfile.read(fname)
        audio = np.array(audio, dtype=np.float32) / 32768.0
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec)
        log_spec = torch.tensor(log_spec, dtype=torch.float32)

        return log_spec, inst_idx, sample_idx

# pre-rendered wav. data (midi w/ nsynth) : to save data loading time
class RenderedMultiInstrumentDataset(Dataset):
    def __init__(self, split = "train", data_path = "/data4/aiproducer_inst/rendered_multi_inst_3/"):
        self.split = split

        if split == "train":
            self.num_samples = 100000
        elif split == "valid":
            self.num_samples = 10000
        
        self.data_path = Path(data_path) / split
        self.sample_dirs = glob.glob(str(self.data_path) + "/*/")
        self.sample_dirs.sort()
        self.sample_dirs = self.sample_dirs[:self.num_samples]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        fnames = glob.glob(self.sample_dirs[idx] + "*.wav")
        fnames.sort()

        track_list = []
        idx_list = []
        for fname in fnames[:-1]:
            sr, audio = scipy.io.wavfile.read(fname)
            audio = np.array(audio, dtype=np.float32) / 32768.0
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128, n_fft=1024)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32)

            track_list.append(log_spec)
            idx_list.append(torch.tensor(int(fname.split("/")[-1].split("_")[0])))
        
        # Mix file will be the last file.
        sr, mix = scipy.io.wavfile.read(fnames[-1])
        mix = np.array(mix, dtype=np.float32) / 32768.0

        mel_spec = librosa.feature.melspectrogram(y=mix, sr=sr, win_length=1024, hop_length=512, n_mels=128, n_fft=1024)
        log_spec = librosa.power_to_db(mel_spec)
        mix = torch.tensor(log_spec, dtype=torch.float32)
        
        # mixed track & track list of single tracks
        return mix, track_list, idx_list

    def collate_fn(self, batch):
        mix = torch.stack([item[0] for item in batch])
        track_list = []
        idx_list = []
        for sample in batch:
            track_list.append(torch.stack(sample[1]))
            idx_list.append(torch.stack(sample[2]))
        return mix, track_list, idx_list


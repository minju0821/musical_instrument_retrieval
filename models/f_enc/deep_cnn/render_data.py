import glob
import os.path
import time
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile
from pathlib import Path
from tqdm import tqdm

from nsynthesizer import NSynthesizer

NSYNTH_SAMPLE_RATE = 16000
NSYNTH_VELOCITIES = [25, 50, 100, 127]
INST_FAM_LIST = ["keyboard", "mallet", "organ", "guitar", "bass", "string", "vocal", "brass", "reed", "flute", "synth_lead", "drum", "etc"]
class InstrumentDataset(Dataset):
    def __init__(self, split='train', nsynth_path="/data1/aiproducer_inst/nsynth/", midi_path="/data1/aiproducer_inst/clean_midi_inst_new/"):
        self.nsynth_path = Path(nsynth_path) / f"nsynth-inst-{split}"
        self.nsynth_dirs = glob.glob(str(self.nsynth_path)+"/*/*/*")
        self.nsynth_dirs.sort()
        self.length = len(self.nsynth_dirs)

        self.midi_path = Path(midi_path) / split
        self.inst_fnames = dict()
        inst_dirs = glob.glob(str(self.midi_path) + "/*")
        inst_dirs = [item for item in inst_dirs if os.path.isdir(item)]
        for inst_dir in inst_dirs:
            fnames = glob.glob(inst_dir+"/*.mid")
            inst_name = inst_dir.split("/")[-1]
            self.inst_fnames[inst_name] = fnames

        self.synth = NSynthesizer(dataset_path = self.nsynth_path,sr=NSYNTH_SAMPLE_RATE,velocities=NSYNTH_VELOCITIES, attack_in_ms = 0, release_in_ms=100)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inst_fam, inst_type, inst_num = self.idx_to_inst(idx)
        start_position = np.random.uniform(low=0.1, high=0.9)
        i = 0
        while True:
            i += 1
            midi_fname = self.random_sample_midi(inst_fam)
            audio = self.synth.partially_render_sequence(sequence=midi_fname, instrument=inst_fam, duration_in_seconds=5,
                                                source_type=inst_type, preset=inst_num, start_position=start_position, min_notes=5 - min(4, int(i/10)))
            if audio is not None:
                break
            else:
                continue
        audio = torch.tensor(audio, dtype=torch.float32)
        return audio

    def idx_to_inst(self,idx):
        inst_path = self.nsynth_dirs[idx]
        inst_fam, inst_type, inst_num = inst_path.split("/")[-3:]
        return inst_fam, inst_type, int(inst_num)

    def random_sample_midi(self, inst_fam):
        return random.choice(self.inst_fnames[inst_fam])

class MultiInstrumentDataset(Dataset):
    def __init__(self, split = "train", nsynth_path="/data1/aiproducer_inst/nsynth/", midi_path="/data1/aiproducer_inst/clean_midi_songs/"):
        self.split = split
        self.nsynth_path = Path(nsynth_path) / f"nsynth-inst-{split}"
        self.nsynth_dirs = glob.glob(str(self.nsynth_path)+"/*/*/*")
        self.nsynth_dirs.sort()
        
        # Put the instrument names in the dictionary with the instrument family as the key.
        # It is used for randomly sampling an instrument for a track.
        self.inst_fam_dict = dict()
        for inst_dir in self.nsynth_dirs:
            inst_fam, inst_type, inst_num = inst_dir.split("/")[-3:]
            if inst_fam not in self.inst_fam_dict:
                self.inst_fam_dict[inst_fam] = []
            self.inst_fam_dict[inst_fam].append((inst_type, inst_num))

        self.midi_path = Path(midi_path) / split
        self.inst_fnames = dict()
        self.song_dirs = glob.glob(str(self.midi_path) + "/*")
        self.song_dirs.sort()
        self.length = len(self.song_dirs)

        self.synth = NSynthesizer(dataset_path = self.nsynth_path,sr=NSYNTH_SAMPLE_RATE,velocities=NSYNTH_VELOCITIES, attack_in_ms = 0, release_in_ms=100)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        song_dir = self.song_dirs[idx]
        track_names = glob.glob(song_dir+"/*.mid")
        # Repeat synthesizing until all the tracks are successfully synthesized.
        i = 0
        while True:
            tick = time.time()
            start_position = np.random.uniform(low=0.1, high=0.9)
            i +=1 
            # print(f"Trying to synthesize {song_dir} {i:05d}")
            stem = []
            inst_info = []
            valid_track_num = 0
            for track_name in track_names:
                inst_fam = track_name.split("/")[-1].split("_")[0]
                if inst_fam == "synth":
                    if self.split == "test":
                        continue
                    inst_fam = "synth_lead"
                # extracted inst_fam could be drum, etc, or the  song name. In this case, skip the for loop.
                if inst_fam not in INST_FAM_LIST or inst_fam == "etc" or inst_fam == "drum":
                    continue
                else:
                    valid_track_num += 1
                inst_type, inst_num = random.choice(self.inst_fam_dict[inst_fam])
                inst_info.append((inst_fam, inst_type, inst_num))
                track_audio = self.synth.partially_render_sequence(sequence=track_name, instrument=inst_fam, duration_in_seconds=5,
                                                source_type=inst_type, preset=inst_num, start_position=start_position, min_notes=(5 - min(4, int(i/10)) ) )
                if track_audio is None:
                    continue
                stem.append(track_audio)
            tock = time.time()
            if len(stem) > 4-min(3, int(i/20)):
                break
        stem = [torch.tensor(item, dtype=torch.float32) for item in stem]
        mix = torch.stack(stem).sum(dim=0)
        mix = mix / torch.max(torch.abs(mix))
        return mix, stem

    def random_sample_inst(self, inst_fam):
        inst_type, inst_num = random.choice(self.inst_fam_dict[inst_fam])
        return inst_fam, inst_type, inst_num
    
    def collate_fn(self, batch):
        mix = torch.stack([item[0] for item in batch])
        track_list = []
        for sample in batch:
            track_list.append(torch.stack(sample[1]))
        return mix, track_list

class RenderedInstrumentDataset(Dataset):
    def __init__(self, split = "train", data_path = "/disk2/aiproducer_inst/rendered_single_inst/",
                 num_samples_per_inst=1000):
        assert num_samples_per_inst == 1000, "악기당 1000 샘플 쓰기로 정했습니당."
        self.num_samples_per_inst = num_samples_per_inst
        self.split = split
        self.data_path = Path(data_path) / split
        self.inst_dirs = glob.glob(str(self.data_path) + "/*/")
        self.inst_dirs.sort()
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
        # audio = torch.tensor(audio, dtype=torch.float32) / 32768.0

        # customized for the deep_cnn
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec)
        log_spec = torch.tensor(log_spec, dtype=torch.float32)

        return log_spec, inst_idx

class RenderedMultiInstrumentDataset(Dataset):
    def __init__(self, split = "train", data_path = "/disk2/aiproducer_inst/rendered_multi_inst/", num_samples=None):
        self.split = split

        if num_samples is not None:
            self.num_samples = num_samples
        else:
            if split == "train":
                self.num_samples = 100000
            elif split == "test":
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
        for fname in fnames[:-1]:
            sr, audio = scipy.io.wavfile.read(fname)
            # audio = torch.tensor(audio, dtype=torch.float32) / 32768.0
            audio = np.array(audio, dtype=np.float32) / 32768.0
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32)
            track_list.append(log_spec)
        
        # Mix file will be the last file.
        sr, mix = scipy.io.wavfile.read(fnames[-1])
        # mix = torch.tensor(mix, dtype=torch.float32) / 32768.0
        mix = np.array(mix, dtype=np.float32) / 32768.0
        mel_spec = librosa.feature.melspectrogram(y=mix, sr=sr, win_length=1024, hop_length=512, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec)
        mix = torch.tensor(log_spec, dtype=torch.float32)
        
        # mixed track & track list of single tracks
        return mix, track_list

    def collate_fn(self, batch):
        mix = torch.stack([item[0] for item in batch])
        track_list = []
        for sample in batch:
            track_list.append(torch.stack(sample[1]))
        return mix, track_list

class RenderedMixInstrumentDataset(Dataset):
    def __init__(self, split = "train", data_path = "/disk2/aiproducer_inst/rendered_multi_inst/", num_samples=None):
        self.split = split

        if num_samples is not None:
            self.num_samples = num_samples
        else:
            if split == "train":
                self.num_samples = 100000
            elif split == "test":
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
        
        sr, mix = scipy.io.wavfile.read(fnames[-1])
        mix = np.array(mix, dtype=np.float32) / 32768.0
        mel_spec = librosa.feature.melspectrogram(y=mix, sr=sr, win_length=1024, hop_length=512, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec)
        mix = torch.tensor(log_spec, dtype=torch.float32)
 
        return mix




def show_spec(audio, fname):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(spec, y_axis='log', x_axis='time', sr=16000, ax=ax)
    ax.set_title(f"Synthesized Audio Sample ({fname})")
    fig.savefig(f"spec_{fname}.png")
    return

def save_audio(audio, fname):
    scipy.io.wavfile.write(fname, 16000, audio)


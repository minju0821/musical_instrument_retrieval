import glob
import random
import os.path
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset

import librosa
import librosa.display
import scipy.io.wavfile
import matplotlib.pyplot as plt

from nsynthesizer import NSynthesizer

NSYNTH_SAMPLE_RATE = 16000
NSYNTH_VELOCITIES = [25, 50, 100, 127]
INST_FAM_LIST = ["keyboard", "mallet", "organ", "guitar", "bass", "string", "vocal", "brass", "reed", "flute", "synth_lead", "drum", "etc"]

random.seed(0)
torch.manual_seed(0)

""" Use Rendered Instrument Dataset (Nlakh) for default """

# synthesize midi data w/ nsynth on the fly
# nsynth dataset path example : "/data/nsynth/nsynth-inst-train/brass/acoustic/0/brass_acoustic_000-067-075.wav"
# lakh   dataset path example : "/data/lakh/train/brass/2000.mid"
class InstrumentDataset(Dataset):
    def __init__(self, split='train', nsynth_path="/data/nsynth/", midi_path="/data/lakh/"):
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

# pre-rendered wav. data (midi w/ nsynth) : to save data loading time
class RenderedInstrumentDataset(Dataset):
    def __init__(self, split = "train", data_path = None,
                 num_samples_per_inst=1000):
        assert num_samples_per_inst == 1000, "1000 samples for each instrument"
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

        return log_spec, inst_idx
   
def show_spec(audio, fname):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(spec, y_axis='log', x_axis='time', sr=16000, ax=ax)
    ax.set_title(f"Synthesized Audio Sample ({fname})")
    fig.savefig(f"spec_{fname}.png")
    return

def save_audio(audio, fname):
    scipy.io.wavfile.write(fname, 16000, audio)

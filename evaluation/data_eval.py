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

NSYNTH_SAMPLE_RATE = 22050
NSYNTH_VELOCITIES = [25, 50, 100, 127]

class InstrumentDataset(Dataset):
    def __init__(self, num_samples, split='train', nsynth_path="/data1/aiproducer_inst/nsynth/", 
                    midi_path="/data1/aiproducer_inst/clean_midi_inst/",
                    duration_in_seconds=10., min_notes=10, num_classes=11):
        
        self.num_samples = num_samples
        self.duration_in_seconds = duration_in_seconds
        self.min_notes = min_notes
        self.num_classes = num_classes
        
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

        self.synth = NSynthesizer(dataset_path = self.nsynth_path, sr=NSYNTH_SAMPLE_RATE, velocities=NSYNTH_VELOCITIES, attack_in_ms = 0, release_in_ms=100)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inst_fam, inst_type, inst_num = self.idx_to_inst(idx)
        start_position = np.random.uniform(low=0.1, high=0.9)

        samples = []
        i = 0
        while i < self.num_samples:
            midi_fname = self.random_sample_midi(inst_fam)
            audio = self.synth.partially_render_sequence(sequence=midi_fname, instrument=inst_fam, 
                                                source_type=inst_type, preset=inst_num, start_position=start_position, 
                                                duration_in_seconds=self.duration_in_seconds, min_notes=self.min_notes)
            
            if audio is None:
                continue
            else:
                i += 1

            audio = torch.tensor(audio, dtype=torch.float32)
            samples.append(audio)

        return samples

    def sample_one(self, idx):
        inst_fam, inst_type, inst_num = self.idx_to_inst(idx)
        start_position = np.random.uniform(low=0.1, high=0.9)
        while True:
            midi_fname = self.random_sample_midi(inst_fam)
            audio = self.synth.partially_render_sequence(sequence=midi_fname, instrument=inst_fam, 
                                                source_type=inst_type, preset=inst_num, start_position=start_position, min_notes=10)
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

    def get_inst_list(self, num_class):
        f = open("/home/haessun/ai_prod/{}_class.txt".format(num_class), 'r')
        lines = f.readlines()

        inst_list = []
        for l in lines:
            inst_list.append(l[:-1])

        return inst_list

def show_spec(audio, fname):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(spec, y_axis='log', x_axis='time', sr=16000, ax=ax)
    ax.set_title(f"Synthesized Audio Sample ({fname})")
    fig.savefig(f"spec_{fname}.png")
    return

def save_audio(audio, fname):
    scipy.io.wavfile.write(f"audio_{fname}.wav", 16000, audio)


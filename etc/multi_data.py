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

from multi_nsynthesizer import NSynthesizer

NSYNTH_SAMPLE_RATE = 22050
NSYNTH_VELOCITIES = [25, 50, 100, 127]
INST_FAM_LIST = ["keyboard", "mallet", "organ", "guitar", "bass", "string", "vocal", "brass", "reed", "flute", "synth_lead", "drum", "etc"]

class InstrumentDataset(Dataset):
    def __init__(self, split='train', nsynth_path="/data1/aiproducer_inst/nsynth/", midi_path="/data1/aiproducer_inst/clean_midi_inst/"):
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

class MultiInstrumentDataset(Dataset):
    def __init__(self, split = "train", nsynth_path="/disk2/aiproducer_inst/nsynth/", midi_path="/disk2/aiproducer_inst/clean_midi_songs/"):
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

        # print(self.inst_fam_dict)

        self.midi_path = Path(midi_path) / split
        self.inst_fnames = dict()
        self.song_dirs = glob.glob(str(self.midi_path) + "/*")
        self.song_dirs.sort()
        self.length = len(self.song_dirs)

        self.synth = NSynthesizer(dataset_path = self.nsynth_path,sr=NSYNTH_SAMPLE_RATE,velocities=NSYNTH_VELOCITIES, attack_in_ms = 0, release_in_ms=100)
        
        self.split = split

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
                    inst_fam = "synth_lead"
                # extracted inst_fam could be drum, etc, or the  song name. In this case, skip the for loop.
                if inst_fam not in INST_FAM_LIST or inst_fam == "etc" or inst_fam == "drum":
                    continue

                elif inst_fam == "synth_lead" and self.split == "test":
                    continue

                else:
                    valid_track_num += 1
                inst_type, inst_num = random.choice(self.inst_fam_dict[inst_fam])
                inst_info.append((inst_fam, inst_type, inst_num))
                track_audio = self.synth.partially_render_sequence(sequence=track_name, instrument=inst_fam, duration_in_seconds=5,
                                                source_type=inst_type, preset=inst_num, start_position=start_position, min_notes=3)
                if track_audio is None:
                    continue
                stem.append(track_audio)
            tock = time.time()
            if len(stem) > 1:
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

class MultiLabelInstrumentDataset(Dataset):
    def __init__(self, split = "train", nsynth_path="/disk2/aiproducer_inst/nsynth/", midi_path="/disk2/aiproducer_inst/clean_midi_songs/"):
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

        # print(self.inst_fam_dict)

        self.midi_path = Path(midi_path) / split
        self.inst_fnames = dict()
        self.song_dirs = glob.glob(str(self.midi_path) + "/*")
        self.song_dirs.sort()
        self.length = len(self.song_dirs)

        self.synth = NSynthesizer(dataset_path = self.nsynth_path,sr=NSYNTH_SAMPLE_RATE,velocities=NSYNTH_VELOCITIES, attack_in_ms = 0, release_in_ms=100)
        
        self.split = split

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
                    inst_fam = "synth_lead"
                # extracted inst_fam could be drum, etc, or the  song name. In this case, skip the for loop.
                if inst_fam not in INST_FAM_LIST or inst_fam == "etc" or inst_fam == "drum":
                    continue

                elif inst_fam == "synth_lead" and self.split == "test":
                    continue

                else:
                    valid_track_num += 1
                inst_type, inst_num = random.choice(self.inst_fam_dict[inst_fam])
                inst_info.append((inst_fam, inst_type, inst_num))
                track_audio = self.synth.partially_render_sequence(sequence=track_name, instrument=inst_fam, duration_in_seconds=5,
                                                source_type=inst_type, preset=inst_num, start_position=start_position, min_notes=3)
                if track_audio is None:
                    continue
                stem.append(track_audio)
            tock = time.time()
            if len(stem) > 1:
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


def show_spec(audio, fname):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(spec, y_axis='log', x_axis='time', sr=16000, ax=ax)
    ax.set_title(f"Synthesized Audio Sample ({fname})")
    fig.savefig(f"spec_{fname}.png")
    return

def save_audio(audio, fname):
    scipy.io.wavfile.write(f"audio_{fname}.wav", 16000, audio)


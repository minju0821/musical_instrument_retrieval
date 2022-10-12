import glob
import warnings
warnings.filterwarnings(action='ignore')

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# others
import wandb
import scipy
import random
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_curve

# torch
import torch
from torch.utils.data import DataLoader

# modules
# from data_eval import InstrumentDataset
# from evaluation.deep_cnn_eval import parse_args, _ConvNet
from deep_cnn import ConvNet

class _EER:
    def __init__(self, encoder, device, num_class=953, num_eval = 20):
        self.encoder = encoder
        self.device = device
        self.num_eval = num_eval

        self.inst_data = InstrumentDataset(split='test', num_samples = self.num_eval, duration_in_seconds=5., min_notes=1, num_classes=num_class)

        self.inst_dataloader = DataLoader(self.inst_data, batch_size=1, shuffle=False)

    def _dist(self, a, b):
        euclidean_dist = ((a-b)**2).sum()**0.5
        return euclidean_dist

    def dist(self, a, b):
        dist = 1 - torch.sum(a*b) / (torch.norm(a) * torch.norm(b) + 1e-08)
        return dist

    # Enrollment set
    def embedding_avg(self, samples):
        embeddings = []
        for i in samples:
            try:
                i = i.to(self.device).unsqueeze(dim=0)
                emb = self.encoder(i)
            except (RuntimeError, ValueError):
                emb = self.encoder(i.to(self.device).squeeze(0))
            embeddings.append(emb)
        emb_avg = torch.stack(embeddings).mean(dim=0)
        return emb_avg

    # Evaluation set
    def evaluate_data(self):
        print('Evaluating...')
        y_score = np.array([])
        for idx, samples in enumerate(tqdm(self.inst_dataloader)):
            target_emb = self.embedding_avg(samples)

            # target sample
            for sample in samples:
                try:
                    render_emb = self.encoder(sample.to(self.device))
                except (RuntimeError, ValueError):
                    render_emb = self.encoder(sample.to(self.device).squeeze(0))

                dist_ = self.dist(target_emb, render_emb)
                y_score = np.append(y_score, np.array([dist_.item()]))

            # non-target sample
            for i in range(self.num_eval):
                idx_diff = random.randint(0,len(self.inst_dataloader) - 1)

                while idx_diff == idx:
                    idx_diff = random.randint(0,len(self.inst_dataloader) - 1)
                try:
                    render_emb = self.encoder(self.inst_data.sample_one(idx_diff).to(self.device))
                except (RuntimeError, ValueError):
                    render_emb = self.encoder(self.inst_data.sample_one(idx_diff).unsqueeze(0).to(self.device))

                dist_ = self.dist(target_emb, render_emb)
                y_score = np.append(y_score, np.array([dist_.item()]))

        return y_score * -1

    def compute_eer(self, fpr, tpr, thresholds):
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        return eer, thresholds[min_index]

    def evaluate(self):
        y_true = np.array([])
        for num in range(53):
            y_true = np.append(y_true, np.repeat([1,0], self.num_eval))
        y_score = self.evaluate_data()

        print(np.shape(y_true), np.shape(y_score))

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        eer, threshold = self.compute_eer(fpr,tpr,thresholds)
        return eer, threshold

class InstrumentDataset_eval:
    def __init__(self, split = "test", data_path = "/disk2/aiproducer_inst/rendered_single_inst/",
                 num_samples_per_inst=1000):
        assert num_samples_per_inst == 1000
        self.num_samples_per_inst = num_samples_per_inst
        self.split = split
        self.data_path = Path(data_path) / split
        self.inst_dirs = glob.glob(str(self.data_path) + "/*/")
        self.inst_dirs.sort()

    def get_samples(self, idx, num_samples):
        samples = []
        sample_idx_list = random.sample(range(self.num_samples_per_inst), num_samples)
        inst_dir = self.inst_dirs[idx]
        for sample_idx in sample_idx_list: 
            fname = inst_dir + f"{sample_idx+1:04d}.wav"
            sr, audio = scipy.io.wavfile.read(fname)
            audio = np.array(audio, dtype=np.float32) / 32768.0
            # audio = torch.tensor(audio, dtype=torch.float32) / 32768.0

            # customized for the deep_cnn
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32)

            samples.append(log_spec)
        return samples

class EER:
    def __init__(self, encoder, device, num_enroll = 5, num_eval = 20):
        self.encoder = encoder
        self.device = device
        self.num_enroll = num_enroll
        self.num_eval = num_eval
        self.inst_data = InstrumentDataset_eval()

    def dist(self, a, b):
        dist = 1 - torch.sum(a*b) / (torch.norm(a) * torch.norm(b) + 1e-08)
        return dist

    def enrollment(self, inst_idx):
        emb = []
        for sample in self.inst_data.get_samples(inst_idx, self.num_enroll):
            emb.append(self.encoder(sample.to(self.device).unsqueeze(0)))
        return torch.stack(emb).mean(dim=0)

    def y_score(self):
        y_score = np.array([])
        print('Evaluating..')
        for inst_idx in tqdm(range(len(self.inst_data.inst_dirs))):
            target_emb = self.enrollment(inst_idx)
            for sample in self.inst_data.get_samples(inst_idx, self.num_eval):
                pos_emb = self.encoder(sample.to(self.device).unsqueeze(0))
                dist = self.dist(target_emb, pos_emb)
                y_score = np.append(y_score, np.array([dist.item()]))
            for i in range(self.num_eval):
                idx_diff = random.randrange(len(self.inst_data.inst_dirs))
                while idx_diff == inst_idx:
                    idx_diff = random.randrange(len(self.inst_data.inst_dirs))
                neg_emb = self.encoder(self.inst_data.get_samples(idx_diff, 1)[0].unsqueeze(0).to(self.device))
                dist = self.dist(target_emb, neg_emb)
                y_score = np.append(y_score, np.array([dist.item()]))
        return y_score * -1

    def compute_eer(self, fpr, tpr, thresholds):
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        return eer, thresholds[min_index]

    def evaluate(self):
        y_true = np.array([])
        for _ in range(len(self.inst_data.inst_dirs)):
            y_true = np.append(y_true, np.repeat([1,0], self.num_eval))
        y_score = self.y_score()
        print(np.shape(y_true), np.shape(y_score))
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        eer, threshold = self.compute_eer(fpr,tpr,thresholds)
        return eer, threshold


f = open("/home/haessun/ai_prod/rendered_02_checkpoints.txt", 'r')

lines = f.readlines()
path = "/home/haessun/ai_prod/models/f_enc_rendered_02/"

# init.
models = {}
for epoch in range(12):
    models[epoch] = {}

for l in lines:
    splited = l.split("_")

    epoch = int(splited[1][5:])
    iter = int(splited[2][4:])
    loss = float(splited[4])
    acc = float(splited[-1])

    models[epoch][iter] = [l, loss, acc]

# sort by epoch(=key)
models = dict(sorted(models.items(), key = lambda item: item[0], reverse = False))

for epoch in models.keys():
    models[epoch] = dict(sorted(models[epoch].items(), key = lambda item: item[0], reverse = False))

print(models.keys())

def calculate(class_dict, gpu, interval):
    cuda = "cuda:{}".format(gpu)
    device = torch.device(torch.device(cuda) if torch.cuda.is_available() else 'cpu')
    print(device)

    model = ConvNet(out_classes=953).to(device)

    for epoch in class_dict.keys():
        for iter in class_dict[epoch].keys():
            if iter % interval == 0:
                model_info = class_dict[epoch][iter]

                # model_info = [l, loss, acc]
                model_path = path + model_info[0][:-1]

                loaded_dict = torch.load(model_path, map_location = device)
                loaded_dict = dict(list(loaded_dict.items())[:-2])
                model.load_state_dict(loaded_dict, strict=False)
                model.eval()
                f_enc = model.forward
                
                eer = EER(f_enc, device)
                eer_score, threshold = eer.evaluate()
                
                wandb.log({
                    "Epoch" : epoch,
                    "Iteration" : iter,
                    "EER_score" : eer_score,
                    "EER_threshold" : threshold,
                    "Train Loss" : model_info[1],
                    "Train Acc." : model_info[2],
                })

num_classes = 953
gpu = 5
interval = 300

wandb.init(
    project = "rendered_data",
    name = "EER_batch_32_class_{}_interval_{}".format(num_classes, interval)
)

calculate(models, gpu, interval)

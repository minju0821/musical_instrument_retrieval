import glob
import warnings
warnings.filterwarnings(action='ignore')

# torch
import torch
from torch.utils.data import Dataset, DataLoader

# others
import numpy as np
import argparse
import random
from sklearn.metrics import roc_curve
from tqdm import tqdm

# modules
from evaluation.data_eval import InstrumentDataset

## haessun
from evaluation.deep_cnn_eval import parse_args, ConvNet

import wandb

class EER:
    def __init__(self, encoder, device, num_class, num_eval = 5):
        self.encoder = encoder
        self.device = device
        self.num_eval = num_eval

        self.inst_data = InstrumentDataset(split='test', num_samples = self.num_eval, duration_in_seconds=5., min_notes=1, num_classes=num_class)

        self.inst_dataloader = DataLoader(self.inst_data, batch_size=1, shuffle=False)

    def dist(self, a, b):
        euclidean_dist = ((a-b)**2).sum()**0.5
        return euclidean_dist

    def _dist(self, a, b):
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

f = open("/home/haessun/ai_prod/batch_128_checkpoints.txt", 'r')

lines = f.readlines()
path = "/home/haessun/ai_prod/test_models/"

class_11, class_28, class_953 = {}, {}, {}

for l in lines:
    splited = l.split(" ")

    # class 11
    if splited[4] == "5818481":
        epoch = int(splited[-1].split("_")[1][5:])
        class_11[epoch] = path + splited[-1][:-1]

    # class 28
    if splited[4] == "5888177":
        epoch = int(splited[-1].split("_")[1][5:])
        class_28[epoch] = path + splited[-1][:-1]

    # class 28
    if splited[4] == "9680753":
        epoch = int(splited[-1].split("_")[1][5:])
        class_953[epoch] = path + splited[-1][:-1]

sorted_11 = dict(sorted(class_11.items(), key = lambda item: item[0], reverse = False))
sorted_28 = dict(sorted(class_28.items(), key = lambda item: item[0], reverse = False))
sorted_953 = dict(sorted(class_953.items(), key = lambda item: item[0], reverse = False))

eval_11_score, eval_28_score, eval_953_score = {}, {}, {}
eval_11_thrs, eval_28_thrs, eval_953_thrs = {}, {}, {}


def calculate(num_classes, class_dict, eval_score, eval_thrs, gpu, interval):
    cuda = "cuda:{}".format(gpu)
    device = torch.device(torch.device(cuda) if torch.cuda.is_available() else 'cpu')
    print(device)

    model = ConvNet(activ="LReLU", out_classes=num_classes).to(device)

    for v in class_dict.keys():

        if v % interval == 0:
            epoch = v
            model_path = class_dict[v]

            loaded_dict = torch.load(model_path, map_location = device)

            loaded_dict = dict(list(loaded_dict.items())[:-2])
            model.load_state_dict(loaded_dict, strict=False)
            model.eval()
            encoder = model.forward
            
            eer = EER(encoder, device, num_class=num_classes)

            eer_score, threshold = eer.evaluate()

            print('EER: ', eer_score)
            print('threshold: ', threshold)
            
            eval_score[v] = eer_score
            eval_thrs[v] = threshold
            
            wandb.log({
                "EER_score" : eer_score,
                "EER_threshold" : threshold
            })

num_classes = 953
gpu = 2
interval = 50

wandb.init(
    project = "trainset_overfit",
    name = "EER_batch_128_class_{}_interval_{}".format(num_classes, interval)
)

if num_classes == 11:
    calculate(11, sorted_11, eval_11_score, eval_11_thrs, gpu, interval)

if num_classes == 28:
    calculate(28, sorted_28, eval_28_score, eval_28_thrs, gpu, interval)

if num_classes == 953:
    calculate(953, sorted_953, eval_953_score, eval_953_thrs, gpu, interval)
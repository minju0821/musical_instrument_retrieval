import os
import glob
import scipy
import random
import librosa
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve

import torch
import numpy as np
from torch import nn

from models import ConvNet

random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--dataset_dir', type=str, default=None, required=True, help="Path to the rendered Nlakh dataset directory.")
    parser.add_argument('--checkpoint_dir', type=str, default=None, required=True, help="Path for the model to be evaluated.")

    args = parser.parse_args()
    return args

# customized dataset for validation to sample the number of num_eval samples of same instrument. (inst_idx)
class InstrumentDataset_eval:
    def __init__(self, split = "valid", data_path = None,
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
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32)

            samples.append(log_spec)
        return samples

"""
Single Instrument Encoder denoted as self.encoder
- for evaluation, we need to get the 1024-dim output from the encoder's 1st linear layer, which is considered to be the embedding of the instrument
- num_enroll : number of enroll samples per instrument (averaged)
- num_eval : num_eval number of samples for both positive and negative instruments
"""
class EER:
    def __init__(self, encoder, data_path, device, num_enroll = 5, num_eval = 20):
        self.encoder = encoder
        self.device = device
        self.num_enroll = num_enroll
        self.num_eval = num_eval
        self.inst_data = InstrumentDataset_eval(data_path=data_path)

    def dist(self, a, b):
        dist = 1 - torch.sum(a*b) / (torch.norm(a) * torch.norm(b) + 1e-08)
        return dist

    def enrollment(self, inst_idx):
        emb = []
        for sample in self.inst_data.get_samples(inst_idx, self.num_enroll):
            emb.append(self.encoder(sample.unsqueeze(0).type(torch.float32).to(self.device))[1])
        return torch.stack(emb).mean(dim=0)

    def y_score(self):
        y_score = np.array([])
        print('Evaluating..')
        for inst_idx in tqdm(range(len(self.inst_data.inst_dirs))):
            target_emb = self.enrollment(inst_idx)
            for sample in self.inst_data.get_samples(inst_idx, self.num_eval):
                pos_emb = self.encoder(sample.unsqueeze(0).type(torch.float32).to(self.device))[1]
                dist = self.dist(target_emb, pos_emb)
                y_score = np.append(y_score, np.array([dist.item()]))
            for i in range(self.num_eval):
                idx_diff = random.randrange(len(self.inst_data.inst_dirs))
                while idx_diff == inst_idx:
                    idx_diff = random.randrange(len(self.inst_data.inst_dirs))
                neg_emb = self.encoder(self.inst_data.get_samples(idx_diff, 1)[0].unsqueeze(0).type(torch.float32).to(self.device))[1]
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
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        eer, threshold = self.compute_eer(fpr,tpr,thresholds)
        return eer, threshold

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus
    DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))
    torch.cuda.empty_cache()

    model_eval = ConvNet(out_classes=953).cuda()
    model_eval = nn.DataParallel(model_eval).to(DEVICE)
    model_eval.load_state_dict(torch.load(args.checkpoint_dir, map_location=DEVICE), strict=False)
    model_eval.eval()

    with torch.no_grad():
        eer = EER(model_eval, data_path=args.dataset_dir, device=DEVICE)
        eer_score, threshold = eer.evaluate()

    print(f"EER: {eer_score:.4f}, Threshold: {threshold:.4f}")

if __name__ == "__main__":
    main()
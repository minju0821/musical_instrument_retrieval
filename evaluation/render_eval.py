import glob
import os.path
import scipy.io.wavfile
from pathlib import Path
import torch
import numpy as np
import random
from sklearn.metrics import roc_curve
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from deep_cnn import parse_args, ConvNet
import librosa

class InstrumentDataset_eval:
    def __init__(self, split = "test", data_path = "/data1/aiproducer_inst/rendered_single_inst/",
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
            # audio = torch.tensor(audio, dtype=torch.float32) / 32768.0

            # customized for the deep_cnn
            mel_spec = librosa.feature.melspectrogram(y=audio.astype(float), sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32)

            samples.append(log_spec)
        return samples

class EER:
    def __init__(self, encoder, device, num_enroll = 5, num_eval = 5):
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
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        eer, threshold = self.compute_eer(fpr,tpr,thresholds)
        return eer, threshold


if __name__ == "__main__":
    args = parse_args()
    cuda = "cuda:{}".format(args.gpu)

    device = torch.device(torch.device(cuda) if torch.cuda.is_available() else 'cpu')


    ''' Your Encoder'''
    model = ConvNet(out_classes=953).to(device)
    loaded_dict = torch.load('/home/haessun/ai_prod/models/f_enc_rendered_00/class953_epoch5_iter25300_trLoss_0.249_trAcc_91.438', map_location = device)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    encoder = model.forward


    eer = EER(encoder, device)
    _eer, _threshold = eer.evaluate()

    print("EER: {:.4f}, Threshold: {:.4f}".format(_eer, _threshold))






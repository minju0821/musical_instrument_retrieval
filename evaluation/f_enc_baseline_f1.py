import scipy
import librosa
import argparse
import numpy as np
import glob
from pathlib import Path
from tqdm.auto import tqdm
from deep_cnn import ConvNet as f_enc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchmetrics.functional as metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=953)
    parser.add_argument('--project_name', type=str, default=None)
    
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--audio_len', type=str, default='1sec')

    args = parser.parse_args()
    
    return args

class RenderedMultiInstrumentDataset(Dataset):
    def __init__(self, split = "train", data_path = "/disk2/aiproducer_inst/rendered_multi_inst_2/", num_samples=None, audio_len="1sec"):
        self.split = split

        if num_samples is not None:
            self.num_samples = num_samples
        else:
            if split == "train":
                self.num_samples = 100000
            elif split == "valid":
                self.num_samples = 10000
        
        self.data_path = Path(data_path) / split
        self.sample_dirs = glob.glob(str(self.data_path) + "/*/")
        self.sample_dirs.sort()
        self.sample_dirs = self.sample_dirs[:self.num_samples]

        self.audio_len = audio_len
    
    def __len__(self):
        return self.num_samples

    def get_inst_idx_list(self, idx):
        wav_list = glob.glob(self.sample_dirs[idx] + "*.wav")
        wav_list.sort()

        inst_idx_list = []

        for wav in wav_list:
            if wav.split("/")[-1] == "mix.wav":
                continue

            inst_idx = int(wav.split("/")[-1].split("_")[0])
            inst_idx_list.append(inst_idx)

        inst_idx_list = torch.Tensor(inst_idx_list)
        return inst_idx_list
            
    def sliding_win(self, wav, sr):
        wav_len = len(wav)

        preprocessed = []

        for i in range(0, wav_len, int(sr/2)):
            tmp = wav[i:i+sr]
            if len(tmp) == 8000:
                continue
            mel_spec = librosa.feature.melspectrogram(y=tmp, sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32)
            preprocessed.append(log_spec)

        preprocessed = torch.stack(preprocessed).squeeze()
        return preprocessed

    def __getitem__(self, idx):
        fnames = glob.glob(self.sample_dirs[idx] + "*.wav")
        fnames.sort()
        
        sr, mix = scipy.io.wavfile.read(fnames[-1])
        mix = np.array(mix, dtype=np.float32) / 32768.0

        if self.audio_len =="1sec":
            mix_list = self.sliding_win(mix, sr)
            inst_idx_list = self.get_inst_idx_list(idx)
            return mix_list, inst_idx_list

        mel_spec = librosa.feature.melspectrogram(y=mix, sr=sr, win_length=1024, hop_length=512, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec)
        mix = torch.tensor(log_spec, dtype=torch.float32)
        
        inst_idx_list = self.get_inst_idx_list(idx)
        return mix, inst_idx_list

def evaluate(f_enc, test_loader, audio_len, DEVICE):
    preds_list = []
    inst_idxes_list = []

    for idx, (mix, inst_idxes) in enumerate(tqdm(test_loader[:10])):
        mix = mix.to(DEVICE)
        inst_idxes = inst_idxes.to(DEVICE)
        
        with torch.no_grad():
            if audio_len == "full":
                f_enc_out = f_enc(mix)
                preds_list.append(f_enc_out)
                inst_idxes_list.append(inst_idxes)

            elif audio_len == "1sec":
                output = []

                for m in mix:
                    output.append(f_enc(m))
                output = torch.stack(output).squeeze()
                summed = torch.sum(output, axis=0)

                preds_list.append(F.normalize(summed, dim=0))
                inst_idxes_list.append(inst_idxes)
        
    preds_list = torch.stack(preds_list)
    inst_idxes_list = torch.stack(inst_idxes_list).squeeze()
    inst_idxes_list = inst_idxes_list.to(torch.int64)

    f1_micro = metrics.f1_score(preds=preds_list, target=inst_idxes_list, num_classes=53, threshold=0.5, average='micro')
    f1_macro = metrics.f1_score(preds=preds_list, target=inst_idxes_list, num_classes=53, threshold=0.5, average='macro')

    return f1_micro, f1_macro


if __name__=='__main__':
    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    # load f_enc
    model = f_enc(out_classes = 953).to(DEVICE)
    loaded_dict = torch.load('/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_00/class953_epoch6_iter22200_trLoss_0.193_trAcc_91.490', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    train_dataset = RenderedMultiInstrumentDataset(split='valid', audio_len=args.audio_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False)


    f1_micro, f1_macro = evaluate(_f_enc, train_loader, args.audio_len, DEVICE)

    f = open("./{}_f1_micro_macro.txt".format(args.audio_len), 'a')
    f.write("f1_micro: {}, f1_macro: {}\n".format(f1_micro, f1_macro))
    f.close()
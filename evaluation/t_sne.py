import glob
import argparse
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from etc.data import InstrumentDataset
from sklearn.manifold import TSNE
from deep_cnn import ConvNet_eval as f_enc
import random
import librosa
import scipy
random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num', type=int, default=None)

    args = parser.parse_args()
    
    return args

if __name__=='__main__':

    DEVICE = torch.device('cuda:{}'.format(5)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    args = parse_args()
    path = "/disk2/aiproducer_inst/rendered_single_inst/{}/".format(args.split)
    dir_list = glob.glob(path + '*')

    # to randomly select 50 instruments among 953 instruments in case of train dataset
    # for valid dataset, we will use all of the 53 instruments
    random.shuffle(dir_list)

    # load f_enc
    model = f_enc(out_classes = 953).to(DEVICE)
    loaded_dict = torch.load('/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class953_epoch8_iter1200_trLoss_0.300_trAcc_89.031', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    datapoints = []
    labels = []

    if args.split == 'train':
        dir_list = dir_list[:50]

    # ex) inst_folder_path = /disk2/aiproducer_inst/rendered_single_inst/train/1
    for inst_folder_path in tqdm(dir_list):
        inst_num = inst_folder_path.split('/')[-1]
        wav_list = glob.glob(inst_folder_path + '/*.wav')
        
        sample_idx = list(range(len(wav_list)))
        random.shuffle(sample_idx)

        for idx in sample_idx[:10]:
            wav_path = wav_list[idx]

            sr, audio = scipy.io.wavfile.read(wav_path)
            audio = np.array(audio, dtype=np.float32) / 32768.0
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32)

            datapoints.append(_f_enc(log_spec.unsqueeze(dim=0).to(DEVICE)).squeeze())
            labels.append(int(inst_num))


    datapoints = [d.cpu().detach().numpy() for d in datapoints]
    print(len(datapoints), datapoints[0].shape)
    model = TSNE(n_components=2)
    transformed = model.fit_transform(np.array(datapoints))

    tsne_np = transformed
    tsne_df = pd.DataFrame(tsne_np, columns=['x', 'y'])
    tsne_df['label'] = labels

    plt.figure(figsize=(10, 10))
    for label in set(labels):
        tmp_df = tsne_df[tsne_df['label'] == label]
        plt.scatter(tmp_df['x'], tmp_df['y'], label=label)

    # plt.show()
    plt.legend()
    plt.savefig('t_sne_{}_{}.png'.format(args.split, args.num))

    

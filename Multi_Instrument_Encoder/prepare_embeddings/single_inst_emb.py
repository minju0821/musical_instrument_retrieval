import os
import random
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
from torch import nn

from models import ConvNet
from dataset import RenderedInstrumentDataset

random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=953)

    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None, required=True, help="Path to checkpoint directory of Single Instrument Encoder.")
    parser.add_argument("--emb_save_dir", type=str, default=None, required=True)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    train_dataset = RenderedInstrumentDataset(split='train')
    valid_dataset = RenderedInstrumentDataset(split='valid')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4, shuffle=False)

    train_path = f'{args.emb_save_dir}/train/'
    valid_path = f'{args.emb_save_dir}/valid/'

    single_inst_enc = ConvNet(out_classes=953).cuda()
    single_inst_enc = nn.DataParallel(single_inst_enc).to(DEVICE)
    loaded_dict = torch.load(f'{args.checkpoint_dir}', map_location=DEVICE)
    single_inst_enc.load_state_dict(loaded_dict, strict=False)
    single_inst_enc.eval()

    with torch.no_grad():

        for i, (inst_mel, inst_idx, sample_idx) in tqdm(enumerate(train_loader)):
            inst_mel, inst_idx, sample_idx = inst_mel.to(DEVICE), inst_idx.to(DEVICE), sample_idx.to(DEVICE)
            sample_idx = sample_idx.item()
            inst_idx = inst_idx.item()

            path_to_save = train_path + str(inst_idx+1).zfill(3)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            out = single_inst_enc(inst_mel)[1]
            np.save(f'{path_to_save}/{sample_idx+1:04d}.npy', out.squeeze().cpu().numpy())


        for i, (inst_mel, inst_idx, sample_idx) in tqdm(enumerate(valid_loader)):
            inst_mel, inst_idx, sample_idx = inst_mel.to(DEVICE), inst_idx.to(DEVICE), sample_idx.to(DEVICE)
            sample_idx = sample_idx.item()
            inst_idx = inst_idx.item()

            path_to_save = valid_path + str(inst_idx+1).zfill(2)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            out = single_inst_enc(inst_mel)[1]
            np.save(f'{path_to_save}/{sample_idx+1:04d}.npy', out.squeeze().cpu().numpy())

            
import os
import random
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
from torch import nn

from models import ConvNet
from dataset import RenderedMultiInstrumentDataset

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

    train_dataset = RenderedMultiInstrumentDataset(split='train')
    valid_dataset = RenderedMultiInstrumentDataset(split='valid')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, collate_fn=train_dataset.collate_fn, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4, collate_fn=valid_dataset.collate_fn, shuffle=False)

    train_path = f"{args.emb_save_dir}/train/"
    valid_path = f"{args.emb_save_dir}/valid/"

    single_inst_enc = ConvNet(out_classes=953).cuda()
    single_inst_enc = nn.DataParallel(single_inst_enc).to(DEVICE)
    loaded_dict = torch.load(f'{args.checkpoint_dir}', map_location=DEVICE)
    single_inst_enc.load_state_dict(loaded_dict, strict=False)
    single_inst_enc.eval()

    with torch.no_grad():
        """  train dataset  """
        for i, (mix_mel, track_list, idx_list) in tqdm(enumerate(train_loader)):
            mix_mel = mix_mel.to(DEVICE)
            track_list = track_list[0].to(DEVICE)
            idx_list = idx_list[0].to(DEVICE)

            mix_to_save = mix_mel.squeeze()
            track_list_to_save = single_inst_enc(track_list)[1]

            path_to_save = train_path + str(i+1).zfill(6)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            np.save(path_to_save + '/mix_mel.npy', mix_to_save.cpu().numpy())
            for idx, track in enumerate(track_list_to_save):
                np.save(path_to_save + f'/{idx_list[idx]}_emb.npy', track.cpu().numpy())

        """  valid dataset  """
        for i, (mix_mel, track_list, idx_list) in tqdm(enumerate(valid_loader)):
            mix_mel = mix_mel.to(DEVICE)
            track_list = track_list[0].to(DEVICE)
            idx_list = idx_list[0].to(DEVICE)

            mix_to_save = mix_mel.squeeze()
            track_list_to_save = single_inst_enc(track_list)[1]

            path_to_save = valid_path + str(i+1).zfill(6)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            np.save(path_to_save + '/mix_mel.npy', mix_to_save.cpu().numpy())
            for idx, track in enumerate(track_list_to_save):
                np.save(path_to_save + f'/{idx_list[idx]}_emb.npy', track.cpu().numpy())

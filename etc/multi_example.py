import pdb
import numpy as np
from torch.utils.data import DataLoader
import torch
import time
import random
from tqdm import tqdm

from multi_data import InstrumentDataset, MultiInstrumentDataset, save_audio
import shutil

from h_enc.h_enc_deep_cnn import ConvNet as f_enc
from h_enc.h_enc import ConvNet as h_enc

def main():
    dataset = MultiInstrumentDataset(split='train')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)

    tick = time.time()

    model = f_enc('LReLU', 953)
    loaded_dict = torch.load('/home/haessun/ai_prod/test_models/test_epoch2585_trloss_0.397_tracc87.093')
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    tqdm_bar = tqdm(enumerate(data_loader))
    for idx, (data, label) in tqdm_bar:
        tock = time.time()
        print("loading duratiaon", tock-tick)

        tick = time.time()

        GT_emb = torch.zeros(2, 1, 1024)

        for idx, label in enumerate(label):
            # for emb in label:
            #     GT_emb[idx] += _f_enc(emb.unsqueeze(dim=0))
            # print(label.size(), label.unsqueeze(dim=1).size())
            GT_emb[idx] = torch.sum(_f_enc(label), dim=0)

        print(GT_emb.squeeze().size())

        _h_enc = h_enc()
        output = _h_enc(data)
        print(output.size())

        break

if __name__ == "__main__":
    main()
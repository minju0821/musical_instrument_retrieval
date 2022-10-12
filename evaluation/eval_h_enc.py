import warnings
warnings.filterwarnings(action='ignore')

# torch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# others
import numpy as np
import argparse
import random
from tqdm import tqdm
import wandb
import time

from multi_data import MultiInstrumentDataset
from h_enc.h_enc import ConvNet as h_enc
from deep_cnn import ConvNet as f_enc

## seed ????
torch.random.manual_seed(0)
random.seed(0)

def main():

    loss_name = 'MSE'
    epoch = 385
    gpu = 5

    cuda = "cuda:{}".format(gpu)
    DEVICE = torch.device(torch.device(cuda) if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    wandb.init(
        project = "h_enc",
        name = "Test_batch_32_loss_{}_epoch_{}".format(loss_name, epoch)
    )

    # MSE Loss
    loaded_dict = torch.load('/home/haessun/ai_prod/h_enc_02/class953_epoch385_trLoss_1.110_MSE', map_location = DEVICE)


    # Cosine Embedding Loss
    # loaded_dict = torch.load('/home/haessun/ai_prod/h_enc_02/class953_epoch410_trLoss_0.398_Cosine', map_location = DEVICE)
    
    # load h_enc
    _h_enc = h_enc().to(DEVICE)
    _h_enc.load_state_dict(loaded_dict, strict=False)
    _h_enc.eval()
    _h_encoder = _h_enc.forward

    # load f_enc
    model = f_enc('LReLU', 953).to(DEVICE)
    loaded_dict = torch.load('/home/haessun/ai_prod/test_models/test_epoch2585_trloss_0.397_tracc87.093', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    train_dataset = MultiInstrumentDataset(split='test')
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
    tqdm_bar = tqdm(enumerate(train_loader))
    
    _loss = []

    for idx, (data, labels) in tqdm_bar:

        start = time.time()

        inputs = data.to(DEVICE)

        # init GT embedding
        GT_emb = torch.zeros(inputs.size()[0], 1, 1024).to(DEVICE)

        # add all the single inst embedding
        for idx, label in enumerate(labels):
            GT_emb[idx] = torch.sum(_f_enc(label.to(DEVICE)), dim=0)
        GT_emb = GT_emb.squeeze()

        out = _h_encoder(inputs)

        # print(out.size(), GT_emb.size(), inputs.size()[0])

        # compute loss
        if loss_name == 'MSE':
            loss = nn.MSELoss()(out, GT_emb)
        elif loss_name == 'Cosine':
            loss = nn.CosineEmbeddingLoss()(out, GT_emb, torch.ones(inputs.size()[0]).to(DEVICE))

        _loss.append(loss.item())

        wandb.log({
            "Test Loss" : loss,
            "Test Time" : time.time() - start
        })


if __name__ == "__main__":
    main()
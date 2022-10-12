import torch
from tqdm.auto import tqdm
from glob import glob
import numpy as np
import os

DEVICE = torch.device('cuda:{}'.format(11)) if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

valid_path = glob('/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb/valid/*')
train_path = glob('/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb/train/*')

with torch.no_grad():
    
    # get f_emb for each data in dataset and save it
    # for i, path in tqdm(enumerate(valid_path)):
    #     f = torch.load(path, map_location=DEVICE).cpu().detach().numpy()
    #     dir_path = '/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb_npy/valid/'

    #     filename = path.split('/')[-1]
    #     inst_num = filename.split('-')[2]
    #     path_to_save = dir_path + inst_num

    #     if not os.path.exists(path_to_save):
    #         os.makedirs(path_to_save)

    #     np.save(f'{path_to_save}/{filename[:-2]}npy', f)

    for i, path in tqdm(enumerate(train_path)):
        f = torch.load(path, map_location=DEVICE).cpu().detach().numpy()
        dir_path = '/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb_npy/train/'

        filename = path.split('/')[-1]
        inst_num = filename.split('-')[2]
        path_to_save = dir_path + inst_num

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        np.save(f'{path_to_save}/{filename[:-2]}npy', f)
    
        

    # get f_emb for each data in dataset and save it
    # for i, (input, idx) in tqdm(enumerate(train_loader)):
    #     input = input.to(DEVICE)
    #     f_emb = _f_enc(input)
    #     idx = idx.item()
    #     torch.save(f_emb, '/disk2/aiproducer_inst/f_embeddings/f_haessun/single_f_emb/train/{}-{}-{}-{}.pt'.format(train_inst_dict[idx][0][0], train_inst_dict[idx][0][1], idx, i))
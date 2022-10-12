import torch
from tqdm.auto import tqdm
from deep_cnn import ConvNet_eval as f_enc
from render_data import RenderedMultiInstrumentDataset
import os
import numpy as np


DEVICE = torch.device('cuda:{}'.format(11)) if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

train_dataset = RenderedMultiInstrumentDataset(split='train')
valid_dataset = RenderedMultiInstrumentDataset(split='valid')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, collate_fn=train_dataset.collate_fn, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4, collate_fn=train_dataset.collate_fn, shuffle=False)

with torch.no_grad():

    """  train dataset  """
    # path = '/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb/train_single/'
    # for i, (mix, track_list, idx_list, mix_audio) in tqdm(enumerate(train_loader)):

    #     track_list, idx_list = track_list[0].to(DEVICE), idx_list[0].to(DEVICE)

    #     path_to_save = path + str(i+1).zfill(6)
    #     if not os.path.exists(path_to_save):
    #         os.makedirs(path_to_save)

    #     for idx, track in enumerate(track_list):
    #         np.save(path_to_save + '/{}.npy'.format(idx_list[idx]), track.cpu().numpy())


    """  valid dataset  """
    path = '/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb/valid_single/'
    for i, (mix, track_list, idx_list, mix_audio) in tqdm(enumerate(valid_loader)):

        track_list, idx_list = track_list[0].to(DEVICE), idx_list[0].to(DEVICE)

        path_to_save = path + str(i+1).zfill(6)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        for idx, track in enumerate(track_list):
            np.save(path_to_save + '/{}.npy'.format(idx_list[idx]), track.cpu().numpy())
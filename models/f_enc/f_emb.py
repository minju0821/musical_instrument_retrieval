import torch
from tqdm.auto import tqdm
from deep_cnn import ConvNet_eval as f_enc
from render_data import RenderedMultiInstrumentDataset
import os
import numpy as np


DEVICE = torch.device('cuda:{}'.format(7)) if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

train_dataset = RenderedMultiInstrumentDataset(split='train')
valid_dataset = RenderedMultiInstrumentDataset(split='valid')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, collate_fn=train_dataset.collate_fn, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4, collate_fn=train_dataset.collate_fn, shuffle=False)

with torch.no_grad():

    # load f_enc
    model = f_enc(out_classes = 953).to(DEVICE)
    loaded_dict = torch.load('/data4/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class953_epoch8_iter1200_trLoss_0.300_trAcc_89.031', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    """  train dataset  """
    path = '/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb/train/'
    for i, (mix, track_list, idx_list, mix_audio) in tqdm(enumerate(train_loader)):

        mix_audio = mix_audio.to(DEVICE)
        mix = mix.to(DEVICE)
        track_list = track_list[0].to(DEVICE)
        idx_list = idx_list[0].to(DEVICE)

        mix_to_save = mix.squeeze()
        track_list_to_save = _f_enc(track_list)

        path_to_save = path + str(i+1).zfill(6)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        np.save(path_to_save + '/mix_audio.npy', mix_audio.cpu().numpy())
        np.save(path_to_save + '/mix.npy', mix_to_save.cpu().numpy())

        for idx, track in enumerate(track_list_to_save):
            np.save(path_to_save + '/{}.npy'.format(idx_list[idx]), track.cpu().numpy())

        summed = torch.sum(track_list_to_save, dim=0)
        np.save(path_to_save + '/summed.npy', summed.cpu().numpy())


    """  valid dataset  """
    # path = '/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb/valid/'
    # for i, (mix, track_list, idx_list, mix_audio) in tqdm(enumerate(valid_loader)):

    #     mix_audio = mix_audio.to(DEVICE)
    #     mix = mix.to(DEVICE)
    #     track_list = track_list[0].to(DEVICE)
    #     idx_list = idx_list[0].to(DEVICE)

    #     mix_to_save = mix.squeeze()
    #     track_list_to_save = _f_enc(track_list)

    #     path_to_save = path + str(i+1).zfill(6)
    #     if not os.path.exists(path_to_save):
    #         os.makedirs(path_to_save)

    #     np.save(path_to_save + '/mix_audio.npy', mix_audio.cpu().numpy())
    #     np.save(path_to_save + '/mix.npy', mix_to_save.cpu().numpy())

    #     for idx, track in enumerate(track_list_to_save):
    #         np.save(path_to_save + '/{}.npy'.format(idx_list[idx]), track.cpu().numpy())

    #     summed = torch.sum(track_list_to_save, dim=0)
    #     np.save(path_to_save + '/summed.npy', summed.cpu().numpy())
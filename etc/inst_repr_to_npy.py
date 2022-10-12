import torch
from tqdm.auto import tqdm
import os
import numpy as np


DEVICE = torch.device('cuda:{}'.format(11)) if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

path = '/home/haessun/ai_prod/inst_repr_lib'
save_path = '/home/haessun/ai_prod/inst_repr_lib_npy'

for i in tqdm(range(953)):

    tmp_pt = torch.load(f'{path}/{i+1}.pt', map_location=DEVICE)
    tmp_np = tmp_pt.cpu().detach().numpy()

    np.save(f'{save_path}/{i}.npy', tmp_np)

import torch
from tqdm.auto import tqdm
import os
import numpy as np
import glob

DEVICE = torch.device('cuda:{}'.format(11)) if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

split = 'train'

path_list = glob.glob(f'/home/haessun/ai_prod/{split}_inst_repr_lib_npy/*.npy')
path_to_save = f'/home/haessun/ai_prod/{split}_inst_repr_lib_npy'

for path in tqdm(path_list):
    inst_num = str(int(path.split('/')[-1][:-4])).zfill(3)
    tmp_np = np.load(path)
    np.save(f'{path_to_save}/{inst_num}.npy', tmp_np)
    os.remove(path)

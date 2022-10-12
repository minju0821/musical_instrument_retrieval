import glob
import torch
import random
from random import shuffle

DEVICE = torch.device('cuda:{}'.format(5)) if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

# train_paths = "/disk2/aiproducer_inst/f_embeddings/f_haessun/train/"
# train_paths = sorted(glob.glob(train_paths + "*"))

valid_paths = "/disk2/aiproducer_inst/f_embeddings/f_haessun/valid/"
valid_paths = sorted(glob.glob(valid_paths + "*"))

random.seed(0)

for i, path in enumerate(valid_paths):
    
    f_emb = torch.load(path)
    # input
    f_mix = f_emb[0].unsqueeze(0).to(DEVICE)

    # expected output
    f_tracks = f_emb[1:].to(DEVICE)
    f_tracks = torch.sum(f_tracks, dim=0).unsqueeze(0)

    print(f_mix.size(), f_tracks.size())

    break
import torch
from glob import glob
import os ; opj=os.path.join
from tqdm import tqdm

def main():
    
    path = "/disk2/aiproducer_inst/f_embeddings/f_haessun/valid"

    lst = glob(opj(path, '*tracks'))
    norm_lst = []
    
    for i, p in enumerate(tqdm(lst)):
        t = torch.load(p)
        t_sum = t.sum(0)
        t_norm = torch.nn.MSELoss()(t_sum, torch.zeros_like(t_sum))
        norm_lst.append(t_norm)
    
    norm_lst = torch.stack(norm_lst)
    print(torch.mean(norm_lst), torch.std(norm_lst))

if __name__=="__main__":
    main()


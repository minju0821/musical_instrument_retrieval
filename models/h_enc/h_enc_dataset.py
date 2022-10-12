import glob
from torch.utils.data import Dataset
import numpy as np

class EmbeddingDataset(Dataset):
    def __init__(self, data_path = "/disk2/aiproducer_inst/f_embeddings/f_haessun/", split = "train_new"):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.dir_list = sorted(glob.glob(self.data_path + self.split + "/*"))

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        mix = np.load(self.dir_list[idx] + "/mix.npy")
        summed = np.load(self.dir_list[idx] + "/summed.npy")

        idxes = glob.glob(self.dir_list[idx] + "/*.npy")
        idxes = [int(i.split("/")[-1].split(".")[0]) for i in idxes if i.split("/")[-1] != "mix.npy" and i.split("/")[-1] != "summed.npy"]

        BCE_trg = np.zeros(953)
        if self.split == "valid_new":
            BCE_trg = np.zeros(53)
        BCE_trg[idxes] = 1

        # error나면 .detach() 붙여보기
        return mix, summed, BCE_trg

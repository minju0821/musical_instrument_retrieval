import os
import json
import random
import argparse
from tqdm import tqdm

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

import timm
from models import ConvNet
from dataset import RenderedNlakhDataset, EmbeddingLibraryDataset

random.seed(0)
torch.manual_seed(0)
SAMPLING_RATE = 16000
VALID_INST = {'bass' : 0, 'brass' : 1, 'flute' : 2, 'guitar' : 3, 'keyboard' : 4, 'mallet' : 5, 'organ' : 6, 'reed' : 7, 'string' : 8, 'vocal' : 9}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default=None)

    parser.add_argument('--num_class', type=int, default=53, choices=[53, 10])
    parser.add_argument("--dataset", type=str, default="random_mix", choices=["rendered", "random_mix"])
    parser.add_argument("--model_size", type=str, default="Large", choices=["Large", "Small"])
    parser.add_argument('--get_idx', type=int, default=1, help='1 ~ 1000')

    parser.add_argument('--nlakh_dataset_dir', type=str, default=None, help="Path to the rendered Nlakh multi dataset directory.")
    parser.add_argument('--single_inst_emb_dir', type=str, default=None, help="Path to the output embeddings of single instruments processed with trained Single Instrument Encoder.")
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="Path to checkpoint directory of Multi Instrument Encoder.")

    parser.add_argument('--use_single_emb', type=bool, default=True, help="If False, get single instrument embedding on the fly.")
    parser.add_argument('--single_inst_enc_dir', type=str, default=None)

    args = parser.parse_args()
    return args

def check_args(args):
    model_name = args.checkpoint_dir.split("/")[-1]
    file_name = f"{args.dataset}_{args.model_size}_{model_name}_{args.num_class}classes"
    f = open(f"predictions_{file_name}.txt", "w")

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus
    DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    return f, DEVICE

def load_models(args, DEVICE):
    """ load Multi_Inst_Encoder """
    if args.model_size == "Large":
        model = timm.create_model('convnext_small', pretrained=False, in_chans=1, num_classes=9 * 1024, drop_path_rate=0.5).cuda()
    elif args.model_size == "Small":
        model = ConvNet().cuda()
    model = nn.DataParallel(model).to(DEVICE)

    loaded_dict = torch.load(args.checkpoint_dir, map_location=DEVICE)
    model.load_state_dict(loaded_dict)
    model.eval()

    if args.use_single_emb:
        return model

    """ load Single_Inst_Encoder """
    single_inst_enc = ConvNet(out_classes=953).cuda()
    single_inst_enc = nn.DataParallel(single_inst_enc).to(DEVICE)
    loaded_dict = torch.load(args.single_inst_enc_dir, map_location=DEVICE)
    single_inst_enc.load_state_dict(loaded_dict, strict=False)
    single_inst_enc.eval()

    return model, single_inst_enc

def write_result(est_list, ans_list, ran_list, score_list, args):
    est_list = np.asarray(est_list)
    ans_list = np.asarray(ans_list)
    ran_list = np.asarray(ran_list)

    avg_list = ['micro', 'macro', 'weighted']

    model_name = args.checkpoint_dir.split("/")[-1]
    f_result = open(f"result_{args.dataset}_{args.model_size}_{model_name}_{args.num_class}classes.txt", "w")
    
    for avg in avg_list:
        f1 = f1_score(ans_list, est_list, average=avg)
        f1_random = f1_score(ans_list, ran_list, average=avg)
        recall = recall_score(ans_list, est_list, average=avg)
        recall_random = recall_score(ans_list, ran_list, average=avg)
        precision = precision_score(ans_list, est_list, average=avg)
        precision_random = precision_score(ans_list, ran_list, average=avg)
        if not args.num_class == 10:
            mAP = average_precision_score(ans_list, score_list, average=avg)

        print("AVERAGE METHOD: ", avg)
        print('F1 Score', f1)
        print('F1 Score (random)', f1_random)
        print('Recall', recall)
        print('Recall (random)', recall_random)
        print('Precision', precision)
        print('Precision (random)', precision_random)
        if not args.num_class == 10:
            print('mean Avg. Precision ', mAP)
        print()

        f_result.write(f"Average method: {avg}\n")
        f_result.write(f'F1 Score : {str(f1)}\n')
        f_result.write(f'F1 Score (random) : {str(f1_random)}\n')
        f_result.write(f'Recall : {str(recall)}\n')
        f_result.write(f'Recall (random) : {str(recall_random)}\n')
        f_result.write(f'Precision : {str(precision)}\n')
        f_result.write(f'Precision (random) : {str(precision_random)}\n')
        if not args.num_class == 10:
            f_result.write(f'mean Avg. Precision : {str(mAP)}\n\n')
    
    f_result.close()

if __name__ == "__main__":
    args = parse_args()

    # VALID_INST[inst_dict[i].split('_')[0]] -> matching instrument index to instrument family index
    if args.num_class == 10:
        f = open('idx_to_fam_inst_valid.json')
        inst_dict = json.load(f)
        inst_dict = {y: x for x, y in inst_dict.items()}
        
    f, DEVICE = check_args(args)
    if args.use_single_emb:
        model = load_models(args, DEVICE)
    else:
        model, single_inst_enc = load_models(args, DEVICE)

    """ build embedding library """
    lib = torch.zeros((53, 1024)).to(DEVICE)
    emb_lib_dataset = EmbeddingLibraryDataset(path=args.single_inst_emb_dir, split='valid', get_idx=args.get_idx)
    emb_lib_loader = DataLoader(emb_lib_dataset, batch_size=1, shuffle=False)
    
    if not args.use_single_emb:
        for idx, repr in tqdm(enumerate(emb_lib_loader)):
            lib[idx] = single_inst_enc(repr.to(DEVICE))
    else:
        for idx, repr in tqdm(enumerate(emb_lib_loader)):
            lib[idx] = repr

    """ load validation dataset """
    dataset = RenderedNlakhDataset(data_path=args.nlakh_dataset_dir , split="valid")
    valid_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    
    f1_score_list = []
    f1_score_random = []
    recall_list = []
    recall_random = []
    precision_list = []
    precision_random = []

    est_list = []
    ans_list = []
    ran_list = []
    score_list = []

    cos_loss = nn.CosineEmbeddingLoss(reduction='none')
    cos_sim = nn.CosineSimilarity()
    for batch_idx, (mix, emb_idx_list) in tqdm(enumerate(valid_loader)):
        mix, emb_idx_list = mix.to(DEVICE), emb_idx_list.to(DEVICE)

        if args.model_size == "Large":
            output = model(mix.squeeze().unsqueeze(dim=0).unsqueeze(dim=0))
        elif args.model_size == "Small":
            output = model(mix.squeeze().unsqueeze(dim=0))
        output = torch.reshape(output, (output.size()[0], 9, -1))
        est_emb = output[0]

        """ estimated one-hot """
        est_one_hot = np.zeros(args.num_class, dtype=int)
        for idx, est in enumerate(est_emb):
            loss = cos_loss(est.repeat((53, 1)), lib, torch.ones(1).to(DEVICE))
            min_idx = torch.argmin(loss)
            if args.num_class == 10:
                min_idx = VALID_INST[inst_dict[min_idx.item()].split('_')[0]]
            est_one_hot[min_idx] = 1
        est_list.append(est_one_hot)

        """ estimated score """
        est_score = np.zeros((9, 53))
        for idx, est in enumerate(est_emb):
            est_score[idx] = cos_sim(est.repeat((53, 1)), lib).detach().cpu() # 53
        est_score = np.max(est_score, axis=0)
        score_list.append(est_score)

        """ random one-hot """
        random_one_hot = np.zeros(args.num_class, dtype=int)
        num_inst = random.randint(2, 9)
        for i in random.sample(range(args.num_class), num_inst):
            random_one_hot[i] = 1
        ran_list.append(random_one_hot)

        """ answer one-hot """ 
        emb_idx_list = emb_idx_list.cpu().detach().numpy()
        ans_one_hot = np.zeros(args.num_class, dtype=int)
        inst_tmp = [i for i in emb_idx_list[0] if not i < 0]
        if args.num_class == 10:
            inst_tmp = [VALID_INST[inst_dict[i].split('_')[0]] for i in inst_tmp]
        ans_one_hot[inst_tmp] = 1
        ans_list.append(ans_one_hot)

        f.write("pred : ")
        f.write(str(np.where(est_one_hot==1)[0].tolist()))
        f.write("\n")
        f.write("answ : ")
        f.write(str(np.where(ans_one_hot==1)[0].tolist()))
        f.write("\n")
        f.write("rand : ")
        f.write(str(np.where(random_one_hot==1)[0].tolist()))
        f.write("\n\n")

    f.close()
    write_result(est_list, ans_list, ran_list, score_list, args)
    

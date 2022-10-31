import os
import random
import argparse
from tqdm.auto import tqdm

import wandb
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

import timm
from models import ConvNet
from asteroid_PIT_loss import PITLossWrapper
from dataset import RenderedNlakhDataset, RandomMixMultiInstrumentDataset, EmbeddingLibraryDataset
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="")

    parser.add_argument('--num_class', type=int, default=953)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument("--dataset", type=str, default="rendered", choices=["rendered", "random_mix"])
    parser.add_argument("--model_size", type=str, default="Large", choices=["Large", "Small"])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument("--get_idx", type=int, default=1, help="1 ~ 1000, read the instructions in dataset.py -> EmbeddingLibraryDataset")
    parser.add_argument("--eval_avg", type=str, default="macro", choices=["micro", "macro", "weighted"])

    # "/data3/aiproducer_inst/f_emb/submission/multi_inst_emb"
    parser.add_argument('--nlakh_dataset_dir', type=str, default=None, help="Path to the rendered Nlakh multi dataset directory.")
    # "/data4/aiproducer_inst/rendered_single_inst/"
    parser.add_argument('--random_dataset_dir', type=str, default=None, help="Path to the rendered Nlakh single dataset directory.")
    # f'/data3/aiproducer_inst/f_emb/submission/single_inst_emb/{split}_default_nfft/*/'
    parser.add_argument('--single_inst_emb_dir', type=str, default=None, required=True, help="Path to the output embeddings of single instruments processed with trained Single Instrument Encoder.")
    parser.add_argument("--checkpoint_dir", type=str, default="/data3/aiproducer_inst/haessun_models/Multi_Instrument_Encoder")

    parser.add_argument('--wandb', type=bool, default=True, help="Make it False when debugging.")
    parser.add_argument('--project_name', type=str, default='Multi Instrument Encoder')

    args = parser.parse_args()
    return args

def train(model, train_loader, optimizer, pit_func, epoch, args, DEVICE):
    model.train()
    train_loss = []

    for batch_idx, data in tqdm(enumerate(train_loader)):
        mix_audio, emb_list, inst_idxes, track_len = data[0].to(DEVICE), data[1].to(DEVICE), data[2], data[3].to(DEVICE)

        optimizer.zero_grad()
        output = model(mix_audio.squeeze().unsqueeze(dim=1).type(torch.float32))
        output = torch.reshape(output, (output.size()[0], 9, -1)) # (batch_size, 1024 * 9) -> (batch_size, 9, 1024)

        loss = pit_func(output, emb_list, track_num=track_len).to(DEVICE)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    
    train_loss = np.mean(train_loss)
    if args.wandb:
        wandb.log({
            "train_loss": train_loss,
            "epoch": epoch,
            "batch_idx": batch_idx
        })
    torch.save(model.state_dict(), f"{args.checkpoint_dir}/{args.dataset}/{epoch}_trLoss_{train_loss:.3f}.pt")
    return

def evaluate(model, valid_loader, lib, args, DEVICE):
    model.eval()
    loss_list = []

    est_list = []
    ans_list = []
    ran_list = []
    score_list = []
    cos_loss = nn.CosineEmbeddingLoss(reduction='none')
    cos_sim = nn.CosineSimilarity()

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(valid_loader)):
            mix_audio, emb_list, inst_idxes, track_len = data[0].to(DEVICE), data[1].to(DEVICE), data[2], data[3].to(DEVICE)
            
            # output = model(mix_audio.squeeze())
            output = model(mix_audio.squeeze().unsqueeze(dim=1).type(torch.float32))
            output = torch.reshape(output, (output.size()[0], 9, -1)) # (batch_size, 1024 * 9) -> (batch_size, 9, 1024)
            
            loss = pit_func(output, emb_list, track_num=track_len).to(DEVICE)
            loss_list.append(loss.item())

            """ f1 score """     
            #### estimated one hot ####
            for i in range(mix_audio.size()[0]):
                est_one_hot = np.zeros(53, dtype=int)
                for idx, est in enumerate(output[i]):
                    loss = cos_loss(est.repeat((53, 1)), lib, torch.ones(1).to(DEVICE))
                    min_idx = torch.argmin(loss)
                    est_one_hot[min_idx] = 1
                est_list.append(est_one_hot)
            
            #### estimated cos score ####
            for i in range(mix_audio.size()[0]):
                est_score = np.zeros((9, 53))
                for idx, est in enumerate(output[i]):
                    est_score[idx] = cos_sim(est.repeat((53, 1)), lib).detach().cpu() # 53
                est_score = np.max(est_score, axis=0)
                score_list.append(est_score)

            #### random one hot ####
            for i in range(mix_audio.size()[0]):
                random_one_hot = np.zeros(53, dtype=int)
                num_inst = random.randint(2, 9)
                for i in random.sample(range(53), num_inst):
                    random_one_hot[i] = 1
                ran_list.append(random_one_hot)

            #### answer one hot ####
            inst_idxes = inst_idxes.cpu().detach().numpy()
            for b, l in enumerate(track_len):
                ans_one_hot = np.zeros(53, dtype=int)
                inst_tmp = [i for i in inst_idxes[b] if not i < 0]
                ans_one_hot[inst_tmp] = 1
                ans_list.append(ans_one_hot)

        est_list = np.asarray(est_list)
        ans_list = np.asarray(ans_list)
        ran_list = np.asarray(ran_list)
        score_list = np.asarray(score_list)

        f1 = f1_score(ans_list, est_list, average=args.avg)
        f1_random = f1_score(ans_list, ran_list, average=args.avg)
        recall = recall_score(ans_list, est_list, average=args.avg)
        recall_random = recall_score(ans_list, ran_list, average=args.avg)
        precision = precision_score(ans_list, est_list, average=args.avg)
        precision_random = precision_score(ans_list, ran_list, average=args.avg)
        mAP = average_precision_score(ans_list, score_list, average=args.avg)
        print("F1 score : {:.3f}, Recall : {:.3f}, Precision : {:.3f}, mAP : {:.3f}".format(f1, recall, precision, mAP))

        if args.wandb:
            wandb.log({
                "Valid F1" : f1,
                "Valid F1_random" : f1_random,
                "Valid Recall" : recall,
                "Valid Recall_random" : recall_random,
                "Valid Precision" : precision,
                "Valid Precision_random" : precision_random,
                "Valid mAP" : mAP,
                "Valid Loss" : sum(loss_list) / len(loss_list)
            })
    return

if __name__=='__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))
    print("Using Dataset: {}".format(args.dataset))
    
    if args.model_size == "Large":
        model = timm.create_model('convnext_small_in22ft1k', pretrained=True, in_chans=1, num_classes=9 * 1024, drop_path_rate=0.5).cuda()
    elif args.model_size == "Small":
        model = ConvNet().cuda()
    model = nn.DataParallel(model).to(DEVICE)

    if args.wandb:
        wandb.init(
            project = args.project_name,
            name = f"Multi-Instrument-Encoder, model_size={args.model_size}, dataset={args.dataset}, batch={args.batch_size}, lr={args.lr}",
        )
        wandb.config = {
            "learning_rate" : args.lr,
            "batch_size" : args.batch_size
        }
        wandb.watch(model)

    pit_func = PITLossWrapper(nn.CosineSimilarity(), pit_from='pw_pt')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.dataset == "rendered":
        train_dataset = RenderedNlakhDataset(args.nlakh_dataset_dir, "train", args.loss)
    elif args.dataset == "random_mix":
        train_dataset = RandomMixMultiInstrumentDataset(audio_path=args.random_dataset_dir, single_inst_emb_path=args.single_inst_emb_dir, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    valid_dataset = RenderedNlakhDataset(data_path = args.data_dir, split = "valid")
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # build target embedding library for multi_inst_encoder with 1 sample of given idx for each unseen instrument class.
    emb_lib_dataset = EmbeddingLibraryDataset(path=args.single_inst_emb_dir, split='valid', get_idx=args.get_idx)
    emb_lib_loader = DataLoader(emb_lib_dataset, batch_size=1, shuffle=False)
    lib = torch.zeros((53, 1024)).to(DEVICE)
    for idx, emb in tqdm(enumerate(emb_lib_loader)):
        lib[idx] = emb

    for epoch in range(args.num_epochs):
        train(model, train_loader, optimizer, pit_func, epoch, args, DEVICE)
        evaluate(model, valid_loader, lib, args, DEVICE)

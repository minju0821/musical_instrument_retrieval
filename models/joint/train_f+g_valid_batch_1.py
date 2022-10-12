import time
import wandb
import random
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from dataset import Single_Mix_Mel_dataset, FewShotReprDataset, OneShotReprDataset
from torch.utils.data import DataLoader
from loss.asteroid_mse import PairwiseMSE
from loss.PIT_loss import PITLossWrapper
from models.deep_cnn import ConvNet, ConvNet_eval, EER
# from models.convnext import ConvNeXt
from sklearn.metrics import f1_score, precision_score, recall_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='f_g_joint')
    parser.add_argument('--wandb', type=bool, default=True)
    
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--loss", type=str, default="Cosine")
    parser.add_argument("--metric_avg", type=str, default='micro')

    args = parser.parse_args()
    
    return args

def train(models, opts, loss, train_loader, epoch, args, DEVICE):
    tqdm_bar = tqdm(enumerate(train_loader))
    for batch_idx, (mix_audio, single_list, single_idxes, track_len) in tqdm_bar:
        models['f'].train()
        models['g'].train()
        for opt in opts:
            opt.zero_grad()

        mix_audio, single_list, single_idxes, track_len = mix_audio.to(DEVICE), single_list.to(DEVICE), single_idxes.to(DEVICE), track_len.to(DEVICE)

        """ TRAIN """
        # single_list (batch_size * 9,  128, 157)
        f_out, f_l1_out = models['f'](single_list)
        # f_out (batch_size * 9, 953)
        f_out = torch.reshape(f_out, (32, 9, -1))
        f_l1_out = torch.reshape(f_l1_out, (32, 9, -1))
        # f_out (batch_size, 9, 953)

        total_track_num = torch.sum(track_len)
        f_out_concat = torch.zeros((total_track_num, 953))

        idx = 0
        for i, l in enumerate(track_len):
            f_out_concat[idx:idx+l] = f_out[i][:l]
            idx += l

        g_out = models['g'](mix_audio)[0]
        g_out = torch.reshape(g_out, (g_out.size()[0], 9, -1))

        """ Compute Loss """
        f_loss_class = loss['f'](f_out_concat.to(DEVICE), single_idxes)
        if args.loss == "Cosine":
            g_loss = loss['g'](g_out, f_l1_out, track_num=track_len, target=torch.ones(mix_audio.size()[0]).to(DEVICE))
        else:
            g_loss = loss['g'](g_out, f_l1_out, track_num=track_len)

        total_loss = f_loss_class + 2 * g_loss
        total_loss.backward()
        
        if args.wandb:
            wandb.log({
                "Epoch" : epoch,
                "Iter." : batch_idx,
                "f_enc loss" : f_loss_class.item(),
                "g_enc loss" : g_loss.item(),
                "total loss" : total_loss.item()
            })
        
        for opt in opts:
            opt.step()
    
    torch.save(models['f'].state_dict(), f'/data4/aiproducer_inst/haessun_models/f+g_joint/epoch_{epoch}_totalLoss_{total_loss}.pt')
    
def evaluate(models, losses, valid_loader, lib_loader, epoch, args, DEVICE):
    models['f'].eval()
    models['g'].eval()

    with torch.no_grad():
        model_eval = ConvNet_eval(out_classes=953).to(DEVICE)
        model_eval.load_state_dict(dict(list(models['f'].state_dict().items())[:-2]))
        model_eval.eval()
        model_eval = model_eval.forward

        eer = EER(model_eval, DEVICE)
        eer_score, threshold = eer.evaluate()

        wandb.log({
            "Valid EER" : eer_score,
            "Valid EER_thres." : threshold,
        })

        lib = torch.zeros((53, 1024)).to(DEVICE)
        for idx, repr in tqdm(enumerate(lib_loader)):
            lib[idx] = repr

        est_list = []
        ans_list = []
        ran_list = []

        cos_loss = nn.CosineEmbeddingLoss(reduction='none')
        tqdm_bar = tqdm(enumerate(valid_loader))
        for idx, (mix_audio, single_list, single_idxes, track_len) in tqdm_bar:
            mix_audio, single_list, single_idxes, track_len = mix_audio.to(DEVICE), single_list.to(DEVICE), single_idxes[:track_len].to(DEVICE), track_len.to(DEVICE)

            ## f_enc로 lib를 만들어야하는데.. valid 다 돌면서 one-shot or few-shot으로 만들어야?
            # f_out = models['f'](single_list)

            g_out = models['g'](mix_audio)[0]
            g_out = torch.reshape(g_out, (9, -1))
            #### estimated one hot ####
            est_one_hot = np.zeros(53, dtype=int)
            for idx, est in enumerate(g_out[0]):
                loss = cos_loss(est.repeat((53, 1)), lib, torch.ones(1).to(DEVICE))
                min_idx = torch.argmin(loss)
                est_one_hot[min_idx] = 1
            est_list.append(est_one_hot)

            #### random one hot ####
            random_one_hot = np.zeros(53, dtype=int)
            num_inst = random.randint(2, 9)
            for i in random.sample(range(53), num_inst):
                random_one_hot[i] = 1
            ran_list.append(random_one_hot)

            #### answer one hot ####
            ans_one_hot = np.zeros(53, dtype=int)
            ans_one_hot[single_idxes.cpu().detach().numpy()] = 1
            ans_list.append(ans_one_hot)

        est_list = np.asarray(est_list)
        ans_list = np.asarray(ans_list)
        ran_list = np.asarray(ran_list)

        f1 = f1_score(ans_list, est_list, average=args.metric_avg)
        f1_random = f1_score(ans_list, ran_list, average=args.metric_avg)
        recall = recall_score(ans_list, est_list, average=args.metric_avg)
        recall_random = recall_score(ans_list, ran_list, average=args.metric_avg)
        precision = precision_score(ans_list, est_list, average=args.metric_avg)
        precision_random = precision_score(ans_list, ran_list, average=args.metric_avg)

        wandb.log({
            "Valid F1" : f1,
            "Valid F1_random" : f1_random,
            "Valid Recall" : recall,
            "Valid Recall_random" : recall_random,
            "Valid Precision" : precision,
            "Valid Precision_random" : precision_random,
        })

    return

if __name__=='__main__':

    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    if args.wandb:
        wandb.init(
            project = args.project_name,
            name = "f+g : batch_{}, loss_{}".format(args.batch_size, args.loss),
        )
        wandb.config = {
            "learning_rate" : args.lr,
            "batch_size" : args.batch_size
        }
    
    f_enc = ConvNet(out_classes=953).to(DEVICE)
    g_enc = ConvNet(out_classes=1024*9).to(DEVICE)
    models = {'f': f_enc, 'g': g_enc}

    f_loss = nn.CrossEntropyLoss()
    if args.loss == "Cosine":
        g_loss = PITLossWrapper(nn.CosineEmbeddingLoss(), pit_from='pw_pt')
    elif args.loss == "MSE":
        g_loss = PITLossWrapper(PairwiseMSE(), pit_from='pw_mtx')
    else:
        raise ValueError("metric must be either cosine or MSE.")
    losses = {'f': f_loss, 'g': g_loss}

    f_opt = torch.optim.Adam(f_enc.parameters(), lr=args.lr)
    g_opt = torch.optim.Adam(g_enc.parameters(), lr=args.lr)
    opts = [f_opt, g_opt]

    train_dataset = Single_Mix_Mel_dataset(split='train', loss='Cosine')
    valid_dataset = Single_Mix_Mel_dataset(split='valid', loss='Cosine')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=valid_dataset.collate_fn)

    lib_dataset = OneShotReprDataset(split='valid', get_idx=100)
    lib_loader = DataLoader(lib_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    torch.cuda.empty_cache()

    epoch = 0
    while True:
        evaluate(models, losses, valid_loader, lib_loader, epoch, args, DEVICE)
        train(models, opts, losses, train_loader, epoch, args, DEVICE)
    
        epoch += 1

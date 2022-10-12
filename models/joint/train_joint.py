import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

from models.convnext import ConvNeXt
from models.convnext import ConvNeXt
from models.g_enc import EncG
# from metric import EER
from eval2 import InstrumentDataset_eval, EER
from data2 import RenderedMultiInstrumentDataset


import numpy as np
import argparse
import wandb
from tqdm import tqdm
from glob import glob
from datetime import datetime
import itertools
import os ; opj=os.path.join

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='conformer2')
    # parser.add_argument('--0', type=str, default=None)
    parser.add_argument('--nowstr', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--wandb-id', type=str, default=None)
    parser.add_argument('--memo', type=str, default='')

    parser.add_argument('--nsynth-path', type=str, default="/disk2/aiproducer_inst/nsynth/")    
    parser.add_argument('--num-classes', type=int, default=953)
    
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--gpu', type=int, default=9)
    # parser.add_argument('--split', type=str, choices=['train', 'valid', 'test'])
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--ep', type=int, default=0)
    parser.add_argument('--lr', type=int, default=5e-5)
    parser.add_argument('--base-dir', type=str, default='./result_joint')
    parser.add_argument('--ckpt-per', type=int, default=2)
    
    
    
    
    """ Validation EER"""
    parser.add_argument('--valid-per', type=int, default=100)
    parser.add_argument('--num-eval', type=int, default=20)

    args = parser.parse_args()
    
    return args

EPS = torch.finfo(float).eps

def train(args):

    if args.nowstr is None:
        if args.debug:
            save_dir = 'debug'
        else:
            args.nowstr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            save_dir = '_'.join([args.nowstr, args.run_name]) if args.run_name != None else args.nowstr
        args.save_dir = opj(args.base_dir, save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        for d in ['train', 'valid', 'ckpt', 'wandb']:
            os.makedirs(opj(args.save_dir, d), exist_ok=True)
    else:
        ckpt_dir = opj(args.base_dir, args.nowstr, 'ckpt')
        checkpoint = torch.load(opj(ckpt_dir, str(args.ep).zfill(5)+'.pt'))
        net.load_state_dict(checkpoint)  
        
        save_dir = '_'.join([args.nowstr, args.run_name]) if args.run_name != None else args.nowstr
        args.save_dir = opj(args.base_dir, args.nowstr)
        
    if not args.debug:
        os.environ['WANDB_START_METHOD'] = 'thread'
        wandb.init(
            project='inst_recognition',
            id=save_dir+args.memo,
            group=args.group,
            dir = opj(args.save_dir),
            resume=False if args.wandb_id == None else True,
            config=args
        )
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    insts = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed',
            'string', 'synth_lead', 'vocal']
    
    label_idx = list(range(args.num_classes))
    
    train_dataset = RenderedMultiInstrumentDataset(split='train', load_embeddings=False)
    valid_dataset = RenderedMultiInstrumentDataset(split='valid', load_embeddings=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_train)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_valid)

    net_f = ConvNeXt(num_classes=953, in_chans=1)
    net_h = ConvNeXt(num_classes=953, in_chans=1)
    net_g = EncG()

    net_f.to(device)
    net_h.to(device)
    net_g.to(device)
    nets = [net_f, net_h, net_g]
    
    loss_fn_embed = nn.MSELoss()
    loss_fn_class = nn.BCELoss()
    act_fn = nn.Softmax(dim=-1)

    optim_f = torch.optim.Adam(net_f.parameters(), lr=args.lr)
    optim_h = torch.optim.Adam(net_h.parameters(), lr=args.lr)
    optim_g = torch.optim.Adam(net_g.parameters(), lr=args.lr)
    optims = [optim_f, optim_h, optim_g]
    
    for epoch in range(0, 10000):
        print('Epoch start:', epoch)
        if not args.debug:
            wandb.log({'epoch': epoch})
        """ Train loop"""
        for net in nets:
            net.train()
        for idx, data in enumerate(tqdm(train_loader, desc='Train loop')):
            for optim in optims:
                optim.zero_grad()
                
            mix_batch = data['mix_batch'].to(device)
            tracks_batch = data['tracks_batch'].to(device)
            inst_labels = data['inst_labels'].to(device)
            
            num_insts = data['num_insts']  # list, 길이=batch
            num_batch = len(num_insts)
                    
            out_f_emb, out_f = net_f(tracks_batch, return_all=True)
            out_h_emb = net_h(mix_batch, return_embed=True)
            
            out_f_embeds_list = torch.split(out_f_emb, num_insts)
            out_f_embed_sums = torch.vstack([torch.sum(embeds, dim=0) for embeds in out_f_embeds_list])
            
            out_f_softmax = act_fn(out_f)
            
            out_g = net_g(out_h_emb)  # batch개. -> extend 해야 함. 각각 num_inst개수 만큼. 
            
            loss_pit_min_loss = 0
            ## 문제점: 4~이상의 악기 분리만 학습하면 나중에 1~2개 분리는 못하지 않나?
            for j, out_f_embeds in enumerate(out_f_embeds_list):  # batch 개수만큼 iterate
                out_temp = out_g[j].repeat(num_insts[j], 1)
                distances = torch.norm(torch.abs(out_temp - out_f_embeds) + EPS, dim=-1)
                min_distance = torch.min(distances)
                loss_pit_min_loss += min_distance / out_temp.size(-1)
            loss_pit_min_loss = loss_pit_min_loss / num_batch
            
            loss_class = loss_fn_class(out_f_softmax, inst_labels)
            loss_embed = loss_fn_embed(out_f_embed_sums, out_h_emb)
            
            total_loss = 100 * loss_class + loss_embed + 200 * loss_pit_min_loss
            if not args.debug:
                wandb.log({'train loss class (f)': loss_class,
                           'train loss_embed (h)': loss_embed,
                           'train loss_pit_min_loss (g)' : loss_pit_min_loss})
                
            total_loss.backward()
            for optim in optims:
                optim.step()

            ## F Classification accuracy
            max_idx = torch.argmax(out_f_softmax, dim=-1, keepdim=True)
            f_one_hot = torch.zeros_like(out_f_softmax)
            f_one_hot.scatter_(-1, max_idx, 1)
            acc_f_batch = accuracy_score(inst_labels.detach().cpu(), f_one_hot.detach().cpu())
            
            if not args.debug:
                wandb.log({'train f accuracy batch': acc_f_batch})
            
            if args.debug:
                if idx > 3:
                    break
                    
        """ Valid loop """
        # if epoch==0:
        #     continue
        
        for net in nets:
            net.eval()
        #         sample, sr = torchaudio.load(path)
        # sample = torchaudio.transforms.Resample(sr, sample_rate)(sample)
        # Load sample wavefiles from single inst valid set. 
        path_single = '/data4/aiproducer_inst/rendered_single_inst/test'
        num_total_inst = 53  # 전체 악기 개수
        num_per_inst = 6  # f_enc에 넣을 음원 악기당 개수
        
        ref_wavs = torch.zeros((num_total_inst, num_per_inst, 80000))
        for i in tqdm(range(num_total_inst), desc='Valid: Loading reference wavs'):
            for j, file in enumerate(glob(opj(path_single, str(i+1), '*.wav'))[:num_per_inst]):  # 5개 랜덤하게 뽑도록 고쳐야 함.
                wav, sr = torchaudio.load(file)
                ref_wavs[i, j] = wav[0]
                    
        with torch.no_grad():
            rest_list= []
            true_inst_labels_list = []
            est_inst_labels_list = []
            
            for idx, data in enumerate(tqdm(valid_loader, desc='Valid loop')):  # batch = 1
                mix = data['mix_batch'].to(device)
                tracks = data['tracks_batch'].to(device)
                num_insts = data['num_insts']
                # inst_idces = data['inst_idces']
                true_inst_labels = data['inst_labels'].sum(dim=0)
                
                # out_f_embeds = [net_f(track, return_embed=True) for track in tracks]  # "Pseudo-Targets"
                ref_wavs = ref_wavs.view(-1, 80000).to(device)
                out_f_embeds = net_f(ref_wavs, return_embed=True)  # "Pseudo-Targets"    (53 * 5, 1024)
                # 이것과 est의 거리를 재야하는데, 평균과의 거리를 잴지, 거리의 평균을 잴지 정해야함. 지금은 거리의 평균으로 구현
            
                out_h_emb = net_h(mix, return_embed=True)  # 1개 나옴 (batch=1)
                
                # while True:  # Ideal... But...
                #     est_f = net_g(out_h_emb)
                #     out_h_emb = out_h_emb - est_f
                #     print(torch.norm(out_h_emb))
                #     if torch.norm(out_h_emb) < threshold:
                #         break
                
                est_inst_list = []
                est_labels = torch.zeros(num_total_inst)
                for i in range(num_insts[0]):  #  batch=1이므로.
                    est_f = net_g(out_h_emb)  # 53개의 embedding중에서, 무언가와 같음. (1, 1024)
                    # est_f 와 53개의 f_emb비교하고, 가장 가까운 거 뽑기.
                    est_f_repeat = est_f.repeat(num_total_inst * num_per_inst, 1)  # (53 * 5 , 1024)
                    distances = torch.norm(est_f_repeat - out_f_embeds, dim=-1).view(num_total_inst, num_per_inst).mean(dim=-1)
                    est_inst = torch.argmin(distances).detach().cpu().numpy()
                    est_labels[est_inst] = 1  # 여러번 나와도 한번만 1
                    out_h_emb = out_h_emb - est_f
                
                
                ## 결과값들
                rest = torch.norm(out_h_emb).detach().cpu().numpy()
                sample_f1_score = f1_score(y_true=true_inst_labels, y_pred = est_labels, average=None)
                
                rest_list.append(rest)
                true_inst_labels_list.append(true_inst_labels.type(torch.int8))
                est_inst_labels_list.append(est_labels.type(torch.int8))
                
                if args.debug:
                    if idx >= 3:
                        break
                    
            true_inst_labels_list = torch.vstack(true_inst_labels_list)
            est_inst_labels_list = torch.vstack(est_inst_labels_list)

            f1_macro = f1_score(true_inst_labels_list, est_inst_labels_list, average='macro')
            f1_micro = f1_score(true_inst_labels_list, est_inst_labels_list, average='micro')

            rest_mean = np.mean(rest_list)
            rest_std = np.sqrt(np.var(rest_list))
            
            if not args.debug:
                wandb.log({'valid rest norm mean':rest_mean,
                           'valid rest norm std': rest_std,
                           'valid f1 macro': f1_macro,
                           'valid f1 micro': f1_micro
                           })


def collate_fn_train(samples):
    mix_batch_list = [sample['mix'] for sample in samples]
    mix_batch = torch.vstack(mix_batch_list)
    
    tracks_batch_list = [sample['tracks'] for sample in samples]  # tracks : torch tensor
    tracks_batch = torch.vstack(tracks_batch_list)
    
    num_insts = [len(sample['tracks']) for sample in samples]
    # num_insts = torch.tensor(num_insts)
    
    inst_names = [sample['inst_names'] for sample in samples]
    inst_names = list(itertools.chain(*inst_names))
    inst_idces = torch.tensor([int(s[:-2]) for s in inst_names])
    inst_labels = torch.eye(953, dtype=torch.float32)[inst_idces]
    
    return dict(mix_batch = mix_batch,  tracks_batch = tracks_batch, num_insts=num_insts, inst_labels = inst_labels)

def collate_fn_valid(samples):
    mix_batch_list = [sample['mix'] for sample in samples]
    mix_batch = torch.vstack(mix_batch_list)
    
    tracks_batch_list = [sample['tracks'] for sample in samples]  # tracks : torch tensor
    tracks_batch = torch.vstack(tracks_batch_list)
    
    num_insts = [len(sample['tracks']) for sample in samples]
    # num_insts = torch.tensor(num_insts)
    
    inst_names = [sample['inst_names'] for sample in samples]
    inst_names = list(itertools.chain(*inst_names))
    inst_idces = torch.tensor([int(s[:-2]) for s in inst_names])
    inst_labels = torch.eye(53, dtype=torch.float32)[inst_idces]
    
    return dict(mix_batch = mix_batch,  tracks_batch = tracks_batch, num_insts=num_insts, inst_idces=inst_idces, inst_labels = inst_labels)

def cal_f1_scores(est, true):
    return None
    
def flatten(lst):
    result = []
    for item in lst:
        result.extend(item)
    
    
if __name__=='__main__':
    args = parse_args()
    train(args)
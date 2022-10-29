import os
import time
import random
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import wandb
import numpy as np

from models import ConvNet
from dataset import RenderedInstrumentDataset
from evaluation import EER

random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=953)
    parser.add_argument('--num_epochs', type=int, default=10)
    
    parser.add_argument('--gpus', type=str, default='0, 1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='/data3/aiproducer_inst/haessun_models/single_inst_encoder/')

    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='submission')

    args = parser.parse_args()
    return args

def train(model, train_loader, optimizer, loss_func, epoch, args, DEVICE):
    model.train()
    
    correct = 0
    data_len = 0
    train_loss = []

    for idx, (data, labels) in tqdm(enumerate(train_loader)):
        inputs, labels = data.to(DEVICE), labels.to(DEVICE)
        
        model.train()
        optimizer.zero_grad()

        out = model(inputs.type(torch.float32))[0]
        loss = loss_func(out, labels)
        loss.backward()

        train_loss.append(loss.item())

        prediction = out.max(1, keepdim = True)[1]
        correct += prediction.eq(labels.view_as(prediction)).sum().item()
        data_len += labels.size()[0]

        optimizer.step()
    
    train_loss = np.mean(train_loss)
    train_acc = 100 * correct / data_len
    print("Train Loss : {:.3f}, Train Acc. : {:.3f}".format(train_loss, train_acc))

    model_name = '{}epoch_{}_trLoss_{:.3f}.pt'.format(args.save_dir, epoch, train_loss)
    torch.save(model.state_dict(), model_name)

    if args.wandb:
        wandb.log({
            "Epoch" : epoch,
            "Iteration" : idx,
            "Train Loss" : train_loss,
            "Train Acc." : train_acc,
        })
    
def evaluate(model, args, DEVICE):
    model.eval()

    with torch.no_grad():
        eer = EER(model, DEVICE)
        eer_score, threshold = eer.evaluate()
        print(f'EER : {eer_score}, EER_thres. : {threshold}')

    if args.wandb:
        wandb.log({
            "Valid EER" : eer_score,
            "Valid EER_thres." : threshold,
        })

if __name__=='__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))
    torch.cuda.empty_cache()

    model = ConvNet(args.num_class).cuda()
    model = nn.DataParallel(model).to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    
    batch_size = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.wandb:
        wandb.init(
            project = args.project_name,
            name = "Single-Instrument-Encoder, class={}, batch={}, lr={}".format(args.num_class, args.batch_size, args.lr),
        )
        wandb.config = {
            "learning_rate" : args.lr,
            "batch_size" : args.batch_size
        }
        wandb.watch(model)

    train_dataset = RenderedInstrumentDataset(split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    for epoch in range(args.num_epochs):
        train(model, train_loader, optimizer, loss_func, epoch, args, DEVICE)
        evaluate(model, args, DEVICE)

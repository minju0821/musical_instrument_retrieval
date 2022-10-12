import glob
import scipy
import random
import librosa
import argparse
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.metrics import roc_curve

import torch
import torch.nn as nn

# from data import InstrumentDataset
from render_data import RenderedInstrumentDataset
# from evaluation.graph_eer_render import EER

import time
import wandb

random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=953)
    parser.add_argument('--project_name', type=str, default=None)
    
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    
    return args

class ConvNet(nn.Module):
    def __init__(self, out_classes):
        super(ConvNet, self).__init__()

        self.conv1 = self.seq(1, 32)
        self.conv2 = self.seq(32, 64)
        self.conv3 = self.seq(64, 128)
        self.conv4 = self.seq(128, 256).append(
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.pool_drop = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout2d(p=0.25)
        )

        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=1024),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Dropout(p=0.5)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=out_classes),
            # nn.LeakyReLU(negative_slope=0.33),
        )

        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.conv4.apply(self.init_weights)
        self.linear1.apply(self.init_weights)
        self.linear2.apply(self.init_weights)

    def seq(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.33)
        )

    # for all conv. & linear layers w/ zero biases
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
    
    def forward(self, x):
        x = x.unsqueeze(dim=1).permute(0, 1, 3, 2)
        
        x = self.pool_drop(self.conv1(x.float()))
        x = self.pool_drop(self.conv2(x))
        x = self.pool_drop(self.conv3(x))

        x = self.conv4(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return x

def train(net, train_loader, optimizer, epoch, DEVICE, loss_func, batch_size):
    net.train()
    
    correct = 0
    train_loss = []
    loading_time = []
    training_time = []

    start = time.time()

    tqdm_bar = tqdm(enumerate(train_loader))
    for idx, (data, labels) in tqdm_bar:
        loading_time.append(time.time() - start)

        inputs, labels = data.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        start_train = time.time()
        out = net(inputs)
        training_time.append(time.time() - start_train)

        loss = loss_func(out, labels)
        loss.backward()

        train_loss.append(loss.item())

        prediction = out.max(1, keepdim = True)[1]
        correct_tmp = prediction.eq(labels.view_as(prediction)).sum().item()
        correct += correct_tmp

        optimizer.step()

        tqdm_bar.set_description("Epoch {} - train loss: {:.3f}  train acc: {:.3f}".format(epoch, loss.item(), (100. * correct_tmp / batch_size)))

        start = time.time()

        # on every 300 iteration and on last iteration
        if idx % 300 == 0:

            train_loss_wandb = sum(train_loss) / len(train_loss)

            """ 
                total 29782 iterations in 1 epoch
                29782 / 300 = 99.26
                953000 - (99 * 300 * 32) = 2600
            """
            if idx % 300 == 0:
                train_acc = 100. * correct / (300 * batch_size)
            elif idx == len(train_loader):
                train_acc = 100. * correct / (2600)
            
            wandb.log({
                "Epoch" : epoch,
                "Iteration" : idx,
                "Train Acc." : train_acc,
                "Train Loss" : train_loss_wandb,
                "Data Loading Time" : np.mean(loading_time),
                "Training Time" : np.mean(training_time)
            })

            torch.save(net.state_dict(), '/home/haessun/ai_prod/models/f_enc_rendered_02/class{}_epoch{}_iter{}_trLoss_{:.3f}_trAcc_{:.3f}'.format(953, epoch, idx, train_loss_wandb, train_acc))

            ## reset variables ##
            loading_time = []
            training_time = []
            train_loss = []
            correct = 0



    train_loss_wandb = sum(train_loss) / len(train_loss)
    train_acc = 100. * correct / (2600)
    
    wandb.log({
        "Epoch" : epoch,
        "Iteration" : idx,
        "Train Acc." : train_acc,
        "Train Loss" : train_loss_wandb,
        "Data Loading Time" : np.mean(loading_time),
        "Training Time" : np.mean(training_time)
    })

    torch.save(net.state_dict(), '/home/haessun/ai_prod/models/f_enc_rendered_02/class{}_epoch{}_iter{}_trLoss_{:.3f}_trAcc_{:.3f}'.format(953, epoch, idx, train_loss_wandb, train_acc))

    return train_loss, train_acc

if __name__=='__main__':

    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    net = ConvNet(out_classes=args.num_class).to(DEVICE)

    wandb.init(
        project = args.project_name,
        name = "w/o valid class={}, batch{}, lr={}".format(args.num_class, args.batch_size, args.lr),
    )
    wandb.config = {
        "learning_rate" : args.lr,
        "batch_size" : args.batch_size
    }
    wandb.watch(net)

    batch_size = args.batch_size
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_dataset = RenderedInstrumentDataset(split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

    torch.cuda.empty_cache()

    epoch = 0
    while True:
        train_loss, train_acc = train(net, train_loader, optimizer, epoch, DEVICE, loss_func, batch_size)

        epoch += 1

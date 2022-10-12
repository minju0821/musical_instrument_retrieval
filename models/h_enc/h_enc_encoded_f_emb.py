import random
import argparse
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn

import time
import wandb
from h_enc_dataset import EmbeddingDataset

random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default=None)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--loss', type=str, default='Cosine')

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
            nn.LeakyReLU(negative_slope=0.33),
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

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
    
    def forward(self, x):
        # (batch, 1, 157, 128)
        x = x.unsqueeze(dim=1).permute(0, 1, 3, 2)
        
        x = self.pool_drop(self.conv1(x.float()))
        x = self.pool_drop(self.conv2(x))
        x = self.pool_drop(self.conv3(x))

        x = self.conv4(x)
        x_1 = self.linear1(x)
        x_2 = self.linear2(x_1)

        return x_1, x_2

def train(net, train_loader, optimizer, epoch, DEVICE, summed_loss, BCE_loss, loss_name, valid_loader):
    train_loss = []
    summed_loss_list = []
    BCE_loss_list = []
    net.train()
    tqdm_bar = tqdm(enumerate(train_loader))

    loading_time = []
    training_time = []

    start = time.time()

    for idx, (input, labels, BCE_trg) in tqdm_bar:
        net.train()

        loading_time.append(time.time() - start)

        inputs, labels = input.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # train
        start_train = time.time()
        linear1, linear2 = net(inputs)
        training_time.append(time.time() - start_train)

        # compute summed loss
        if loss_name == 'MSE':
            summed_loss_out = summed_loss(linear1, labels)
        elif loss_name == 'Cosine':
            summed_loss_out = summed_loss(linear1, labels, torch.ones(inputs.size()[0]).to(DEVICE))

        # compute BCE loss
        bce_loss_out = BCE_loss(linear2, BCE_trg.to(DEVICE))

        loss = summed_loss_out + bce_loss_out
        loss.backward()

        train_loss.append(loss.item())
        summed_loss_list.append(summed_loss_out.item())
        BCE_loss_list.append(bce_loss_out.item())

        optimizer.step()

        tqdm_bar.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))

        start = time.time()

        if (idx+1) % 625 == 0:

            train_loss_wandb = sum(train_loss) / len(train_loss)
            
            wandb.log({
                "Train Loss" : train_loss_wandb,
                "Train Summed Loss" : sum(summed_loss_list) / len(summed_loss_list),
                "Train BCE Loss" : sum(BCE_loss_list) / len(BCE_loss_list),
                "Data Loading Time" : np.mean(loading_time),
                "Training Time" : np.mean(training_time),
                "Epoch" : epoch,
                "Iteration" : idx+1
            })

            torch.save(net.state_dict(), '/disk2/aiproducer_inst/haessun_models/h_enc/{}/h_enc_+BCE_scratch/epoch_{}_iter_{}_trLoss_{:.3f}'.format(loss_name, epoch, idx+1, train_loss_wandb))

            ## reset variables ##
            loading_time = []
            training_time = []
            train_loss = []

            evaluate(net, valid_loader, DEVICE, summed_loss, loss_name)

def evaluate(net, valid_loader, DEVICE, summed_loss, loss_name):
    start = time.time()

    net.eval()
    summed_loss_list = []

    with torch.no_grad():

        for idx, (inputs, labels, _) in tqdm(enumerate(valid_loader)):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            linear1, _ = net(inputs)

            # compute loss
            if loss_name == 'MSE':
                summed_loss_out = summed_loss(linear1, labels)
            elif loss_name == 'Cosine':
                summed_loss_out = summed_loss(linear1, labels, torch.ones(inputs.size()[0]).to(DEVICE))

            summed_loss_list.append(summed_loss_out.item())

    wandb.log({
        "Valid Summed Loss" : sum(summed_loss_list) / len(summed_loss_list),
        "Valid time" : time.time() - start
    })


if __name__=='__main__':

    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    net = ConvNet(out_classes=953).to(DEVICE)
    ## finetuning ##
    # loaded_dict = torch.load('/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class953_epoch8_iter1200_trLoss_0.300_trAcc_89.031', map_location = DEVICE)
    # loaded_dict = dict(list(loaded_dict.items())[:-2])
    # net.load_state_dict(loaded_dict, strict=False)

    wandb.init(
        project = args.project_name,
        name = "h_enc_rendered_+BCE_scratch : loss={}, batch{}, lr={}".format(args.loss, args.batch_size, args.lr),
    )
    wandb.config = {
        "learning_rate" : args.lr,
        "batch_size" : args.batch_size
    }
    wandb.watch(net)

    batch_size = args.batch_size
    # summed_loss = nn.CrossEntropyLoss()

    if args.loss == 'MSE':
        summed_loss = nn.MSELoss()
    elif args.loss == 'Cosine':
        summed_loss = nn.CosineEmbeddingLoss()
    BCE_loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_dataset = EmbeddingDataset(split='train_new')
    valid_dataset = EmbeddingDataset(split='valid_new')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    torch.cuda.empty_cache()

    epoch = 0
    while True:
        train(net, train_loader, optimizer, epoch, DEVICE, summed_loss, BCE_loss, args.loss, valid_loader)

        epoch += 1


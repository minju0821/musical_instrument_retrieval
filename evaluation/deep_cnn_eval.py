import glob
import random
import json
import argparse
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn
import librosa
import librosa.display

from data import InstrumentDataset

import time
import wandb

random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=11)
    parser.add_argument('--audio_len', type=float, default=5.)

    parser.add_argument('--dur_in_sec', type=float, default=10.)
    parser.add_argument('--min_notes', type=int, default=10)

    parser.add_argument('--activ', type=str, default='LReLU')
    parser.add_argument('--project_name', type=str, default=None)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--valid-per', type=int, default=10)

    args = parser.parse_args()
    
    return args

class ConvNet(nn.Module):
    def __init__(self, activ, out_classes):
        super(ConvNet, self).__init__()

        if activ=='LReLU':
            relu = nn.LeakyReLU(negative_slope=0.33)
        elif activ=='PReLU':
            relu = nn.PReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            relu,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            relu
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            relu,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            relu
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            relu,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            relu
        )

        self.pool_drop = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout2d(p=0.25)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            relu,
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            relu,
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=1024),
            relu,
            nn.Dropout(p=0.5)
        )

        # self.linear2 = nn.Sequential(
        #     nn.Linear(in_features=1024, out_features=out_classes),
        #     # relu,
        # )

        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.conv4.apply(self.init_weights)
        self.linear1.apply(self.init_weights)
        # self.linear2.apply(self.init_weights)

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
        # x = self.linear2(x)

        return x

def train(net, train_loader, optimizer, epoch, DEVICE, loss_func, wandb_tr_img, batch_size):
    train_loss = []
    correct = 0
    net.train()
    tqdm_bar = tqdm(enumerate(train_loader))

    loading_time = []
    training_time = []

    start = time.time()

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

        tqdm_bar.set_description("Epoch {} - train loss: {:.6f}  train acc: {:.6f}".format(epoch, loss.item(), (100. * correct_tmp / batch_size)))

        wandb_tr_img.append(wandb.Image(
            inputs[0], caption="Pred: {} Truth: {}".format(prediction[0].item(), labels[0])
        ))

        start = time.time()

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = 100. * correct / len(train_loader.dataset)

    wandb.log({
        "Examples": wandb_tr_img,
        "Train Acc" : train_acc,
        "Train Loss" : train_loss,
        "Data Loading Time" : np.mean(loading_time),
        "Training Time" : np.mean(training_time)
    })

    return train_loss, train_acc

def evaluate(net, test_loader, DEVICE, loss_func, wandb_val_img):

    net.eval()
    test_loss = []
    correct = 0

    with torch.no_grad():
        for inputs, label in tqdm(test_loader):
            inputs, label = inputs.to(DEVICE), label.to(DEVICE)
            
            out = net(inputs)
            test_loss.append(loss_func(out, label).item())

            prediction = out.max(1, keepdim=True)[1].to(DEVICE)
            # print(prediction.squeeze())
            correct += prediction.eq(label.view_as(prediction)).sum().item()

            wandb_val_img.append(wandb.Image(
                inputs[0], caption="Pred: {} Truth: {}".format(prediction[0].item(), label[0])
            ))

    test_loss = sum(test_loss) / len(test_loss)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    wandb.log({
        "Examples": wandb_val_img,
        "Valid Acc" : test_accuracy,
        "Valid Loss" : test_loss
    })

    return test_loss, test_accuracy, correct
    
if __name__=='__main__':

    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    net = ConvNet(activ=args.activ, out_classes=args.num_class).to(DEVICE)

    wandb.init(
        project = args.project_name,
        name = "class={}, audioLen={}, {}, batch{}, lr={}".format(args.num_class, args.audio_len, args.activ, args.batch_size, args.lr),
    )
    wandb.config = {
        "learning_rate" : args.lr,
        "epochs" : 300,
        "batch_size" : args.batch_size
    }
    wandb.watch(net)
    wandb_val_img = []
    wandb_tr_img = []

    lr = 0.001
    batch_size = args.batch_size
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # train_dataset = TrainDataset(num_classes=args.num_class, audio_len=args.audio_len)
    # valid_dataset = ValidDataset(num_classes=args.num_class, audio_len=args.audio_len)

    train_dataset = InstrumentDataset(split='train', duration_in_seconds=args.dur_in_sec, min_notes=args.min_notes, num_classes=args.num_class)
    # valid_dataset = InstrumentDataset(split='test', duration_in_seconds=args.dur_in_sec, min_notes=args.min_notes, num_classes=args.num_class)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    epoch = 0
    check_valid_loss = 0
    valid_loss_before = float('inf')
    # history = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'test_cor':[]}
    history = {'train_loss':[], 'train_acc':[]}

    torch.cuda.empty_cache()

    epoch = 0
    while True:
        train_loss, train_acc = train(net, train_loader, optimizer, epoch, DEVICE, loss_func, wandb_tr_img, batch_size)
        # valid_loss, valid_acc, correct = evaluate(net, valid_loader, DEVICE, loss_func, wandb_val_img)

        # print("\n[EPOCH: {}]. \tModel: ConvNet, \tEval Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(epoch, valid_loss, valid_acc))
        
        if epoch % 5 == 0:
            torch.save(net.state_dict(), './test_models_02/class{}_epoch{}_trLoss_{:.3f}_trAcc{:.3f}'.format(args.num_class, epoch, train_loss, train_acc))

        epoch += 1

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        # history['test_loss'].append(valid_loss)
        # history['test_acc'].append(valid_acc)
        # history['test_cor'].append(correct)

    f = open("./test_models/test_history.txt", 'w')
    f.write(json.dumps(history))
    f.close()
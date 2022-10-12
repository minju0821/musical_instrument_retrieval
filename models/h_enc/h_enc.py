import random
import argparse
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn

# from multi_data import MultiInstrumentDataset
from render_data import RenderedInstrumentDataset
from deep_cnn import ConvNet_eval as f_enc
# 나중에 이걸로 다시 돌리자.. batch, architecture, dataset 다 바뀜 // from model_batch32_newData
# from deep_cnn import ConvNet_new

import time
import wandb

random.seed(0)
torch.manual_seed(0)

# f_enc에서 (stem)의 각 single inst audio를 process하고, 이걸 다 더해줌 -> ground truth
#       -> 어떤 model 쓸지 정함
# h_enc에서 multi-audio(mix)를 process하고, ground truth랑 비교해서 loss 뽑아줌
#       -> 어떤 loss 쓸지 정함 

def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=953)

    parser.add_argument('--dur_in_sec', type=float, default=5.)
    parser.add_argument('--min_notes', type=int, default=5)

    parser.add_argument('--activ', type=str, default='LReLU')
    parser.add_argument('--project_name', type=str, default=None)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--loss', type=str, default='MSE')

    args = parser.parse_args()
    
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=953)
    parser.add_argument('--project_name', type=str, default=None)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--loss', type=str, default='Cosine')

    args = parser.parse_args()
    
    return args

class ConvNet(nn.Module):
    def __init__(self):
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

        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.conv4.apply(self.init_weights)
        self.linear1.apply(self.init_weights)

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
        x = self.linear1(x)

        return x

def train(net, train_loader, optimizer, epoch, DEVICE, loss_func, loss_name, _f_enc):
    train_loss = []
    net.train()
    tqdm_bar = tqdm(enumerate(train_loader))

    loading_time = []
    training_time = []
    f_emb_time = []

    start = time.time()

    for idx, (data, labels) in tqdm_bar:

        loading_time.append(time.time() - start)

        inputs = data.to(DEVICE)

        # init GT embedding
        GT_emb = torch.zeros(inputs.size()[0], 1, 1024).to(DEVICE)

        emb_start = time.time()

        # add all the single inst embedding
        for idx, label in enumerate(labels):
            GT_emb[idx] = torch.sum(_f_enc(label.to(DEVICE)), dim=0)
        GT_emb = GT_emb.squeeze()

        f_emb_time.append(time.time() - emb_start)

        optimizer.zero_grad()

        # train
        start_train = time.time()
        out = net(inputs)
        training_time.append(time.time() - start_train)

        # compute loss
        if loss_name == 'MSE':
            loss = loss_func(out, GT_emb)
        elif loss_name == 'Cosine':
            loss = loss_func(out, GT_emb, torch.ones(inputs.size()[0]).to(DEVICE))
        
        loss.backward()

        train_loss.append(loss.item())

        optimizer.step()

        tqdm_bar.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))

        start = time.time()

        if idx % 300 == 0:

            train_loss_wandb = sum(train_loss) / len(train_loss)
            
            wandb.log({
                "Train Loss" : train_loss_wandb,
                "Data Loading Time" : np.mean(loading_time),
                "Training Time" : np.mean(training_time)
            })

            torch.save(net.state_dict(), '/disk2/aiproducer_inst/haessun_models/h_enc/h_enc_rendered_00/epoch_{}_iter_{}_trLoss_{:.3f}'.format(epoch, idx, train_loss_wandb))

            ## reset variables ##
            loading_time = []
            training_time = []
            train_loss = []

    train_loss_wandb = sum(train_loss) / len(train_loss)       
    wandb.log({
        "Train Loss" : train_loss_wandb,
        "Data Loading Time" : np.mean(loading_time),
        "Training Time" : np.mean(training_time)
    })
    torch.save(net.state_dict(), '/disk2/aiproducer_inst/haessun_models/h_enc/h_enc_rendered_00/epoch_{}_iter_{}_trLoss_{:.3f}'.format(epoch, idx, train_loss_wandb))

    return train_loss

    
def evaluate(net, test_loader, DEVICE, loss_func):

    net.eval()
    test_loss = []
    correct = 0

    with torch.no_grad():

        for inputs, label in tqdm(test_loader):
            inputs, label = inputs.to(DEVICE), label.to(DEVICE)
            
            out = net(inputs)
            test_loss.append(loss_func(out, label).item())

            prediction = out.max(1, keepdim=True)[1].to(DEVICE)
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss = sum(test_loss) / len(test_loss)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    wandb.log({
        "Valid Acc" : test_accuracy,
        "Valid Loss" : test_loss
    })

    return test_loss, test_accuracy, correct
    

## f embedding 뽑아놓고 사용?! ##
   
if __name__=='__main__':

    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    net = ConvNet().to(DEVICE)

    wandb.init(
        project = args.project_name,
        name = "h_enc : class={}, batch{}, lr={}, loss={}".format(args.num_class,  args.batch_size, args.lr, args.loss),
    )
    wandb.config = {
        "learning_rate" : args.lr,
        "batch_size" : args.batch_size
    }
    wandb.watch(net)

    batch_size = args.batch_size
    # loss_func = nn.CrossEntropyLoss()
    
    if args.loss == 'MSE':
        loss_func = nn.MSELoss()
    elif args.loss == 'Cosine':
        loss_func = nn.CosineEmbeddingLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_dataset = RenderedInstrumentDataset(split='train')
    test_dataset = RenderedInstrumentDataset(split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, collate_fn=train_dataset.collate_fn)

    torch.cuda.empty_cache()

    # load f_enc
    model = f_enc(out_classes = 953).to(DEVICE)
    loaded_dict = torch.load('/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_00/class953_epoch6_iter22200_trLoss_0.193_trAcc_91.490', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    epoch = 0
    while True:
        train_loss = train(net, train_loader, optimizer, epoch, DEVICE, loss_func, args.loss, _f_enc)

        epoch += 1



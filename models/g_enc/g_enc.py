import glob
import time
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import wandb
import argparse
from h_enc_dataset import EmbeddingDataset
from g_enc.g_data import InstrumentReprDataset
from h_enc_encoded_f_emb import ConvNet as _h_enc

torch.seed(0)

# default model architecture for f_enc -> last linear layer output dimension is 953 (for 953 classes)
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
    
    def forward(self, x, get_repr=False):
        x = x.unsqueeze(dim=1).permute(0, 1, 3, 2)
        
        x = self.pool_drop(self.conv1(x.float()))
        x = self.pool_drop(self.conv2(x))
        x = self.pool_drop(self.conv3(x))

        x = self.conv4(x)
        x = self.linear1(x)

        if get_repr:
            return x

        x = self.linear2(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default=None)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--loss', type=str, default='MSE')

    args = parser.parse_args()
    
    return args

def train(net, train_loader, epoch, inst_repr_lib):
    net.train()
    train_loss = []
    tqdm_bar = tqdm(enumerate(train_loader))

    loading_time = []
    training_time = []

    start = time.time()

    for idx, (input, labels) in tqdm_bar:
        loading_time.append(time.time() - start)
        input, labels = input.to(DEVICE), labels.to(DEVICE)

        start_train = time.time()
        while labels.size() != 0:
            optimizer.zero_grad()

            # predicts the predominant instrument class
            out = net(input)

            # get the minimum loss among the tracks and the corresponding label
            loss, min_label = compute_min_loss(out, labels)

            # arithmetically subract the predominant class f_emb from the input
            input -= inst_repr_lib[min_label]

            # delete the label from the list
            labels = labels[labels!=min_label]

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            tqdm_bar.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))

            wandb.log({
                'train_loss': loss.item(),
                'epoch': epoch,
                'idx': idx
            })

        training_time.append(time.time() - start_train)
        start = time.time()

    wandb.log({
        "Data Loading Time" : np.mean(loading_time),
        "Training Time" : np.mean(training_time),
    })

def valid(net, valid_loader, loss_name, epoch):
    net.eval()

    with torch.no_grad():
        return

def compute_min_loss(out, labels):
    min_loss = float('inf')
    min_label = -1
    for label in labels:
        loss = nn.CrossEntropyLoss()(out, label)
        if loss < min_loss:
            min_loss = loss
            min_label = label
    return min_loss, min_label

def get_inst_repr_lib(path="/home/haessun/ai_prod/inst_repr_lib/"):
    inst_num_list = range(1, 954)
    inst_repr_lib = {}
    for inst_num in inst_num_list:
        inst_repr_path = path + str(inst_num) + ".pt"
        inst_repr_lib[inst_num] = torch.load(inst_repr_path)
    return inst_repr_lib

def compute_similarity(out_repr, repr_loader, loss_name):
    min_loss = float('inf')
    min_loss_idx = -1
    for repr, inst_num in repr_loader:
        loss = repr_loss_func(out_repr, repr, loss_name)
        if loss < min_loss:
            min_loss = loss
            min_loss_idx = inst_num
    
    return min_loss_idx, min_loss

def repr_loss_func(out_repr, repr, loss_name):
    if loss_name == 'MSE':
        loss = nn.MSELoss()(out_repr, repr)
    elif loss_name == 'Cosine':
        loss = nn.CosineSimilarity()(out_repr, repr, torch.ones(out_repr.size()[0]).to(DEVICE))
    else:
        raise ValueError('Invalid loss name')

    return loss

if __name__=='__main__':

    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    ## finetuning f_enc to make g_enc ##
    # to get a single predominant inst label among multi-inst input
    net = ConvNet().to(DEVICE)
    loaded_dict = torch.load('/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class953_epoch8_iter1200_trLoss_0.300_trAcc_89.031', map_location = DEVICE)
    net.load_state_dict(loaded_dict, strict=False)

    ## h_enc for validation input = h_enc(multi-inst_audio) ##
    h_enc = _h_enc().to(DEVICE)
    loaded_dict = torch.load('', map_location = DEVICE)
    h_enc.load_state_dict(loaded_dict, strict=False)
    h_enc.eval()
    h_enc = h_enc.forward

    wandb.init(
        project = args.project_name,
        name = "g_enc : batch{}, lr={}, loss={}".format(args.num_class,  args.batch_size, args.lr, args.loss),
    )
    wandb.config = {
        "learning_rate" : args.lr,
        "batch_size" : args.batch_size
    }
    wandb.watch(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    repr_dataset = InstrumentReprDataset()
    repr_dataloader = torch.utils.data.DataLoader(repr_dataset, batch_size=1)

    # for teacher forcing -> get item을 했을 때, Tracklist와 함께 inst_num list가 필요함
    # (input = summed f_tracks, label = inst_num list)
    train_dataset = EmbeddingDataset(split='train')
    valid_dataset = EmbeddingDataset(split='valid')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=4, shuffle=False)

    inst_repr_lib = get_inst_repr_lib()
    epoch = 0
    while True:
        train(net, train_loader, epoch, inst_repr_lib)
        valid(net, valid_loader, args.loss, epoch)

        epoch +=1
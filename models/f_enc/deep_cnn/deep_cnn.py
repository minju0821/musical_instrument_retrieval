import glob
import time
import random
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import wandb
import scipy
import librosa
import numpy as np
from sklearn.metrics import roc_curve

import torch
import torch.nn as nn

from render_data import RenderedInstrumentDataset

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
    
    def forward(self, x):
        x = x.unsqueeze(dim=1).permute(0, 1, 3, 2)
        
        x = self.pool_drop(self.conv1(x.float()))
        x = self.pool_drop(self.conv2(x))
        x = self.pool_drop(self.conv3(x))

        x = self.conv4(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return x

# model architecture for validation w/ removed last linear layer from the default one
# -> to get the embedded representation of 1024-dim
class ConvNet_eval(nn.Module):
    def __init__(self, out_classes):
        super(ConvNet_eval, self).__init__()

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
            # nn.Linear(in_features=1024, out_features=out_classes),
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

# customized dataset for validation to sample the number of num_eval samples of same inst. (inst_idx)
class InstrumentDataset_eval:
    def __init__(self, split = "test", data_path = "/disk2/aiproducer_inst/rendered_single_inst/",
                 num_samples_per_inst=1000):
        assert num_samples_per_inst == 1000
        self.num_samples_per_inst = num_samples_per_inst
        self.split = split
        self.data_path = Path(data_path) / split
        self.inst_dirs = glob.glob(str(self.data_path) + "/*/")
        self.inst_dirs.sort()

    def get_samples(self, idx, num_samples):
        samples = []
        sample_idx_list = random.sample(range(self.num_samples_per_inst), num_samples)
        inst_dir = self.inst_dirs[idx]
        for sample_idx in sample_idx_list: 
            fname = inst_dir + f"{sample_idx+1:04d}.wav"
            sr, audio = scipy.io.wavfile.read(fname)
            audio = np.array(audio, dtype=np.float32) / 32768.0
            # audio = torch.tensor(audio, dtype=torch.float32) / 32768.0

            # customized for the deep_cnn
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32)

            samples.append(log_spec)
        return samples

class EER:
    def __init__(self, encoder, device, num_enroll = 5, num_eval = 20):
        self.encoder = encoder
        self.device = device
        self.num_enroll = num_enroll
        self.num_eval = num_eval
        self.inst_data = InstrumentDataset_eval()

    def dist(self, a, b):
        dist = 1 - torch.sum(a*b) / (torch.norm(a) * torch.norm(b) + 1e-08)
        return dist

    def enrollment(self, inst_idx):
        emb = []
        for sample in self.inst_data.get_samples(inst_idx, self.num_enroll):
            emb.append(self.encoder(sample.to(self.device).unsqueeze(0)))
        return torch.stack(emb).mean(dim=0)

    def y_score(self):
        y_score = np.array([])
        print('Evaluating..')
        for inst_idx in tqdm(range(len(self.inst_data.inst_dirs))):
            target_emb = self.enrollment(inst_idx)
            for sample in self.inst_data.get_samples(inst_idx, self.num_eval):
                pos_emb = self.encoder(sample.to(self.device).unsqueeze(0))
                dist = self.dist(target_emb, pos_emb)
                y_score = np.append(y_score, np.array([dist.item()]))
            for i in range(self.num_eval):
                idx_diff = random.randrange(len(self.inst_data.inst_dirs))
                while idx_diff == inst_idx:
                    idx_diff = random.randrange(len(self.inst_data.inst_dirs))
                neg_emb = self.encoder(self.inst_data.get_samples(idx_diff, 1)[0].unsqueeze(0).to(self.device))
                dist = self.dist(target_emb, neg_emb)
                y_score = np.append(y_score, np.array([dist.item()]))
        return y_score * -1

    def compute_eer(self, fpr, tpr, thresholds):
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        return eer, thresholds[min_index]

    def evaluate(self):
        y_true = np.array([])
        for _ in range(len(self.inst_data.inst_dirs)):
            y_true = np.append(y_true, np.repeat([1,0], self.num_eval))
        y_score = self.y_score()
        print(np.shape(y_true), np.shape(y_score))
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        eer, threshold = self.compute_eer(fpr,tpr,thresholds)
        return eer, threshold


def train(net, train_loader, optimizer, epoch, DEVICE, loss_func, batch_size):
    net.train()
    
    correct = 0
    train_loss = []
    loading_time = []
    training_time = []

    start = time.time()

    pred_len = 0

    tqdm_bar = tqdm(enumerate(train_loader))
    for idx, (data, labels) in tqdm_bar:
        net.train()

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
        pred_len += len(prediction)
        correct_tmp = prediction.eq(labels.view_as(prediction)).sum().item()
        correct += correct_tmp

        optimizer.step()

        tqdm_bar.set_description("Epoch {} - train loss: {:.3f}  train acc: {:.3f}".format(epoch, loss.item(), (100. * correct_tmp / batch_size)))

        start = time.time()

        # on every 300 iteration and on last iteration
        if idx % 300 == 0:
            evaluate(net, DEVICE)

            train_loss_wandb = sum(train_loss) / len(train_loss)
            train_acc = 100. * correct / (300 * batch_size)

            model_name = '/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class{}_epoch{}_iter{}_trLoss_{:.3f}_trAcc_{:.3f}'.format(953, epoch, idx, train_loss_wandb, train_acc)
            torch.save(net.state_dict(), model_name)

            wandb.log({
                "Epoch" : epoch,
                "Iteration" : idx,
                "Train Acc." : train_acc,
                "Train Loss" : train_loss_wandb,
                "Data Loading Time" : np.mean(loading_time),
                "Training Time" : np.mean(training_time)
            })

            ## reset variables ##
            loading_time = []
            training_time = []
            train_loss = []
            correct = 0
            pred_len = 0

    evaluate(net, DEVICE)
            
    train_loss_wandb = sum(train_loss) / len(train_loss)
    train_acc = 100. * correct / pred_len
    model_name = '/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class{}_epoch{}_iter{}_trLoss_{:.3f}_trAcc_{:.3f}'.format(953, epoch, idx, train_loss_wandb, train_acc)
    torch.save(net.state_dict(), model_name)

    wandb.log({
        "Epoch" : epoch,
        "Iteration" : idx,
        "Train Acc." : train_acc,
        "Train Loss" : train_loss_wandb,
        "Data Loading Time" : np.mean(loading_time),
        "Training Time" : np.mean(training_time)
    })
    
def evaluate(net, DEVICE):
    net.eval()

    net_eval = ConvNet_eval(out_classes=953).to(DEVICE)
    net_eval.load_state_dict(dict(list(net.state_dict().items())[:-2]))
    net_eval.eval()
    net_eval = net_eval.forward

    with torch.no_grad():
        eer = EER(net_eval, DEVICE)
        eer_score, threshold = eer.evaluate()

    wandb.log({
        "Valid EER" : eer_score,
        "Valid EER_thres." : threshold,
    })


if __name__=='__main__':

    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    net = ConvNet(out_classes=args.num_class).to(DEVICE)

    wandb.init(
        project = args.project_name,
        name = "class={}, batch{}, lr={}".format(args.num_class, args.batch_size, args.lr),
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    torch.cuda.empty_cache()

    epoch = 0
    while True:
        train(net, train_loader, optimizer, epoch, DEVICE, loss_func, batch_size)

        epoch += 1

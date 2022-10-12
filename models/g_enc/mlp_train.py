import random
import argparse
from tqdm.auto import tqdm
import numpy as np
import wandb
import torch
import torch.nn as nn

from mlp_model_pit import MLP
from g_enc_cnn_data import EmbeddingDataset
from torch.utils.data import DataLoader
from PIT_loss import PITLossWrapper
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--metric", type=str, default="Cosine")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=2000)

    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--log_dir", type=str, default="/data4/aiproducer_inst/haessun_models/g_enc/plt_00/logs/")
    parser.add_argument("--checkpoint_dir", type=str, default="/data4/aiproducer_inst/haessun_models/g_enc/plt_00/checkpoints/")
    parser.add_argument("--data_dir", type=str, default="/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb/")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--loss", type=str, default="Cosine")

    args = parser.parse_args()

    return args

class MLP(nn.Module):
    def __init__(self, hparams):
        super(MLP, self).__init__()

        # Model definition
        self.layers = [nn.Linear(hparams.embedding_dim, hparams.hidden_dim), nn.ReLU()]
        for _ in range(hparams.num_hidden_layers - 1):
            self.layers.append(nn.Linear(hparams.hidden_dim, hparams.hidden_dim))
            self.layers.append(nn.ReLU())
        
        # maximum number of tracks in mix_audio == 9
        self.layers.append(nn.Linear(hparams.hidden_dim, hparams.embedding_dim * 9))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        print(x.size())
        return self.model(x)

def pit_loss(y_hat, y, pit_func, loss_name):
    if loss_name == 'Cosine':
        return pit_func(y_hat, y, target=torch.ones_like(y[0]))
    elif loss_name == 'MSE':
        return pit_func(y_hat, y)

def train(model, train_loader, optimizer, pit_func, loss_name, epoch, device):
    model.train()
    train_loss = 0
    for batch_idx, (mix_audio, emb_list, emb_len) in tqdm(enumerate(train_loader)):
        mix_audio, emb_list, emb_len = mix_audio.to(device), emb_list.to(device), emb_len.to(device)

        optimizer.zero_grad()
        output = model(mix_audio)

        # (batch_size, 1024 * 9) -> (batch_size, 9, 1024)
        output = torch.reshape(output, (output.size()[0], 9, -1))

        loss = pit_loss(output, emb_list[:emb_len], pit_func, loss_name)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    torch.save(model.state_dict(), f"{args.checkpoint_dir}/{epoch}_{train_loss}.pt")
    wandb.log({"train_loss": train_loss})
    return

def evaluate(model, valid_loader, device, loss_name):
    model.eval()
    loss_list = []

    with torch.no_grad():
        for batch_idx, (mix_audio, emb_list, emb_len) in tqdm(enumerate(valid_loader)):
            mix_audio, emb_list, emb_len = mix_audio.to(device), emb_list.to(device), emb_len.to(device)
            
            output = model(mix_audio)
            # (batch_size, 1024 * 9) -> (batch_size, 9, 1024)
            output = torch.reshape(output, (output.size()[0], 9, -1))

            loss = pit_loss(output, emb_list[:emb_len], pit_func, loss_name)

            loss_list.append(loss.item())

    wandb.log({
        "Valid Summed Loss" : sum(loss_list) / len(loss_list)
    })
    return

if __name__=='__main__':
    args = parse_args()

    DEVICE = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    model = MLP(args).to(DEVICE)

    wandb.init(
        project = 'g_enc',
        name = "g_enc_pit_00 : loss={}, batch{}, lr={}".format(args.loss, args.batch_size, args.lr),
    )
    wandb.config = {
        "learning_rate" : args.lr,
        "batch_size" : args.batch_size
    }
    wandb.watch(model)

    if args.loss == "Cosine":
        pit_func = PITLossWrapper(nn.CosineEmbeddingLoss(), pit_from='pw_mtx')
    elif args.loss == "MSE":
        pit_func = PITLossWrapper(nn.MSELoss(), pit_from='pw_mtx')
    else:
        raise ValueError("metric must be either cosine or MSE.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = EmbeddingDataset(args.data_dir, "train")
    valid_dataset = EmbeddingDataset(args.data_dir, "valid")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=valid_dataset.collate_fn)

    epoch = 0
    while True:
        train(model, train_loader, optimizer, pit_func, args.loss, epoch, DEVICE)
        evaluate(model, valid_loader, DEVICE, args.loss)

        epoch += 1

        
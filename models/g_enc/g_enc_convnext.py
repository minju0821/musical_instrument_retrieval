import random
import argparse
from tqdm.auto import tqdm
import wandb
import torch
import torch.nn as nn
from g_enc_cnn_data import EmbeddingDataset
from g_enc.PIT_loss import PITLossWrapper
from torch.utils.data import DataLoader
from g_enc.asteroid_mse import PairwiseMSE
import numpy as np
from ConvNext.convnext_git.models.convnext import ConvNeXt

random.seed(0)
torch.manual_seed(0)
DEVICE = torch.device('cuda:{}'.format(10)) if torch.cuda.is_available else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=953)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument("--loss", type=str, default="Cosine")
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="/data4/aiproducer_inst/haessun_models/g_enc/convnext/checkpoints")
    parser.add_argument("--data_dir", type=str, default="/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb/")

    args = parser.parse_args()
    
    return args

def train(model, train_loader, optimizer, pit_func, epoch, loss_name):
    model.train()
    train_loss = []
    for batch_idx, (mix_audio, emb_list, emb_len) in tqdm(enumerate(train_loader)):
        mix_audio, emb_list, emb_len = mix_audio.to(DEVICE), emb_list.to(DEVICE), emb_len.to(DEVICE)

        optimizer.zero_grad()
        # output = model(mix_audio.squeeze())
        output = model(mix_audio)
        # (batch_size, 1024 * 9) -> (batch_size, 9, 1024)
        output = torch.reshape(output, (output.size()[0], 9, -1))

        if loss_name == "Cosine":
            loss = pit_func(output, emb_list, track_num=emb_len, target=torch.ones(32).to(DEVICE))
        else:
            loss = pit_func(output, emb_list, track_num=emb_len)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 200 == 0:
            wandb.log({
                "train_loss": np.mean(train_loss),
                "epoch": epoch,
                "batch_idx": batch_idx
            })

    train_loss = np.mean(train_loss)
    torch.save(model.state_dict(), f"{args.checkpoint_dir}/{loss_name}_pw_pt/{epoch}_{train_loss:.3f}.pt")
    
    return

def evaluate(model, valid_loader, loss_name):
    model.eval()
    loss_list = []

    with torch.no_grad():
        for batch_idx, (mix_audio, emb_list, emb_len) in tqdm(enumerate(valid_loader)):
            mix_audio, emb_list, emb_len = mix_audio.to(DEVICE), emb_list.to(DEVICE), emb_len.to(DEVICE)
            
            # output = model(mix_audio.squeeze())
            output = model(mix_audio)
            # (batch_size, 1024 * 9) -> (batch_size, 9, 1024)
            output = torch.reshape(output, (output.size()[0], 9, -1))

            if loss_name == "Cosine":
                loss = pit_func(output, emb_list, track_num=emb_len, target=torch.ones(mix_audio.size()[0]).to(DEVICE))
            else:
                loss = pit_func(output, emb_list, track_num=emb_len)
            loss_list.append(loss.item())

    wandb.log({
        "Valid Loss" : sum(loss_list) / len(loss_list)
    })
    return

def cosine_loss(y_hat, y, batch_size=32):
    loss_func = nn.CosineEmbeddingLoss(reduce=False)
    target = torch.tensor(1.).to(DEVICE)

    # pw_loss = np.zeros((batch_size, 9, 9))
    # for b in range(batch_size):
    #     for h_idx in range(9):
    #         for y_idx in range(9):
    #             pw_loss[b, h_idx, y_idx] = loss_func(y_hat[b, h_idx], y[b, y_idx], target).item()

    pw_loss2 = np.zeros((batch_size, 9, 9))
    for b in range(batch_size):
        for y_idx in range(9):
            for h_idx in range(9):
                pw_loss2[b, y_idx, h_idx] = loss_func(y_hat[b, h_idx], y[b, y_idx], target).item()
    pw_loss = pw_loss2

    return torch.tensor(pw_loss, dtype=torch.float32).to(DEVICE)

if __name__=='__main__':
    args = parse_args()

    print("Using PyTorch version: {}, DEVICE: {}".format(torch.__version__, DEVICE))

    model = ConvNeXt(num_classes=953, in_chans=1).to(DEVICE)

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
        # pit_func = PITLossWrapper(cosine_loss, pit_from='pw_mtx')
        pit_func = PITLossWrapper(nn.CosineEmbeddingLoss(), pit_from='pw_pt')
    elif args.loss == "MSE":
        pit_func = PITLossWrapper(PairwiseMSE(), pit_from='pw_mtx')
    else:
        raise ValueError("metric must be either cosine or MSE.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = EmbeddingDataset(args.data_dir, "train", args.loss, args.batch_size)
    valid_dataset = EmbeddingDataset(args.data_dir, "valid", args.loss, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4)

    epoch = 0
    while True:
        train(model, train_loader, optimizer, pit_func, epoch, loss_name=args.loss)
        evaluate(model, valid_loader, args.loss)

        epoch += 1

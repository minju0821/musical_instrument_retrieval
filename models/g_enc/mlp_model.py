import torch
import pytorch_lightning as pl
import torch.nn as nn

class MLP(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Move hparams to self.
        if hparams.metric == "cosine":
            self.metric = nn.CosineSimilarity(dim=2)
        elif hparams.metric == "euclidean":
            def euclidean(x, y):
                return (x-y).pow(2).sum(dim=2).pow(0.5)
            self.metric = euclidean 
        else:
            raise ValueError("metric must be either cosine or euclidean.")
        self.lr = hparams.lr

        # Model definition
        self.layers = [nn.Linear(hparams.embedding_dim, hparams.hidden_dim), nn.ReLU()]
        for _ in range(hparams.num_hidden_layers - 1):
            self.layers.append(nn.Linear(hparams.hidden_dim, hparams.hidden_dim))
            self.layers.append(nn.ReLU())

            
        self.layers.append(nn.Linear(hparams.hidden_dim, hparams.embedding_dim))
        self.model = nn.Sequential(*self.layers)
    
    def pit_loss(self, y_hat, y):
        # y_hat: extracted embedding. (batch_size, 1024)
        # y: original embeddings. invalid elements are filled with infinity. (batch_size, max_tracks, 1024)

        # repeat the same y_hat for each track and calculate the loss.
        y_hat = y_hat.unsqueeze(1).repeat(1, y.shape[1], 1)
        distance = self.metric(y_hat, y)

        # mask out invalid elements
        distance_mask = torch.ones_like(distance)
        distance_mask[y[:,:,0]==1e8] = 1e8
        distance = distance * distance_mask

        # find the minimum loss.
        min_loss = torch.min(distance, dim=1).values
        mean_min_loss = torch.mean(min_loss)
        return mean_min_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        x, y = batch
        y_hat = self(x)
        loss = self.pit_loss(y_hat, y)
        with torch.autograd.detect_anomaly():
            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()

        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.pit_loss(y_hat, y)
        self.log("val_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
import torch
import pytorch_lightning as pl
import torch.nn as nn
from PIT_loss import PITLossWrapper

class MLP(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if hparams.loss == "Cosine":
            self.loss = PITLossWrapper(nn.CosineEmbeddingLoss(), pit_from='pw_mtx')
        elif hparams.loss == "MSE":
            self.loss = PITLossWrapper(nn.MSELoss(), pit_from='pw_mtx')
        else:
            raise ValueError("metric must be either cosine or MSE.")

        self.lr = hparams.lr
        self.metric = hparams.loss

        # Model definition
        self.layers = [nn.Linear(hparams.embedding_dim, hparams.hidden_dim), nn.ReLU()]
        for _ in range(hparams.num_hidden_layers - 1):
            self.layers.append(nn.Linear(hparams.hidden_dim, hparams.hidden_dim))
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(hparams.hidden_dim, hparams.embedding_dim))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

    def pit_loss(self, y_hat, y):
        if self.metric == 'Cosine':
            return self.loss(y_hat, y, target=torch.ones_like(y[0]))
        elif self.metric == 'MSE':
            return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # y = (batch, 1024) -> (y_len, 1024)
        x, y, y_len = batch
        y_hat = self(x)
        loss = self.pit_loss(y_hat, y[:y_len])

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
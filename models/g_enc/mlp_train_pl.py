import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from mlp_model_pit import MLP
from g_enc_cnn_data import EmbeddingDataset
from torch.utils.data import DataLoader

import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--metric", type=str, default="Cosine")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=2000)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--log_dir", type=str, default="/data4/aiproducer_inst/haessun_models/g_enc/plt_00/logs/")
    parser.add_argument("--checkpoint_dir", type=str, default="/data4/aiproducer_inst/haessun_models/g_enc/plt_00/checkpoints/")
    parser.add_argument("--data_dir", type=str, default="/data4/aiproducer_inst/f_embeddings/f_haessun/rendered_multi_inst_3_f_emb/")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--loss", type=str, default="Cosine")
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # Logger
    wandb_logger = WandbLogger(project="g_enc", name="mlp_pit")

    # Model
    model = MLP(args)
    wandb_logger.watch(model)

    # Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="mlp-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10
    )

    # Dataset
    train_dataset = EmbeddingDataset(args.data_dir, "train")
    valid_dataset = EmbeddingDataset(args.data_dir, "valid")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=valid_dataset.collate_fn)

    # Train
    trainer.fit(model, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()
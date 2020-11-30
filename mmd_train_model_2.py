import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from mmd_vae_model_2 import CancerSamplesDataset, Bimodal_MMD_VAE, compute_mmd

if __name__ == "__main__":
    # Training hyperparameters
    n_epochs = 200
    batch_size = 128
    early_stopping_patience = 5
    num_workers = 4

    train_ds = CancerSamplesDataset(join("data", "sample_subtype_encodings.csv"),
                                    join("data", "sorted_mutations.json"),
                                    join("data", "mutations_mapping_split.json"),
                                    train=True)
    val_ds = CancerSamplesDataset(join("data", "sample_subtype_encodings.csv"),
                                  join("data", "sorted_mutations.json"),
                                  join("data", "mutations_mapping_split.json"),
                                  train=False)
    model = Bimodal_MMD_VAE()

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            drop_last=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3,
                                momentum=0.9,
                                weight_decay=2e-5)
    reconstruction_loss = nn.BCELoss()

    def loss_function(X_del, X_nd, z, y_del, y_nd):
        r_del = reconstruction_loss(y_del, X_del)
        r_nd = reconstruction_loss(y_nd, X_nd)
        true_samples = torch.randn(batch_size, 50, requires_grad=False)
        mmd_loss = compute_mmd(true_samples, z)
        loss = r_del + r_nd + mmd_loss
        return loss, r_del, r_nd, mmd_loss

    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("models"):
        os.mkdir("models")

    train_iter = 0
    val_iter = 0
    best_val_loss = np.inf
    patience_counter = 0
    tb_writer = SummaryWriter(os.path.join("logs", "mmd_vae"))
    for epoch in range(n_epochs):
        print("="*30)
        print("Epoch ", epoch)

        # Training epoch
        avg_train_loss = 0
        model.train()

        pbar = tqdm(total=len(train_loader))
        for idx, batch in enumerate(train_loader):
            train_iter += 1

            X_del, X_nd, _ = batch
            y_del, y_nd, z = model(X_del, X_nd)

            train_loss, r_del, r_nd, mmd_loss = loss_function(
                X_del, X_nd, z, y_del, y_nd)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss = train_loss.item()
            avg_train_loss += train_loss

            tb_writer.add_scalar('Loss/train', train_loss, train_iter)
            tb_writer.add_scalar('Loss/train_mmd_loss', mmd_loss, train_iter)
            tb_writer.add_scalar('Loss/train_r_del', r_del, train_iter)
            tb_writer.add_scalar('Loss/train_r_nd', r_nd, train_iter)

            avg_loss = avg_train_loss/(idx+1)
            pbar.set_description(
                f"Train loss {train_loss}, running avg loss {avg_loss}")
            pbar.update(1)
        pbar.close()

        avg_train_loss /= len(train_loader)
        tb_writer.add_scalar('AvgLoss/train', avg_train_loss, epoch)

        # Validation epoch

        avg_val_loss = 0

        model.eval()
        pbar = tqdm(total=len(val_loader))
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                val_iter += 1

                X_del, X_nd, _ = batch
                y_del, y_nd, z = model(X_del, X_nd)

                val_loss, r_del, r_nd, mmd_loss = loss_function(
                    X_del, X_nd, z, y_del, y_nd)
                val_loss = val_loss.item()
                avg_val_loss += val_loss

                tb_writer.add_scalar('Loss/val', val_loss, val_iter)
                tb_writer.add_scalar('Loss/val_mmd_loss', mmd_loss, val_iter)
                tb_writer.add_scalar('Loss/val_r_del', r_del, val_iter)
                tb_writer.add_scalar('Loss/val_r_nd', r_nd, val_iter)
                avg_loss = avg_val_loss/(idx+1)
                pbar.set_description(
                    f"Val loss {val_loss}, running avg loss {avg_loss}")
                pbar.update(1)
            pbar.close()

        avg_val_loss /= len(val_loader)

        tb_writer.add_scalar('AvgLoss/val', avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            print("New minimum validation loss, saving model.")
            patience_counter = 0
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            savepath = os.path.join("models", 'mmd_vae.ckpt')
            torch.save(state, savepath)
        else:
            patience_counter += 1
            if patience_counter > early_stopping_patience:
                print(
                    f"**********\nNo improvements for the last {str(early_stopping_patience)} epochs, stopping training")
                break

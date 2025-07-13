import torch
import torch.nn as nn
from PWSA import MultiModalLongformerQuality

import torch
import torch.nn as nn


class BPWaveformRegressor(nn.Module):
    """
    UPDATED: Model for waveform regression with an upsampling head.
    """

    def __init__(self, pre_trained_encoder, encoder_output_dim, encoder_seq_len, output_seq_len):
        super().__init__()
        self.encoder = pre_trained_encoder
        self.regressor_head = nn.Linear(
            in_features=encoder_seq_len * encoder_output_dim,
            out_features=output_seq_len
        )

    def forward(self, x):
        features = self.encoder(x)
        flat_features = torch.flatten(features, start_dim=1)
        predictions = self.regressor_head(flat_features)

        return predictions


import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BPDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        phy_files = sorted(glob.glob(os.path.join(data_path, "*_phy.npy")))

        all_phy = []
        all_bp = []

        print("Loading data...")
        for phy_file_path in tqdm(phy_files, desc="Loading files"):
            bp_file_path = phy_file_path.replace("_phy.npy", "_bp.npy")

            if os.path.exists(bp_file_path):
                phy_data = np.load(phy_file_path)
                bp_data = np.load(bp_file_path)
                all_phy.append(phy_data)
                all_bp.append(bp_data)
        self.phy_records = np.concatenate(all_phy, axis=0)
        self.bp_records = np.concatenate(all_bp, axis=0)

        print(f"Data loaded successfully! Total samples: {len(self.phy_records)}")

    def __len__(self):
        return len(self.phy_records)

    def __getitem__(self, idx):
        phy_sample = torch.tensor(self.phy_records[idx], dtype=torch.float32)
        bp_sample = torch.tensor(self.bp_records[idx], dtype=torch.float32)
        return phy_sample, bp_sample

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

def plot_predictions(model, dataloader, device, num_plots=5):
    model.eval()
    plt.figure(figsize=(15, num_plots * 3))

    with torch.no_grad():
        phy_signals, true_bps = next(iter(dataloader))
        phy_signals = phy_signals.to(device)
        predicted_bps = model(phy_signals).cpu().numpy()
        true_bps = true_bps.cpu().numpy()

    for i in range(min(num_plots, len(true_bps))):
        plt.subplot(num_plots, 1, i + 1)
        plt.plot(true_bps[i], label="Ground Truth BP", color='blue', alpha=0.8)
        plt.plot(predicted_bps[i], label="Predicted BP", color='red', linestyle='--')
        plt.title(f"Sample #{i + 1}")
        plt.ylabel("Blood Pressure")
        plt.legend()
        if i == num_plots - 1:
            plt.xlabel("Time Steps")

    plt.tight_layout()
    plt.savefig("bp_prediction_vs_truth.png")
    print("Prediction plot saved as bp_prediction_vs_truth.png")
    plt.show()


if __name__ == '__main__':
    CONFIG = {
        "data_path": "./save",
        "model_path": "./model_saved/pwsa_teacher.pth",
        "encoder_output_dim": 512,
        "backbone": "pwsa",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = MultiModalLongformerQuality(2, 512, 4, 2, 256, 8)
    checkpoint = torch.load(f"model_saved/pwsa_teacher.pth")
    backbone.load_state_dict(checkpoint["model_state_dict"])
    encoder = backbone.encoder

    for param in encoder.parameters():
        param.requires_grad = True

    model = BPWaveformRegressor(
        pre_trained_encoder=encoder,
        encoder_output_dim=CONFIG["encoder_output_dim"],
        encoder_seq_len=18,
        output_seq_len=9000
    ).to(device)

    full_dataset = BPDataset(data_path=CONFIG["data_path"])
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    print("Starting training...")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_train_loss = 0

        for phy, bp in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}"):
            phy, bp = phy.to(device), bp.to(device)
            optimizer.zero_grad()
            predicted_bp = model(phy)
            loss = criterion(predicted_bp, bp)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.6f}")
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for phy, bp in val_loader:
                phy, bp = phy.to(device), bp.to(device)
                predicted_bp = model(phy)
                loss = criterion(predicted_bp, bp)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.6f}")

    print("Training finished!")
    torch.save(model.state_dict(), 'bp_regressor_final.pth')

    print("Plotting predictions from validation set...")
    plot_predictions(model, val_loader, device, num_plots=5)
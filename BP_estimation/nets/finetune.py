import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# (在这里粘贴上面定义的 BPWaveformRegressor 和 BPDataset 类)
# ... (BPWaveformRegressor class code)
# ... (BPDataset class code)

def plot_predictions(model, dataloader, device, num_plots=5):
    """
    使用模型进行预测，并绘制预测结果与真实值的对比图。
    """
    model.eval()  # 设置为评估模式
    plt.figure(figsize=(15, num_plots * 3))

    with torch.no_grad():  # 推理时不需要计算梯度
        # 从dataloader中取一个batch的数据
        phy_signals, true_bps = next(iter(dataloader))
        phy_signals = phy_signals.to(device)

        # 获得模型预测
        predicted_bps = model(phy_signals).cpu().numpy()

        # 将真实BP也移到CPU
        true_bps = true_bps.cpu().numpy()

    # 绘制指定数量的对比图
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
    plt.savefig("bp_prediction_vs_truth.png")  # 保存图像
    print("Prediction plot saved as bp_prediction_vs_truth.png")
    plt.show()


if __name__ == '__main__':
    # --- 1. 配置参数 ---
    CONFIG = {
        "data_path": "./dataset/save",  # 包含 .npy 文件的 "save" 文件夹路径
        "model_path": "./model_saved/pwsa_teacher.pth",  # 预训练模型路径
        "encoder_output_dim": 512,  # 你的编码器输出维度，请根据你的模型确认
        "backbone": "pwsa",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # --- 2. 加载预训练编码器 ---
    # 这里我们模拟加载你的模型，请替换成你实际的加载代码
    # from your_model_file import MultiModalLongformerQuality # 假设你的模型定义在这个文件

    # 模拟一个编码器，你需要替换成你的真实模型
    # backbone = MultiModalLongformerQuality(2, 512, 4, 2, 256, 8)
    # checkpoint = torch.load(CONFIG["model_path"])
    # backbone.load_state_dict(checkpoint["model_state_dict"])
    # pre_trained_encoder = backbone.encoder

    # !!! 临时替代方案：在你集成真实模型前，可使用以下模拟编码器进行测试 !!!
    class MockEncoder(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.conv = nn.Conv1d(2, dim, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x.permute(0, 2, 1)  # -> (batch, seq_len, dim)


    pre_trained_encoder = MockEncoder(CONFIG["encoder_output_dim"])
    # !!! 替换结束 !!!

    for param in pre_trained_encoder.parameters():
        param.requires_grad = True  # 确保编码器参数可以被微调

    # --- 3. 创建回归模型 ---
    model = BPWaveformRegressor(
        pre_trained_encoder=pre_trained_encoder,
        encoder_output_dim=CONFIG["encoder_output_dim"]
    ).to(device)

    # --- 4. 准备数据 ---
    full_dataset = BPDataset(data_path=CONFIG["data_path"])

    # 分割数据集为训练集和验证集 (e.g., 90% train, 10% validation)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # --- 5. 定义损失函数和优化器 ---
    criterion = nn.MSELoss()  # 均方误差损失，非常适合回归任务
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # --- 6. 训练循环 ---
    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        model.train()  # 设置为训练模式
        total_train_loss = 0

        for phy, bp in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}"):
            phy, bp = phy.to(device), bp.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            predicted_bp = model(phy)

            # 计算损失
            loss = criterion(predicted_bp, bp)

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.6f}")

        # (可选) 在每个epoch后进行验证
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
    torch.save(model.state_dict(), 'bp_regressor_final.pth')  # 保存最终模型

    # --- 7. 推理和可视化 ---
    print("Plotting predictions from validation set...")
    plot_predictions(model, val_loader, device, num_plots=5)
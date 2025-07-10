import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# class Net(nn.Module):
#     def __init__(self, input_dim, activation, num_class):
#         super(Net, self).__init__()
#
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             activation,
#             nn.Dropout(0.5),
#
#             nn.Linear(1024, 512),
#             activation,
#             nn.Dropout(0.5),
#
#             nn.Linear(512, 64),
#             activation,
#             nn.Dropout(0.25),
#
#             nn.Linear(64, num_class)
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# X_train, X_test, y_train, y_test = train_test_split(ppg, bp, test_size=0.30, random_state=42)
#
# X_train_tensor = torch.from_numpy(X_train[:1000000]).float()
# y_train_tensor = torch.from_numpy(y_train[:1000000]).float()
#
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
#
# input_dim = X_train.shape[1]
# activation_fn = nn.ReLU()
# classes = 1
#
# model = Net(input_dim=input_dim, activation=activation_fn, num_class=classes)
#
# loss_fn = nn.HuberLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# epochs = 5
# model.train()
#
# for epoch in range(epochs):
#     epoch_loss = 0.0
#     for i, (inputs, targets) in enumerate(train_loader):
#         outputs = model(inputs)
#         loss = loss_fn(outputs, targets)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

import numpy as np
import os
path = "D:\database\BP\save\part_1_dbp.npy"
data = np.load(path)
print(data.shape)

dir = "D:\database\BP\save"
print(os.listdir(dir))
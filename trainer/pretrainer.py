import torch
from model import ResNet18
from dataset import SiamDataset
from torch.utils.data import random_split, DataLoader
import losses
import numpy as np
from tqdm import tqdm
import utils



device = 'cuda'

# Setting
batch_size = 256
backbone = "ResNet18"
pair_data_path = "data/mimic/pair_segments"
lr = 1e-4
min_lr = 1e-6
epochs = 1000
ratio_train_val = 0.9
model_save_path = "model_saved"
warmup_epochs = 10
weight_decay = 0.04
weight_decay_end = 0.4
momentum_teacher = 0.996


# ======================== set dataset and dataloader =====================
dataset = SiamDataset(pair_data_path)

print("Total numbers of pre-training pairs:", len(dataset))
print("Using the model of:", backbone)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ================== building teacher and student models =================
# Initiate Student and Teacher encoder
student = ResNet18()
teacher = ResNet18()

student, teacher = student.cuda(), teacher.cuda()

total_params = sum(p.numel() for p in teacher.parameters())
trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# teacher and student start with the same weights
teacher.load_state_dict(student.state_dict())

# Frozen teacher, only use backward train student
for p in teacher.parameters():
    p.requires_grad = False


# =================== build loss, optimizer and schedulers =================
# self-distillation loss function
self_distill_loss = losses.SimpleDINOLoss(out_dim=64).cuda()

# build adam optimizer
params_groups = utils.get_params_groups(student)
optimizer = torch.optim.Adam(params_groups ,lr=lr)

# init schedulers
lr_schedule = utils.cosine_scheduler(
    lr * (batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
    min_lr,
    epochs, len(dataloader),
    warmup_epochs=warmup_epochs,
)

wd_schedule = utils.cosine_scheduler(
    weight_decay,
    weight_decay_end,
    epochs, len(dataloader),
)

# momentum parameter is increased to 1. during training with a cosine schedule
momentum_schedule = utils.cosine_scheduler(momentum_teacher, 1,
                                           epochs, len(dataloader))
print(f"Loss, optimizer and schedulers ready.")

# ====================== Start Training =========================
losses_list = []

best_loss = float('inf')
patience = 20
epochs_no_improve = 0

for epoch in range(0, epochs):
    losses_per_epoch = []
    pbar = tqdm(enumerate(dataloader))

    for batch_idx, (x1, x2) in tqdm(enumerate(dataloader)):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[batch_idx]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[batch_idx]

        x1, x2 = x1.to("cuda", dtype=torch.float32), x2.to("cuda", dtype=torch.float32)

        teacher_output = teacher(x1) # good signal as input of teacher
        student_output = student(x2) # bad signal as input of student

        loss = self_distill_loss(student_output, teacher_output)

        # student update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[batch_idx]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        losses_per_epoch.append(loss.cpu().data.numpy())

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dataloader),
                100. * batch_idx / len(dataloader),
                loss.item()))

    print(f"Training loss {np.mean(losses_per_epoch)}")
    losses_list.append(np.mean(losses_per_epoch))

    # 保存模型
    if losses_list[-1] < best_loss:
        print("Model is going to save")
        print(f"last loss: {losses_list[-1]} | best loss: {best_loss}")
        best_loss = losses_list[-1]
        epochs_no_improve = 0

        # save teacher model
        torch.save(
            {'model_state_dict': teacher.state_dict()},
            f'{model_save_path}/{backbone}_teacher.pth'
        )

        # torch.save(
        #     {'model_state_dict': student.state_dict()},
        #     f'{model_save_path}/{backbone}_student.pth'
        # )
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs.")

    # Early Stopping
    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break

utils.plot_losses(losses_list, save_path='train_val_loss_curve.png')
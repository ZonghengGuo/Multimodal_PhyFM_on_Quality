import argparse
from dataset import dataset
from torch.utils.data import random_split, DataLoader
import numpy as np
from tqdm import tqdm
from trainer import utils
from trainer.model import FourierSpectrumProcessor
from trainer.losses import EMALoss, calculate_rec_loss
from trainer.model import MultiModalTransformerQuality
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Multimodal_PhyFM_on_Quality Pretraining Stage')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of samples per batch.')
    parser.add_argument('--backbone', type=str, default="ResNet18",
                        help='The architecture of the feature extractor')
    parser.add_argument('--pair_data_path', type=str, default="data/mimic/pair_segments",
                        help='Path to the directory containing paired data segments.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for learning rate schedulers like cosine annealing.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Total number of training epochs.')
    parser.add_argument('--ratio_train_val', type=float, default=0.9,
                        help='Split ratio for training and validation data.')
    parser.add_argument('--model_save_path', type=str, default="model_saved",
                        help='Path to the directory where trained models will be saved.')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs for the learning rate scheduler.')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='Initial weight decay value for the optimizer.')
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help='Final weight decay value, often used with schedulers.')
    parser.add_argument('--momentum_teacher', type=float, default=0.996,
                        help='Momentum for updating the teacher model in self-supervised learning frameworks (e.g., MoCo, DINO).')
    parser.add_argument('--out_dim', type=int, default=500,)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    pair_paths = [
        r"E:\PhyData\mimic-iv\physionet.org\files\mimic4wdb\0.1.0\pair",
        r"E:\PhyData\VitalDB\physionet.org\files\vitaldb\1.0.0\pair"
    ]

    # ======================== set dataset and dataloader =====================
    dataset = dataset.SiamDataset(pair_paths)

    print("Total numbers of pre-training pairs:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ================== building teacher and student models =================
    # Initiate Student and Teacher encoder
    student = MultiModalTransformerQuality(2, args.out_dim, 4, 2, 256)
    teacher = MultiModalTransformerQuality(2, args.out_dim, 4, 2, 256)

    student, teacher = student.cuda(), teacher.cuda()

    spectrum = FourierSpectrumProcessor(target_sequence_length=args.out_dim)

    total_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    params_in_M = total_params / 1_000_000
    print(f"Total parameters in the model: {params_in_M:.2f} M")  # 保留两位小数

    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())

    # Frozen teacher, only use backward train student
    for p in teacher.parameters():
        p.requires_grad = False

    # =================== build loss, optimizer and schedulers =================
    # self-distillation loss function
    self_distill_loss = EMALoss(out_dim=args.out_dim).cuda()

    # build adam optimizer
    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.Adam(params_groups, lr=args.lr)

    # init schedulers
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(dataloader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(dataloader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(dataloader))
    print(f"Loss, optimizer and schedulers ready.")

    # ====================== Start Training =========================
    losses_list = []

    best_loss = float('inf')
    patience = 20
    epochs_no_improve = 0

    for epoch in range(0, args.epochs):
        losses_per_epoch = []

        pbar = tqdm(enumerate(dataloader))

        for batch_idx, (x1, x2) in tqdm(enumerate(dataloader)):
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[batch_idx]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[batch_idx]

            x1, x2 = x1.to("cuda", dtype=torch.float32), x2.to("cuda", dtype=torch.float32)

            amp_x1, pha_x1 = spectrum(x1)
            amp_x2, pha_x2 = spectrum(x2)

            teacher_feature, teacher_amp, teacher_pha = teacher(x1)  # good signal as input of teacher
            student_feature, student_amp, student_pha = student(x2)  # bad signal as input of student

            loss_amp = calculate_rec_loss(student_amp, amp_x1)
            loss_pha = calculate_rec_loss(student_pha, pha_x1)

            EMA_loss = self_distill_loss(student_feature, teacher_feature)

            loss = loss_amp + loss_pha + EMA_loss

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
                'Train Epoch: {} [{}/{} ({:.0f}%)] Total Loss: {:.6f} Amp Loss: {:.6f} Pha Loss: {:.6f} EMA loss: {:.6f}'.format(
                    epoch, batch_idx + 1, len(dataloader),
                           100. * batch_idx / len(dataloader),
                    loss.item(), loss_amp.item(), loss_pha.item(), EMA_loss.item()))

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
                f'{args.model_save_path}/{args.backbone}_teacher.pth'
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








import argparse
from dataset import dataset
import numpy as np
from tqdm import tqdm
from pretrainer import utils
from models.Transformer import FourierSpectrumProcessor
from pretrainer.losses import EMALoss, calculate_rec_loss
from models.Transformer import MultiModalTransformerQuality
from models.ResNet import MultiModalResNet101Quality
from models.ResNet import MultiModalResNet18Quality
from models.Mamba import MultiModalMambaQuality
from models.PWSA import MultiModalLongformerQuality
import torch
import torch.nn as nn
import os


def get_args():
    parser = argparse.ArgumentParser(description='Multimodal_PhyFM_on_Quality Pretraining Stage with DataParallel')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Number of samples per batch.')
    parser.add_argument('--backbone', type=str, required=True,
                        help='The architecture of the feature extractor')
    parser.add_argument('--window_size', type=int, default=8,
                        help='The window size of physiological windowed sparse attention.')
    parser.add_argument('--pair_data_path', type=str, default="data/mimic/pair_segments",
                        help='Path to the directory containing paired data segments.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for learning rate schedulers like cosine annealing.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Total number of training epochs.')
    parser.add_argument('--ratio_train_val', type=float, default=0.9,
                        help='Split ratio for training and validation data.')
    parser.add_argument('--model_save_path', type=str, default="model_saved",
                        help='Path to the directory where trained models will be saved.')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs for the learning rate scheduler.')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='Initial weight decay value for the optimizer.')
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help='Final weight decay value, often used with schedulers.')
    parser.add_argument('--momentum_teacher', type=float, default=0.7,
                        help='Momentum for updating the teacher model in self-supervised learning frameworks.')
    parser.add_argument('--out_dim', type=int, default=512, help='Output feature dimension.')
    # parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for data loading.")
    parser.add_argument("--load_weights", type=bool, default=True, help="Load newest weights.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    pair_paths = [
        "/root/cross/mimic/pair",
        "/root/cross/vitaldb/pair"
    ]

    os.makedirs(args.model_save_path, exist_ok=True)

    # ======================== set dataset and dataloader =====================
    dataset = dataset.SiamDataset(pair_paths)

    print("Total numbers of pre-training pairs:", len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================== building teacher and student models =================
    if args.backbone == "pwsa":
        student = MultiModalLongformerQuality(2, 512, 4, 2, 256, args.window_size)
        teacher = MultiModalLongformerQuality(2, 512, 4, 2, 256, args.window_size)
    elif args.backbone == "pwsa_large":
        student = MultiModalLongformerQuality(2, 512, 4, 21, 512, args.window_size)
        teacher = MultiModalLongformerQuality(2, 512, 4, 21, 512, args.window_size)
    elif args.backbone == "pwsa_huge":
        student = MultiModalLongformerQuality(2, 512, 8, 50, 2048, args.window_size)
        teacher = MultiModalLongformerQuality(2, 512, 8, 50, 2048, args.window_size)
    elif args.backbone == 'transformer':
        student = MultiModalTransformerQuality(2, 512, 4, 2, 256)
        teacher = MultiModalTransformerQuality(2, 512, 4, 2, 256)
    elif args.backbone == 'resnet101':
        student = MultiModalResNet101Quality(2, 200, 18)
        teacher = MultiModalResNet101Quality(2, 200, 18)
    elif args.backbone == 'resnet18':
        student = MultiModalResNet18Quality(2, 200, 18)
        teacher = MultiModalResNet18Quality(2, 200, 18)
    elif args.backbone == 'mamba':
        student = MultiModalMambaQuality(2, 400, 2, 256)
        teacher = MultiModalMambaQuality(2, 400, 2, 256)
    else:
        raise ValueError(
            f"Unsupported backbone: '{args.backbone}'. Please choose from ['pwas', 'resnet', 'transformer', 'mamba'].")

    student = student.to(device)
    teacher = teacher.to(device)
    spectrum = FourierSpectrumProcessor(target_sequence_length=args.out_dim).to(device)

    if args.load_weights:
        teacher_ckpt = utils.find_latest_checkpoint(args.model_save_path, args.backbone, "teacher")
        if teacher_ckpt:
            checkpoint = torch.load(teacher_ckpt)
            teacher.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded teacher weights from {teacher_ckpt}")
        else:
            print("No teacher checkpoint found.")

        student_ckpt = utils.find_latest_checkpoint(args.model_save_path, args.backbone, "student")
        if student_ckpt:
            checkpoint = torch.load(student_ckpt)
            student.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded student weights from {student_ckpt}")
        else:
            print("No student checkpoint found.")


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        student = nn.DataParallel(student)
        teacher = nn.DataParallel(teacher)
        spectrum = nn.DataParallel(spectrum)

    total_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    params_in_M = total_params / 1_000_000
    print(f"Total parameters in the model: {params_in_M:.2f} M")

    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())

    # Frozen teacher, only use backward train student
    for p in teacher.parameters():
        p.requires_grad = False

    # =================== build loss, optimizer and schedulers =================
    self_distill_loss = EMALoss(out_dim=args.out_dim).to(device)

    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.Adam(params_groups, lr=args.lr)

    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size / 256.,
        args.min_lr,
        args.epochs, len(dataloader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(dataloader),
    )

    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(dataloader))
    print(f"Loss, optimizer and schedulers ready.")

    # ====================== Start Training =========================
    losses_list = []
    best_loss = float('inf')

    for epoch in range(0, args.epochs):
        losses_per_epoch = []

        # 移除 sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (x1, x2) in pbar:
            global_step = epoch * len(dataloader) + batch_idx
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[global_step]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[global_step]

            x1, x2 = x1.to(device, dtype=torch.float32), x2.to(device, dtype=torch.float32)

            if args.backbone == 'resnet101' or args.backbone == 'resnet18':
                teacher_feature = teacher(x1)
                student_feature = student(x2)
                EMA_loss = self_distill_loss(student_feature, teacher_feature)
                loss = EMA_loss
            else:
                amp_x1, pha_x1 = spectrum(x1)
                amp_x2, pha_x2 = spectrum(x2)
                teacher_feature, teacher_amp, teacher_pha = teacher(x1)
                student_feature, student_amp, student_pha = student(x2)
                loss_amp = calculate_rec_loss(student_amp, amp_x1)
                loss_pha = calculate_rec_loss(student_pha, pha_x1)
                EMA_loss = self_distill_loss(student_feature, teacher_feature)
                loss = 0.1 * loss_amp + 0.1 * loss_pha + EMA_loss


            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                m = momentum_schedule[global_step]
                student_params = student.module.parameters() if isinstance(student,
                                                                           nn.DataParallel) else student.parameters()
                teacher_params = teacher.module.parameters() if isinstance(teacher,
                                                                           nn.DataParallel) else teacher.parameters()
                for param_q, param_k in zip(student_params, teacher_params):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            losses_per_epoch.append(loss.item())

            if args.backbone == 'resnet101' or args.backbone == 'resnet18':
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Total Loss: {:.6f} EMA loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(dataloader),
                               100. * batch_idx / len(dataloader),
                        loss.item(), EMA_loss.item()))
            else:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Total Loss: {:.6f} Amp Loss: {:.6f} Pha Loss: {:.6f} EMA loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(dataloader),
                               100. * batch_idx / len(dataloader),
                        loss.item(), loss_amp.item(), loss_pha.item(), EMA_loss.item()))


        avg_epoch_loss = np.mean(losses_per_epoch)
        losses_list.append(avg_epoch_loss)

        print(f"Training loss {avg_epoch_loss}")

        print("Model is going to save")

        model_to_save = student.module if isinstance(student, nn.DataParallel) else student
        teacher_to_save = teacher.module if isinstance(teacher, nn.DataParallel) else teacher

        torch.save(
            {'model_state_dict': teacher_to_save.state_dict()},
            f'{args.model_save_path}/{args.backbone}_teacher_{epoch}.pth'
        )

        torch.save(
            {'model_state_dict': model_to_save.state_dict()},
            f'{args.model_save_path}/{args.backbone}_student_{epoch}.pth'
        )

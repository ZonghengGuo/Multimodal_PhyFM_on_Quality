import torch
import torch.nn as nn
import torch.nn.functional as F

class EMALoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)

        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach()

        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()

        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def calculate_rec_loss(rec, target):
    """Calculates the reconstruction loss."""
    rec_loss = F.smooth_l1_loss(rec, target)
    return rec_loss
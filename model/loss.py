

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss1(nn.Module):
    def __init__(self):
        super(ReconstructionLoss1, self).__init__()

        self.mse = nn.MSELoss(reduction='none')

    def forward(self, recon, target):
        """
        recon: [batch_size, num_features]
        target: [batch_size, num_features]
        返回每个样本的平均 MSE 损失: [batch_size]
        """
        loss = self.mse(recon, target)
        loss = loss.mean(dim=1)
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, predictions, labels):
        return self.bce(predictions, labels)

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()


        self.mse = nn.MSELoss(reduction='none')

    def forward(self, recon, target):


        return self.mse(recon, target)

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds, target):
        return self.bce(preds, target)


import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="none")


    def forward(self, anchor_features, positive_features, all_negatives):
        """
        计算 InfoNCE 损失。
        Args:
            anchor_features (torch.Tensor): 锚点嵌入, 形状 [B, D].
            positive_features (torch.Tensor): 正样本嵌入, 形状 [B, D].  <-- 这里是关键修改
            all_negatives (torch.Tensor): 包含批内和外部负样本的总池子, 形状 [N_total_neg, D].
        """
        device = anchor_features.device
        B, D = anchor_features.shape


        l_pos = torch.bmm(
            anchor_features.view(B, 1, D),
            positive_features.view(B, D, 1)
        ).squeeze(-1)


        sim_matrix_neg = torch.matmul(anchor_features, all_negatives.T)


        if all_negatives.shape[0] >= B:
             mask = torch.eye(B, dtype=torch.bool, device=device)
             sim_matrix_neg[:, :B].masked_fill_(mask, -float('inf'))


        logits = torch.cat([l_pos, sim_matrix_neg], dim=1)
        logits /= self.temperature


        labels = torch.zeros(B, dtype=torch.long, device=device)


        loss_per_sample = self.criterion(logits, labels)


        return loss_per_sample
class ReconWeightedInfoNCELoss(nn.Module):
    """
    一个将重构误差作为权重的InfoNCE损失函数。
    它会更关注那些重构误差小（即更像正常样本）的困难负样本。
    """
    def __init__(self, temperature=0.1, margin=0.5):
        """
        Args:
            temperature (float): InfoNCE中的温度系数.
            margin (float): M, 控制重构误差影响的超参数.
        """
        super(ReconWeightedInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self,
                anchor_emb,
                positive_emb,
                negative_embs,
                negative_recon_errors):
        """
        Args:
            anchor_emb (torch.Tensor): 锚点嵌入, [B, D].
            positive_emb (torch.Tensor): 正样本嵌入, [B, D].
            negative_embs (torch.Tensor): 所有负样本的嵌入, [N_neg, D].
            negative_recon_errors (torch.Tensor): 对应所有负样本的重构误差, [N_neg].
        """
        device = anchor_emb.device
        B, D = anchor_emb.shape
        N_neg = negative_embs.shape[0]


        l_pos = torch.einsum('bd,bd->b', anchor_emb, positive_emb).unsqueeze(-1)


        sim_neg = torch.einsum('bd,nd->bn', anchor_emb, negative_embs)


        min_err = negative_recon_errors.min()
        max_err = negative_recon_errors.max()


        if max_err - min_err > 1e-6:
            norm_recon_errors = (negative_recon_errors - min_err) / (max_err - min_err)
        else:
            norm_recon_errors = torch.zeros_like(negative_recon_errors)


        penalty = self.margin * (1.0 - norm_recon_errors)


        adjusted_sim_neg = sim_neg + penalty.unsqueeze(0)


        if N_neg >= B:
             mask = torch.eye(B, dtype=torch.bool, device=device)

             adjusted_sim_neg[:, :B].masked_fill_(mask, -float('inf'))


        logits = torch.cat([l_pos, adjusted_sim_neg], dim=1) / self.temperature


        labels = torch.zeros(B, dtype=torch.long, device=device)
        loss_per_sample = self.criterion(logits, labels)

        return loss_per_sample

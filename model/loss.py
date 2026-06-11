import torch
import torch.nn as nn


class ReconstructionLoss1(nn.Module):
    def __init__(self):
        super(ReconstructionLoss1, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, recon, target):
        loss = self.mse(recon, target)
        return loss.mean(dim=1)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, predictions, labels):
        return self.bce(predictions, labels)


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, recon, target):
        return self.mse(recon, target)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds, target):
        return self.bce(preds, target)


class ReconWeightedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, margin=0.5):
        super(ReconWeightedInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, anchor_emb, positive_emb, negative_embs, negative_recon_errors):
        device = anchor_emb.device
        bsz, _ = anchor_emb.shape
        n_neg = negative_embs.shape[0]

        l_pos = torch.einsum("bd,bd->b", anchor_emb, positive_emb).unsqueeze(-1)
        sim_neg = torch.einsum("bd,nd->bn", anchor_emb, negative_embs)

        min_err = negative_recon_errors.min()
        max_err = negative_recon_errors.max()
        if max_err - min_err > 1e-6:
            norm_recon_errors = (negative_recon_errors - min_err) / (max_err - min_err)
        else:
            norm_recon_errors = torch.zeros_like(negative_recon_errors)

        adjusted_sim_neg = sim_neg + self.margin * (1.0 - norm_recon_errors).unsqueeze(0)

        if n_neg >= bsz:
            mask = torch.eye(bsz, dtype=torch.bool, device=device)
            adjusted_sim_neg[:, :bsz].masked_fill_(mask, -float("inf"))

        logits = torch.cat([l_pos, adjusted_sim_neg], dim=1) / self.temperature
        labels = torch.zeros(bsz, dtype=torch.long, device=device)
        return self.criterion(logits, labels)

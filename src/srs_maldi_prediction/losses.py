import torch
import torch.nn as nn
import numpy as np


class SpectrumLoss(nn.Module):
    """Spectrum cosine similarity based loss."""

    def __init__(self):
        super(SpectrumLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, predicted, target):
        # predicted/target: [B, C, H, W]
        predicted_flat = predicted.view(predicted.size(0), predicted.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        cos_sim = self.cosine_similarity(predicted_flat, target_flat)
        spectrum_loss = 1 - cos_sim
        return spectrum_loss.mean()


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss (mean angle across channels)."""

    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, input, target):
        # input/target: [B, C, H, W]
        input_flat = input.view(input.size(0), input.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        inner_product = torch.sum(input_flat * target_flat, dim=2)
        input_norm = torch.norm(input_flat, dim=2)
        target_norm = torch.norm(target_flat, dim=2)
        cos_theta = inner_product / (input_norm * target_norm + 1e-8)
        angle = torch.acos(torch.clamp(cos_theta, -1 + 1e-8, 1 - 1e-8))
        loss = torch.mean(angle)
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier (smooth L1-like) loss."""

    def __init__(self, eps=1e-8):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps ** 2))
        return loss / (c * b * h * w)


def compute_channel_correlation(pred, target):
    """Compute per-channel Pearson correlation for pred/target tensors of shape [N, C, H, W]."""
    N, C, H, W = pred.shape
    pred_flat = pred.permute(1, 0, 2, 3).reshape(C, -1)
    target_flat = target.permute(1, 0, 2, 3).reshape(C, -1)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    pred_std = pred_flat.std(dim=1, unbiased=False) + 1e-8
    target_std = target_flat.std(dim=1, unbiased=False) + 1e-8
    covariance = ((pred_flat - pred_mean) * (target_flat - target_mean)).mean(dim=1)
    correlation = covariance / (pred_std * target_std)
    return correlation


def spectral_angle_mapper(predicted, target):
    """NumPy implementation of SAM for [H, W, C] images."""
    assert predicted.shape == target.shape
    height, width, channels = predicted.shape
    pred_flat = predicted.reshape(-1, channels)
    target_flat = target.reshape(-1, channels)
    dot_product = np.sum(pred_flat * target_flat, axis=1)
    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)
    epsilon = 1e-8
    cos_theta = dot_product / (pred_norm * target_norm + epsilon)
    cos_theta = np.clip(cos_theta, -1 + 1e-7, 1 - 1e-7)
    angle = np.arccos(cos_theta)
    mean_angle = np.mean(angle)
    return mean_angle

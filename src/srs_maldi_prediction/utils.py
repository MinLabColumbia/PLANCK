"""Utility helpers: prediction, patch restoration, evaluation metrics and plotting."""

import os
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import cv2


def prediction(model, test_loader, device=None):
    """Run model inference over a dataloader and return a list of HxWxC numpy arrays."""
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            outputs = np.transpose(outputs, (0, 2, 3, 1))
            preds.extend(outputs)
    return preds


def restore_image(patches, original_shape, patch_size=512):
    """Restore full image from non-overlapping patches.

    patches: iterable of HxWxC arrays in row-major order
    original_shape: (H, W, C)
    """
    N, M, C = original_shape
    full_image = np.zeros(original_shape, dtype=patches[0].dtype)
    num_patches_per_row = M // patch_size + (1 if M % patch_size != 0 else 0)
    for index, patch in enumerate(patches):
        row_index = index // num_patches_per_row
        col_index = index % num_patches_per_row
        start_row = row_index * patch_size
        start_col = col_index * patch_size
        end_row = start_row + patch.shape[0]
        end_col = start_col + patch.shape[1]
        end_row = min(end_row, N)
        end_col = min(end_col, M)
        full_image[start_row:end_row, start_col:end_col, :] = patch[:end_row-start_row, :end_col-start_col, :]
    return full_image


def restore_image_with_overlap(patches, original_shape, patch_size=512, overlap=0):
    """Restore image from possibly overlapping patches by averaging overlaps.

    patches are assumed to be provided in scanning order (row by row).
    """
    N, M, C = original_shape
    step = patch_size - overlap
    patches_per_row = int(np.ceil((M - overlap) / step))
    patches_per_column = int(np.ceil((N - overlap) / step))
    full_image = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    patch_index = 0
    for row in range(patches_per_column):
        for col in range(patches_per_row):
            if patch_index >= len(patches):
                break
            start_row = row * step
            start_col = col * step
            end_row = min(start_row + patch_size, N)
            end_col = min(start_col + patch_size, M)
            patch = patches[patch_index]
            patch_index += 1
            full_image[start_row:end_row, start_col:end_col, :] += patch[0:(end_row-start_row), 0:(end_col-start_col), :]
            count_map[start_row:end_row, start_col:end_col, :] += 1
    full_image /= np.maximum(count_map, 1)
    return full_image


def create_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.max(kernel)


def restore_patches(patches_list, large_image_shape, patch_size, sigma=50):
    """Blend tiled patches back into a large image using a Gaussian weight window."""
    H, W, C = large_image_shape
    patch_height, patch_width = patch_size, patch_size
    large_image = np.zeros(large_image_shape, dtype=np.float32)
    weight_matrix = np.zeros(large_image_shape, dtype=np.float32)
    gaussian_weights = create_gaussian_kernel(patch_size, sigma).astype(np.float32)
    gaussian_weights = np.expand_dims(gaussian_weights, axis=2)
    patch_idx = 0
    for y in range(0, H, patch_height):
        for x in range(0, W, patch_width):
            current_patch_height = min(patch_height, H - y)
            current_patch_width = min(patch_width, W - x)
            patch = patches_list[patch_idx]
            patch_idx += 1
            valid_patch = patch[:current_patch_height, :current_patch_width, :]
            valid_gaussian_weights = gaussian_weights[:current_patch_height, :current_patch_width, :]
            large_image[y:y+current_patch_height, x:x+current_patch_width, :] += valid_patch * valid_gaussian_weights
            weight_matrix[y:y+current_patch_height, x:x+current_patch_width, :] += valid_gaussian_weights
    large_image = np.divide(large_image, weight_matrix, where=weight_matrix!=0)
    return large_image


def evaluate(pred_img, maldi):
    """Evaluator: applies exp-transform and computes per-channel metrics.

    Returns: numpy array shape [C, 5] with [mse, mape, psnr, ssim, pearson]
    """
    metric_all = []
    maldi_t = np.exp(maldi) - 1
    pred_t = np.exp(pred_img) - 1
    for idx in range(maldi_t.shape[2]):
        maldi_spec = np.reshape(maldi_t[:, :, idx], (-1,))
        pred_spec = np.reshape(pred_t[:, :, idx], (-1,))
        mse = mean_squared_error(maldi_spec, pred_spec)
        mape = mean_absolute_percentage_error(maldi_spec, pred_spec)
        if PeakSignalNoiseRatio is not None:
            psnr = PeakSignalNoiseRatio(data_range=np.max(maldi_t[:, :, idx]))
            psnr_score = psnr(torch.from_numpy(maldi_t[:, :, idx]), torch.from_numpy(pred_t[:, :, idx]))
        else:
            psnr_score = np.nan
        if StructuralSimilarityIndexMeasure is not None:
            ssim = StructuralSimilarityIndexMeasure()
            ssim_score = ssim(torch.from_numpy(pred_t[:, :, idx]).unsqueeze(0).unsqueeze(0),
                              torch.from_numpy(maldi_t[:, :, idx]).unsqueeze(0).unsqueeze(0))
        else:
            ssim_score = np.nan
        try:
            corr, _ = pearsonr(maldi_spec, pred_spec)
        except Exception:
            corr = np.nan
        metric = [mse, mape, psnr_score, ssim_score, corr]
        metric_all.append(metric)
    metric_all = np.asarray(metric_all)
    print(np.nanmean(metric_all, axis=0))
    return metric_all


def spectral_angle_mapper(predicted, target):
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


def plot_paired_image(maldi, pred, idx, vmax=None, vmin=None, vmax2=None, vmin2=None, outpath=None):
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=False)
    axs[0].imshow(cv2.rotate(maldi[:, :, idx], cv2.ROTATE_90_CLOCKWISE), vmax=vmax2, vmin=vmin2, cmap='jet')
    axs[0].set_title("MALDI m/z {}".format(idx))
    axs[1].imshow(cv2.rotate(pred[:, :, idx], cv2.ROTATE_90_CLOCKWISE), vmax=vmax, vmin=vmin, cmap='jet')
    axs[1].set_title("FTIR Prediction")
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    if outpath:
        plt.savefig(outpath)
    fig.tight_layout()

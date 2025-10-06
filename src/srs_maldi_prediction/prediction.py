"""Prediction utilities: separate prediction and evaluation.

This module provides two main functions:
- `predict` : load a model checkpoint, run the model on a PairedNPYDataset and return patches or restored full image.
- `evaluate` : compute per-channel and aggregate metrics between predicted and target full images.

The implementation separates prediction and evaluation concerns so you can call them independently.
"""

import os
import warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import HyperPix2pixGenerator
from datasets import PairedNPYDataset
from utils import prediction as run_prediction, restore_image_with_overlap
from config import MODEL_DIR, DATA_DIR

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from scipy.stats import pearsonr


def predict(checkpoint_path: str,
            test_image_files: List[str],
            test_target_files: Optional[List[str]] = None,
            batch_size: int = 8,
            device: Optional[torch.device] = None,
            out_dir: Optional[str] = None,
            input_nc: int = 4,
            output_nc: int = 100,
            ngf: int = 64,
            num_downs: int = 4,
            patch_size: int = 512,
            use_dropout: bool = False) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Load model checkpoint and run prediction on test dataset.

    Args:
        checkpoint_path: path to generator checkpoint (.pth)
        test_image_files: list of .npy file paths for inputs
        test_target_files: optional list of .npy file paths for targets (used only to construct PairedNPYDataset)
        batch_size: dataloader batch size
        device: torch device; if None, auto-detect
        out_dir: optional directory to save predictions (if provided)
        input_nc/output_nc/ngf/num_downs/use_dropout: model hyperparameters used by the training pipeline

    Returns:
        (patches, full_image_or_None)
        - patches: list of predicted patches as numpy arrays (H x W x C)
        - full_image_or_None: restored full image if restore_to provided, else None
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print("device: ", device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # build dataset/dataloader
    if test_target_files is None:
        # PairedNPYDataset requires target list; create dummy zero arrays on the fly by reusing inputs
        test_target_files = test_image_files

    testset = PairedNPYDataset(test_image_files, test_target_files, transform=None)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # instantiate model and load weights
    model = HyperPix2pixGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, num_downs=num_downs, use_dropout=use_dropout)
    # allow passing just a filename which will be resolved against MODEL_DIR
    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        candidate = os.path.join(MODEL_DIR, checkpoint_path)
        if os.path.exists(candidate):
            checkpoint_path = candidate

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only = True)
    # allow loading either state_dict or full model checkpoint
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)

    patches = run_prediction(model, testloader, device=device)

    if out_dir:
        # save patches as a single numpy file (object dtype to preserve ragged arrays)
        np.save(os.path.join(out_dir, 'pred_patches.npy'), np.asarray(patches, dtype=object))

    return patches


def convert_patches_to_full_image(patches: np.ndarray, restore_to: dict, out_dir=None) -> np.ndarray:
    '''
    Convert pred patches to full images
    '''
    full_image = None
    shape = restore_to['shape']
    patch_size = restore_to.get('patch_size', patch_size)
    overlap = restore_to.get('overlap', 0)
    full_image = restore_image_with_overlap(patches, shape, patch_size=patch_size, overlap=overlap)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'pred_full.npy'), full_image)
    return full_image


def evaluate(pred_img: np.ndarray, maldi: np.ndarray) -> np.ndarray:
    '''
    Perform evaluation on prediction
    '''
    # apply exp transform to original scale
    maldi = np.exp(maldi) - 1
    pred_img = np.exp(pred_img) - 1
    metric_all = []
    for idx in range(maldi.shape[2]):
        maldi_spec = np.reshape(maldi[:,:,idx], (maldi.shape[0]*maldi.shape[1]))
        pred_spec = np.reshape(pred_img[:,:,idx], (maldi.shape[0]*maldi.shape[1]))
        mse = mean_squared_error(maldi_spec, pred_spec)

        psnr = PeakSignalNoiseRatio(data_range = np.max(maldi[:,:,idx]))
        psnr_score = psnr(torch.from_numpy(pred_img[:,:,idx]), torch.from_numpy(maldi[:,:,idx]))

        ssim = StructuralSimilarityIndexMeasure()
        ssim_score = ssim(torch.from_numpy(pred_img[:,:,idx]).unsqueeze(0).unsqueeze(0), 
                          torch.from_numpy(maldi[:,:,idx]).unsqueeze(0).unsqueeze(0))

        corr, _ = pearsonr(maldi_spec, pred_spec)
        metric = [mse, psnr_score, ssim_score, corr]
        metric_all.append(metric)
  
    metric_all = np.asarray(metric_all)
    return metric_all


def spectral_angle_mapper(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute spectral angle mapper (SAM) averaged over pixels (not torchmetrics version).
    """
    assert predicted.shape == target.shape, "Predicted and target images must have the same shape."
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


def evaluate_patches(patches: List[np.ndarray], 
                     test_targets: List[str], 
                     out_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Run evaluation on patches directly

    Args:
        patches: list of patch arrays (H x W x C)
        test_targets: list of target paths
        out_dir: optional directory to save restored image and metrics

    Returns:
        {'metrics': metric_all, 'sam': sam_value}
    """
    target = np.load(test_targets[0])
    metric_all = []
    for patch_idx in range(len(patches)):
        pred_img = patches[patch_idx]
        target_img = target[patch_idx]
        metric_patch = evaluate(pred_img, target_img)
        metric_all.append(np.mean(metric_patch, axis =0))
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'metrics.npy'), metric_all)

    return metric_all


if __name__ == '__main__':
    # Example usage (adjust paths before running)
    # resolve example checkpoint and data files against configured folders
    ckpt = os.path.join(MODEL_DIR, 'srs_maldi', 'hyperpix2pix_best_generator.pth')
    test_images = [
        os.path.join(DATA_DIR, 'srs_maldi', 'patch_512', 'srs_quarter_coronal_126_512_sample.npy'),
    ]
    test_targets = [
        os.path.join(DATA_DIR, 'srs_maldi', 'patch_512', 'maldi_quarter_coronal_126_512_bc_sample.npy'),
    ]
    
   
    patches = predict(ckpt, test_images, 
                            test_targets, 
                            batch_size=4, 
                            patch_size=512,
                            out_dir=os.path.join(DATA_DIR, 'pred_sample'), 
                            )
    print("predict patch length: ", len(patches))

    
    # Evaluate on full image if has full image path
    '''
     test_targets_full = [
        os.path.join(DATA_DIR, 'srs_maldi', 'full_data', 'quarter_coronal_126_maldi_147_batch_correct_upscale30.npy'),
    ]
    target_full = np.load(test_targets_full[0])
    restore_to = {
        'shape': target_full.shape,  # set to full image shape if you have it, e.g. (H, W, C)
        'patch_size': 512,
        'overlap': 0,
        'target_path': test_targets[0]
    }
    full_image = convert_patches_to_full_image(patches, restore_to, output_dir=None)
    print('Predicted full image shape:', full.shape)
    metrics = evaluate(full, target_full) 
    print("Eval metrics, [mse, psnr, ssim, corr]: ", np.mean(metrics, axis = 0))
    '''



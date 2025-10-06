"""Load a saved PLS model, run prediction and evaluation.

This script complements `build_pls.py` and provides CLI options to load a model
checkpoint, run predictions on provided IR files (paired with MALDI ground truth
if available), and print/save evaluation metrics.
"""

import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch

def load_data_to_spec(path: str) -> np.ndarray:
    img = np.load(path)
    spec = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    spec = spec[:,1:208]
    return spec


def rms_norm(spec: np.ndarray) -> np.ndarray:
    tic = np.sqrt(np.mean(np.power(spec, 2)))
    return spec / (tic + 1e-12)


def build_train_sets(ir_paths: List[str], maldi_paths: List[str], apply_rms: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for ir_path, maldi_path in zip(ir_paths, maldi_paths):
        X = load_data_to_spec(ir_path)
        y = np.load(maldi_path)
        if y.ndim == 3:
            y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])
        if apply_rms:
            X = rms_norm(X)
        X_list.append(X)
        y_list.append(y)

    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)
    return X_all, y_all


def evaluate(pred_img: np.ndarray, maldi: np.ndarray) -> np.ndarray:
    
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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--ir-files', nargs='+', required=True)
    p.add_argument('--maldi-files', nargs='*', default=[], help='Optional ground-truth MALDI files to evaluate against')
    p.add_argument('--model-checkpoint', required=True, help='Path to saved PLS model (.pkl)')
    # by default do not apply RMS normalization; use --rms to enable
    p.add_argument('--rms', dest='apply_rms', action='store_true', help='Enable RMS normalization of IR (default: False)')
    p.set_defaults(apply_rms=False)
    # by default apply exp-to-maldi; provide a flag to disable it
    p.add_argument('--no-exp-to-maldi', dest='apply_exp_to_maldi', action='store_false', help='Disable exp(x)-1 transform on MALDI (default: apply)')
    p.set_defaults(apply_exp_to_maldi=True)
    p.add_argument('--out-pred', type=str, help='Optional path to save predictions as .npy')
    args = p.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if '__file__' in globals() else os.getcwd()
    ir_paths = [os.path.join(repo_root, p) if not os.path.isabs(p) else p for p in args.ir_files]
    maldi_paths = [os.path.join(repo_root, p) if not os.path.isabs(p) else p for p in args.maldi_files]

    print('Loading model from', args.model_checkpoint)
    with open(args.model_checkpoint, 'rb') as f:
        model = pickle.load(f)

    print('Loading IR data...')
    X = np.vstack([load_data_to_spec(p) for p in ir_paths])
    if args.apply_exp_to_maldi:
        print('Note: apply-exp-to-maldi is provided but will only affect evaluation when MALDI ground truth is provided')
    if args.apply_rms:
        X = np.vstack([rms_norm(load_data_to_spec(p)) for p in ir_paths])

    print('Predicting...')
    pred = model.predict(X)

    if args.out_pred:
        out_dir = os.path.dirname(args.out_pred)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        np.save(args.out_pred, pred)
        print('Saved predictions to', args.out_pred)

    if maldi_paths:
        print('Loading MALDI ground truth...')
        y = np.vstack([np.load(p) for p in maldi_paths])
        if args.apply_exp_to_maldi:
            try:
                y = np.exp(y) - 1
                y[y < 0] = 0
            except Exception:
                print('Warning: failed to apply exp transform to maldi; continuing with raw values')

        print('Evaluating predictions...')
        pred = pred.reshape(y.shape)
        metrics = evaluate(pred, y)
        print("Eval metrics, [mse, psnr, ssim, corr]: ", np.mean(metrics, axis = 0))


if __name__ == '__main__':
    main()

"""Train and save a PLS regression model.

This script is derived from pls_pipeline.py and focuses on training only.
"""

import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression


def load_data_to_spec(path: str) -> np.ndarray:
    img = np.load(path)
    spec = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    spec = spec[:,1:208] # Remove wavenumbers at two ends which are nealry 0 across all pixels in ir data
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


def build_model(n_components: int = 10) -> PLSRegression:
    return PLSRegression(n_components=n_components)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--ir-files', nargs='+', required=True)
    p.add_argument('--maldi-files', nargs='+', required=True)
    p.add_argument('--n-components', type=int, default=10)
    p.add_argument('--out-model', type=str, default='model_checkpoints/pls_model.pkl')
    # by default do not apply RMS normalization; use --rms to enable
    p.add_argument('--rms', dest='apply_rms', action='store_true', help='Enable RMS normalization of IR (default: False)')
    p.set_defaults(apply_rms=False)
    # by default apply exp-to-maldi; provide a flag to disable it
    p.add_argument('--no-exp-to-maldi', dest='apply_exp_to_maldi', action='store_false', help='Disable exp(x)-1 transform on MALDI (default: apply)')
    p.set_defaults(apply_exp_to_maldi=True)
    args = p.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if '__file__' in globals() else os.getcwd()
    ir_paths = [os.path.join(repo_root, p) if not os.path.isabs(p) else p for p in args.ir_files]
    maldi_paths = [os.path.join(repo_root, p) if not os.path.isabs(p) else p for p in args.maldi_files]

    print('Loading data...')
    X, y = build_train_sets(ir_paths, maldi_paths, apply_rms=args.apply_rms)
    if args.apply_exp_to_maldi:
        try:
            y = np.exp(y) - 1
            y[y < 0] = 0
        except Exception:
            print('Warning: failed to apply exp transform to maldi; continuing with raw values')

    print('Data shapes:', X.shape, y.shape)

    print('Building model...')
    model = build_model(n_components=args.n_components)

    print('Fitting model...')
    model.fit(X, y)

    out_model_path = args.out_model
    out_dir = os.path.dirname(out_model_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_model_path, 'wb') as f:
        pickle.dump(model, f)
    print('Model saved to', out_model_path)


if __name__ == '__main__':
    main()

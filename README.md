# PLANCK
© 2025 The Trustees of Columbia University in the City of New York. The codes and data in this work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only. To obtain a license to the codes and data in this work for commercial purposes, please contact Columbia Technology Ventures at techventures@columbia.edu.

Overview
--------
This is repository for PLANCK: super-multiplex optical imaging without labeling.
It contains codes for label free multiplex imaging using vibrational imaging (IR and SRS) to predict MALDI imaging data:
- A classical machine-learning pipeline (PLS) that predicts MALDI from IR spectra. This is implemented as a simple script for reproducible runs.
- A self-designed deep-learning pipeline (Hyperpix2pix) based on pix2pix that predicts MALDI from SRS images, with specific optimization for hyperspectral image prediction. This is provided as a package under `src/srs_maldi_prediction/` and includes training and prediction utilities

Repository structure
--------------------

- `src/ir_maldi_prediction/` Training and prediction scripts: `build_pls.py` (train) and `predict_pls.py` (predict/evaluate).
- `src/srs_maldi_prediction/` — package implementing the deep-learning model Hy
- `data/` — data location. Paired sample ir and maldi data is stored in `data/ir_malid/full_data/`.
Paired sample srs and maldid data is in `data/srs_malid/patch_512/`, which are data patches split in 512 sizes. Note that we only include sample (partial) data in this repo, due to the large data size (more than 5GB per image file of mouse whole brain tissue section. The performance in the sample data does not reflect the performance of full data. The full data can be obtained from the corresponding author upon reasonable requests. 
- `model_checkpoints/` — trained model checkpoints for both pls and deep learning
- `requirements.txt` — suggested Python packages; adjust `torch` to match your CUDA runtime when needed.

IR→MALDI (PLS) — quick summary
------------------------------------

Location: `src/ir_maldi_prediction/` (see `build_pls.py` and `predict_pls.py`)

What it does:
- Loads IR and MALDI .npy files (full images or flattened H*W x C arrays).
- Optionally applies the exp(x) - 1 transform to MALDI values (enabled by default in the scripts).
- Trains a Partial Least Squares regression model using scikit-learn's `PLSRegression`.
- Saves the trained model to `model_checkpoints/` and provides a separate prediction/evaluation script.


Deep learning (SRS→MALDI) — quick summary
----------------------------------------

Location: `src/srs_maldi_prediction/` (package)

What it contains:
- `model.py` — U-Net generator (`HyperPix2pixGenerator`) and PatchGAN discriminator (`HyperPix2pixDiscriminator`).
- `datasets.py` — `PairedNPYDataset` and augmentation utilities.
- `losses.py` — custom losses (SAM, Charbonnier, spectrum/cosine losses) and helpers.
- `utils.py` — prediction loop, patch stitching (with overlap handling), evaluation helpers.
- `train.py` — training loop and `train_entrypoint` convenience function.
- `prediction.py` — checkpoint loading, prediction, and evaluation utilities (notebook-style `evaluate` is used).
- `config.py` — default paths and hyperparameters (DATA_DIR, MODEL_DIR, default hyperparams).

Key defaults and design choices
-----------------------------
- Patch size default: 512.
- Train/validation split: 85/15 (fixed seed for reproducibility).
- Default hyperparameters: batch_size=16, pretrain_epochs=50, num_epochs=200, lambda_L2=500.
- Checkpoint saving: models are saved under `model_checkpoints/` by default.

Quick start
-----------
1. Create a Python environment and install dependencies. Installing `torch` should follow the instructions for your CUDA version. Example (CPU-only):

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. PLS model (example):

Test the trained model using saved checkpoint
```bash
python src/ir_maldi_prediction/predict_pls.py --ir-files data/ir_maldi/full_data/ir_sagittal_1_sample.npy --maldi-files data/ir_maldi/full_data/maldi_sagittal_1_sample.npy --model-checkpoint model_checkpoints/ir_maldi/pls_model.pkl
```

3. Run deep-learning (quick example):

```bash
python src/srs_maldi_prediction/train.py
```

4. Run prediction with a saved model checkpoint:

```bash
python src/srs_maldi_prediction/prediction.py
```







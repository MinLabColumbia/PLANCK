import os

# Data paths (adjust as needed)
ROOT = os.path.dirname(__file__)
# package is located at <repo>/src/srs_maldi_prediction -> repo root is two levels up
REPO_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
MODEL_DIR = os.path.join(REPO_ROOT, 'model_checkpoints')
FIG_DIR = os.path.join(REPO_ROOT, 'figures')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Training defaults
DEFAULTS = {
    'batch_size': 16,
    'num_epochs': 200,
    'pretrain_epochs': 50,
    'lr_G': 1e-4,
    'lr_D': 4e-5,
    'lambda_L2': 500,
    'lambda_cosine': 1
}

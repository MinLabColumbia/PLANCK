import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate


class CustomTransform:
    def __init__(self):
        pass

    def __call__(self, img, target):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            target = np.flip(target, axis=1).copy()
        random.seed(seed)
        if random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
            target = np.flip(target, axis=0).copy()
        angle = random.uniform(-30, 30)
        img = self.rotate_image(img, angle)
        target = self.rotate_image(target, angle)
        return img, target

    def rotate_image(self, img, angle):
        rotated_img = rotate(img, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
        return rotated_img


class PairedNPYDataset(Dataset):
    def __init__(self, image_files, target_files, transform=None):
        self.image_files = image_files
        self.target_files = target_files
        self.data_indices = self._get_data_indices()
        self.transform = transform

    def _get_data_indices(self):
        indices = []
        for file_idx, (image_file, target_file) in enumerate(zip(self.image_files, self.target_files)):
            image_array = np.load(image_file, mmap_mode='r')
            target_array = np.load(target_file, mmap_mode='r')
            if len(image_array) != len(target_array):
                raise ValueError(f"Mismatch in number of images and targets in file pair {file_idx}: {len(image_array)} images, {len(target_array)} targets")
            indices.extend([(file_idx, i) for i in range(len(image_array))])
        return indices

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        file_idx, data_idx = self.data_indices[idx]
        image_file = self.image_files[file_idx]
        target_file = self.target_files[file_idx]
        image_array = np.load(image_file, mmap_mode='r')
        target_array = np.load(target_file, mmap_mode='r')
        img = image_array[data_idx]
        target = target_array[data_idx]
        if self.transform:
            img, target = self.transform(img, target)
        img = np.transpose(img, (2, 0, 1))
        target = np.transpose(target, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        target[target < 0] = 0
        return img, target

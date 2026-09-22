import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class Train_Dataset(Dataset):
    def __init__(self, image_dir, filelist, patch_size=(512, 512)):
        self.image_dir = image_dir
        self.file_list = os.path.join(self.image_dir, filelist)
        with open(self.file_list) as f:
            self.input_names = [i.strip() for i in f.readlines()]
        self.patch_size = patch_size

    def get_images(self, index):
        low_img_name = self.input_names[index]
        low_img = Image.open(low_img_name)
        low_img = self.random_crop(np.asarray(low_img))
        
        return {"low_img": low_img}

    def random_crop(self, img, start=None, patch_size=None):
        height, width = img.shape[:2]
        patch_size_h, patch_size_w = patch_size or self.patch_size

        if start is None:
            x = random.randint(0, width - patch_size_w - 1)
            y = random.randint(0, height - patch_size_h - 1)
        else:
            x, y = start

        img_patch = img[y: y + patch_size_h, x: x + patch_size_w, :]

        if np.random.randint(2):  # Random horizontal flip
            img_patch = np.flip(img_patch, axis=1)
        if np.random.randint(2):  # Random vertical flip
            img_patch = np.flip(img_patch, axis=0)

        return torch.tensor(img_patch.copy() / 255.0).permute(2, 0, 1).float()

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)

class Test_Dataset(Dataset):
    def __init__(self, image_dir, filelist):
        self.image_dir = image_dir
        self.file_list = os.path.join(self.image_dir, filelist)
        with open(self.file_list) as f:
            self.input_names = [i.strip() for i in f.readlines()]

    def get_images(self, index):
        low_img_name = self.input_names[index]
        img_name = os.path.basename(low_img_name)
        
        low_img = np.asarray(Image.open(low_img_name))
        low_img = torch.tensor(low_img / 255.0).permute(2, 0, 1).float()
        
        return {"low_img": low_img, "img_name": img_name}

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)

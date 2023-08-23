import random
import cv2
import torch

from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset


class PIEDataset(Dataset):
    def __init__(self, data_dir, input_size=(400, 640), geo_aug=True):
        self.data_dir = data_dir
        self.img_paths = self.load_items()
        self.input_size = input_size
        self.geo_aug = geo_aug
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.4330], std=[0.2349])
        ])

    def load_items(self):
        img_paths = []
        mode = self.data_dir.name
        label_file = self.data_dir / f'{mode}.txt'
        f = open(str(label_file), mode='r', encoding='utf-8')
        for item in f.readlines():
            # extract image path
            item_parts = item.strip().split(';')
            img_path = self.data_dir / item_parts[0]
            # load to list
            img_paths.append(img_path)
        return img_paths

    def __getitem__(self, idx):
        # Read ground truth image
        gt_img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
        gt_img = TF.to_tensor(gt_img)

        # Apply random degradation to create input image
        in_img = gt_img.clone()
        # Add random noise
        noise_type = random.choice(['gaussian'])  # TODO: add more noise types  
        if noise_type == 'gaussian':
            mean = 0
            var = random.uniform(0.01, 0.05)
            sigma = var ** 0.5
            noise = torch.randn(in_img.size()) * sigma + mean
            in_img = torch.clamp(in_img + noise, 0, 1)
        contrast_factor = random.uniform(0.5, 1.0)
        in_img = TF.adjust_contrast(in_img, contrast_factor)

        # Apply transform to both input and ground truth images
        in_img = self.transform(in_img)
        gt_img = self.transform(gt_img)

        # Random geometric transformation and resize image and heatmap
        if self.geo_aug:  # Resize image and heatmap
            in_img, gt_img = self.geometric_trans(in_img, gt_img)
        else:
            in_img, gt_img = self.aspect_resize(in_img, gt_img)

        return in_img, gt_img

    def geometric_trans(self, in_img, gt_img):
        # Random horizontal flip
        if random.random() < 0.5:
            in_img = TF.hflip(in_img)
            gt_img = TF.hflip(gt_img) if gt_img is not None else None

        # Random vertical flip
        if random.random() < 0.5:
            in_img = TF.vflip(in_img)
            gt_img = TF.vflip(gt_img) if gt_img is not None else None

        # Random rotation
        if random.random() < 0.2:
            angle = random.randint(-90, 90)
            in_img = TF.rotate(in_img, angle)
            gt_img = TF.rotate(gt_img, angle) if gt_img is not None else None

        # Random crop resize
        if random.random() < 0.8:
            in_img, gt_img = self.crop_resize(in_img, gt_img)
        else:
            in_img, gt_img = self.aspect_resize(in_img, gt_img)
        return in_img, gt_img
    
    def crop_resize(self, in_img, gt_img=None, crop_prob=0.2):
        # random crop
        if random.random() < crop_prob:
            img_size = list(in_img.shape[-2:])

            crop_rate = random.uniform(0.7, 0.9)
            crop_size = [int(img_size[0] * crop_rate), int(img_size[1] * crop_rate)]

            offset_rate = random.uniform(-0.2, 0.2)
            dy, dx = [int(img_size[0] * offset_rate), int(img_size[1] * offset_rate)]

            in_img = TF.crop(in_img, dy, dx, crop_size[0], crop_size[1])
            gt_img = TF.crop(gt_img, dy, dx, crop_size[0], crop_size[1]) if gt_img is not None else None
        # resize
        in_img = TF.resize(in_img, [self.input_size[0], self.input_size[1]], antialias=True)
        gt_img = TF.resize(gt_img, [self.input_size[0], self.input_size[1]], antialias=True) if gt_img is not None else None
        return in_img, gt_img
    
    def aspect_resize(self, in_img, gt_img=None):
        img_h, img_w = in_img.shape[-2:]
        scale = min(self.input_size[0] / img_h, self.input_size[1] / img_w)
        scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)
        dx = (self.input_size[1] - scaled_w) // 2
        dy = (self.input_size[0] - scaled_h) // 2

        scaled_in_img = TF.resize(in_img, [scaled_h, scaled_w], antialias=True)
        dst_in_img = torch.full((1, self.input_size[0], self.input_size[1]), 0.5, dtype=torch.float32)
        dst_in_img[:, dy:dy + scaled_h, dx:dx + scaled_w] = scaled_in_img

        scaled_gt_img = TF.resize(gt_img, [scaled_h, scaled_w], antialias=True) if gt_img is not None else None
        if scaled_gt_img is not None:
            dst_gt_img = torch.zeros((1, self.input_size[0], self.input_size[1]), dtype=torch.float32)
            dst_gt_img[:, dy:dy + scaled_h, dx:dx + scaled_w] = scaled_gt_img
        else:
            dst_gt_img = None

        return dst_in_img, dst_gt_img

    def __len__(self):
        return len(self.img_paths)


class PIEDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, input_size, batch_size):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.input_size = input_size
        self.batch_size = batch_size
        self.hmap_train = self.hmap_val = self.hmap_test = self.hmap_infer = None

    def setup(self, stage=None):
        # set image path
        train_dir = self.root_dir / 'train'
        val_dir = self.root_dir / 'test'
        test_dir = self.root_dir / 'test'
        # set dataset
        self.hmap_train = PIEDataset(train_dir, self.input_size, geo_aug=True)
        self.hmap_val = PIEDataset(val_dir, self.input_size, geo_aug=False)
        self.hmap_test = PIEDataset(test_dir, self.input_size, geo_aug=False)

    def train_dataloader(self):
        return DataLoader(self.hmap_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.hmap_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.hmap_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)

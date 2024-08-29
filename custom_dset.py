import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(self, root, maskroot, transforms_=None, img_size=128, mask_size=16, mode="train", classes=False,masking="random"):
        self.root = root
        self.maskroot = maskroot
        self.transform = transforms_
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.masking = masking
        self.classes = classes
        if classes:
            self.classes = sorted(os.listdir(root))
            self.classes_to_idx = {
                image_class: idx for idx, image_class in enumerate(self.classes)
            }
            self.files = []
            self.targets = []

            for image_class in self.classes:
                class_paths = list(
                    filter(
                        lambda x: x.endswith(".jpg") or x.endswith(".png"),
                        sorted(os.listdir(os.path.join(root, image_class))),
                    )
                )
                np.random.shuffle(class_paths)

                for image in class_paths:
                    image_path = os.path.join(root, image_class, image)
                    self.files.append(image_path)
                    self.targets.append(self.classes_to_idx[image_class])
        else:
            self.files = []
            for file in os.listdir(root):
                if file.endswith((".png", ".jpg")):
                    self.files.append(os.path.join(root, file))
            
        self.files = self.files[:30000]


    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1
        mask = torch.zeros((3, self.img_size, self.img_size))
        mask[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part, mask

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        return img,0

    def __len__(self):
        return len(self.files)
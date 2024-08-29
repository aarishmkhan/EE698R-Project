import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, maskroot, transforms_=None, img_size=128, mask_size=16, mode="train", classes=False,masking="random"):
        self.root = root
        self.maskroot = maskroot
        self.transform = transforms.Compose(transforms_)
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
            self.files = sorted(glob.glob("%s/*.png" % root))
            
            # self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

        if self.masking == "custom":
            self.mask_files = sorted(glob.glob("%s/*.png" % maskroot))


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
        if self.mode == "train":
            if self.masking == "random":
                # For training data perform random mask
                masked_img, aux, mask = self.apply_random_mask(img)
            # elif self.masking == "center":
            #     # For test data mask the center of the image
            #     masked_img, aux = self.apply_center_mask(img)
            elif self.masking == "custom":
                mask = Image.open(self.mask_files[index % len(self.mask_files)])
                mask = transforms.Resize((self.img_size,self.img_size))(mask)
                mask = transforms.Grayscale(num_output_channels=1)(mask)
                mask = transforms.ToTensor()(mask)
                mask = mask/255 # check if mask is binary mask or not
                masked_img = img * (1 - mask)
                aux = img * mask
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)
            return img, masked_img, aux

        return img, masked_img, aux, mask

    def __len__(self):
        return len(self.files)
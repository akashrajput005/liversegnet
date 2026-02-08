import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, mask

class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask

class RandomRotation:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        return image, mask

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask

class ToTensor:
    def __call__(self, image, mask):
        image = F.to_tensor(image)
        # Mask to long tensor for CrossEntropyLoss, or float for BCE
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return image, mask

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask

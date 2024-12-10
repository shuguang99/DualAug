# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, heavy_transform):
        self.base_transform = base_transform
        self.heavy_transform = heavy_transform

    def __call__(self, x):
        base1 = self.base_transform(x)
        base2 = self.base_transform(x)
        q = self.heavy_transform(x)
        k = self.heavy_transform(x)
        return [base1, base2, q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

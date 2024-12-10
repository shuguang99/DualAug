from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple
from torch.utils.data import Dataset
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import abc
import random
import os


class FewShotDataset(Dataset):

    num_classes: int = None
    class_names: int = None

    def __init__(self, examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 synthetic_dir: str = None):

        self.examples_per_class = examples_per_class
        self.generative_aug = generative_aug

        self.synthetic_probability = synthetic_probability
        self.synthetic_dir = synthetic_dir
        self.synthetic_examples = defaultdict(list)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5]),
        ])
        
        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented
    
    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def generate_augmentations(self, num_repeats: int):

        self.synthetic_examples.clear()
        options = product(range(len(self)), range(num_repeats))

        for idx, num in tqdm(list(
                options), desc="Generating Augmentations"):

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

            image, label = self.generative_aug(
                image, label, self.get_metadata_by_idx(idx))

            if self.synthetic_dir is not None:

                pil_image, image = image, os.path.join(
                    self.synthetic_dir, f"aug-{idx}-{num}.png")

                pil_image.save(image)

            self.synthetic_examples[idx].append((image, label))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.synthetic_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            image, label = random.choice(self.synthetic_examples[idx])
            if isinstance(image, str): image = Image.open(image)

        else:

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

        return self.transform(image), label


class DualFewShotDataset(Dataset):
    num_classes: int = None
    class_names: int = None

    def __init__(self, examples_per_class: int = None,
                 basic_generative_aug: GenerativeAugmentation = None,
                 heavy_generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 synthetic_dir: str = None):

        self.examples_per_class = examples_per_class
        self.basic_generative_aug = basic_generative_aug
        self.heavy_generative_aug = heavy_generative_aug

        self.synthetic_probability = synthetic_probability
        self.synthetic_dir = synthetic_dir
        self.synthetic_examples = defaultdict(list)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)

    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented

    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented

    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    from itertools import product
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import os

    def basic_generate_augmentations(self, model, num_repeats: int):
        self.synthetic_examples.clear()
        basic_scores = {}
        basic_options = product(range(len(self)), range(num_repeats))
        # 生成基本的增强图像
        for idx, num in tqdm(list(basic_options), desc="Generating Basic Augmentations"):
            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

            basic_image, label = self.basic_generative_aug(image, label, self.get_metadata_by_idx(idx))

            basic_score = torch.max(F.softmax(model(self.transform(basic_image).unsqueeze(0).cuda()) / 1000, dim=1), dim=1)[0]  # 修改为使用 image
            if self.synthetic_dir is not None:
                pil_image = basic_image
                image_path = os.path.join(self.synthetic_dir, f"basic_aug-{idx}-{num}.png")
                pil_image.save(image_path)
            self.synthetic_examples[idx].append((image_path, label))
            basic_scores[f"basic_aug-{idx}-{num}.png"] = basic_score

        # 计算基本增强分数的均值和标准差
        basic_mean = torch.mean(torch.stack(list(basic_scores.values())), dim=0)
        basic_std = torch.std(torch.stack(list(basic_scores.values())), dim=0)
        self.threshold = basic_mean - 1.0 * basic_std

    def dual_generate_augmentations(self, model, num_repeats: int):
        model.eval()
        self.synthetic_examples.clear()
        heavy_scores = {}
        heavy_options = product(range(len(self)), range(num_repeats))

        # Dualaug
        for idx, num in tqdm(list(heavy_options), desc="Generating Heavy Augmentations"):
            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

            heavy_image, label = self.heavy_generative_aug(image, label, self.get_metadata_by_idx(idx))
            heavy_score = torch.max(F.softmax(model(self.transform(heavy_image).unsqueeze(0).cuda()) / 1000, dim=1), dim=1)[0]
            if self.synthetic_dir is not None:
                pil_image = heavy_image
                image_path = os.path.join(self.synthetic_dir, f"heavy_aug-{idx}-{num}.png")
                pil_image.save(image_path)

            heavy_scores[f"heavy_aug-{idx}-{num}.png"] = heavy_score

        id_key = [key for key, heavy_score in heavy_scores.items() if heavy_score > self.threshold]
        dual_options = product(range(len(self)), range(num_repeats))

        for idx, num in tqdm(list(dual_options), desc="Generating Dual Augmentations"):
            if f"heavy_aug-{idx}-{num}.png" in id_key:
                image_path = os.path.join(self.synthetic_dir, f"heavy_aug-{idx}-{num}.png")
            else:
                image_path = os.path.join(self.synthetic_dir, f"basic_aug-{idx}-{num}.png")

            pil_image = Image.open(image_path)
            save_path = os.path.join(self.synthetic_dir, f"aug-{idx}-{num}.png")
            pil_image.save(save_path)
            self.synthetic_examples[idx].append((save_path, label))
        model.train()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.synthetic_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            image, label = random.choice(self.synthetic_examples[idx])
            if isinstance(image, str): image = Image.open(image)

        else:

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

        return self.transform(image), label
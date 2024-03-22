import copy
import logging
import os

import PIL
from typing import Optional, Callable

import torchvision
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C
from torch.utils.data import SubsetRandomSampler, Sampler
from torch.utils.data.dataset import ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms


from Augment.augmentations import *

from Augment.augmentations import Lighting
from Augment.common import get_logger, copy_and_replace_transform, stratified_split, denormalize
from Augment.imagenet import ImageFolder

logger = get_logger('Augment')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) # these are for CIFAR 10, not for cifar100 actaully. They are pretty similar, though.


import os
from typing import Optional, Callable

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            heavy_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root=root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
        self.heavy_transform = heavy_transform
        self.pre_transform = transforms.Compose(transform.transforms[:4])
        self.norm_transform = transforms.Compose(transform.transforms[4:6])
        if C.get()['cutout'] > 0:
            self.cutout_transform = transform.transforms[6]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if (self.transform is not None) and (self.heavy_transform is not None):
            img = self.pre_transform(img)
            if C.get()['cutout'] > 0:
                basic_img, cutout_x, cutout_y = self.cutout_transform(self.norm_transform(img), get_info=True)
                heavy_img = self.cutout_transform(self.norm_transform(self.heavy_transform(img)), x=cutout_x, y=cutout_y,
                                               use_info=True)
            else:
                basic_img = self.norm_transform(img)
                heavy_img = self.norm_transform(self.heavy_transform(img))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return basic_img, heavy_img, target, index

# CIFAR100 default
class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            heavy_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR100, self).__init__(root=root,
                                      train=train,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)
        self.heavy_transform = heavy_transform
        self.pre_transform = transforms.Compose(transform.transforms[:4])
        self.norm_transform = transforms.Compose(transform.transforms[4:6])
        if C.get()['cutout'] > 0:
            self.cutout_transform = transform.transforms[6]
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if (self.transform is not None) and (self.heavy_transform is not None):
            img = self.pre_transform(img)
            if C.get()['cutout'] > 0:
                basic_img, cutout_x, cutout_y = self.cutout_transform(self.norm_transform(img), get_info=True)
                heavy_img = self.heavy_transform(img)
                heavy_img = self.cutout_transform(self.norm_transform(heavy_img), x=cutout_x,
                                                   y=cutout_y,
                                                   use_info=True)
            else:
                basic_img = self.norm_transform(img)
                if C.get()['heavy_randaug'].get('softaug', False) == True:
                    heavy_img, k = self.heavy_transform(img)
                else:
                    heavy_img = self.heavy_transform(img)
                heavy_img = self.norm_transform(heavy_img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if C.get()['heavy_randaug'].get('softaug', False) == True:
            return basic_img, heavy_img, target, index, k
        else:
            return basic_img, heavy_img, target, index

class SVHN(torchvision.datasets.SVHN):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        heavy_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(SVHN, self).__init__(root=root,
                                      split=split,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)
        self.heavy_transform = heavy_transform
        self.pre_transform = transforms.Compose(transform.transforms[:2])
        self.norm_transform = transforms.Compose(transform.transforms[2:4])
        self.cutout_transform = transform.transforms[4]
    def __getitem__(self, index: int) :
        img, target = self.data[index], int(self.labels[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if (self.transform is not None) and (self.heavy_transform is not None):
            img = self.pre_transform(img)
            basic_img, cutout_x, cutout_y = self.cutout_transform(self.norm_transform(img), get_info=True)
            heavy_img = self.cutout_transform(self.norm_transform(self.heavy_transform(img)), x=cutout_x, y=cutout_y,
                                               use_info=True)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return basic_img, heavy_img, target, index

def get_dataloaders(dataset, batch, dataroot, split=0.15, split_idx=0, distributed=False, started_with_spawn=False, summary_writer=None, try_heavy=True):
    print(f'started with spawn {started_with_spawn}')

    dataset_info = {}
    pre_transform_train = transforms.Compose([])
    if 'heavy_randaug' in C.get():
        # heavy aug
        assert not C.get()['heavy_randaug'].get('corrected_sample_space') and not C.get()['heavy_randaug'].get(
            'google_augmentations')
        if C.get()['heavy_randaug'].get('softaug', False) == True :
            heavy_transform_train = aug_lib.UniAugmentWeightedWithk(probs=C.get()['heavy_randaug'].get('weights',
                                                                                                         None))
        else:
            heavy_transform_train = get_randaugment(n=C.get()['heavy_randaug']['N'],
                                                 m=C.get()['heavy_randaug']['M'],
                                                 weights=C.get()['heavy_randaug'].get('weights',
                                                                                       None),
                                                 bs=C.get()['batch'])

    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        dataset_info['mean'] = _CIFAR_MEAN
        dataset_info['std'] = _CIFAR_STD
        dataset_info['img_dims'] = (3,32,32)
        dataset_info['num_labels'] = 100 if '100' in dataset and 'ten' not in dataset else 10
    elif 'pre_transform_cifar' in dataset:
        pre_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        dataset_info['mean'] = _CIFAR_MEAN
        dataset_info['std'] = _CIFAR_STD
        dataset_info['img_dims'] = (3, 32, 32)
        dataset_info['num_labels'] = 100 if '100' in dataset and 'ten' not in dataset else 10
    elif 'svhn' in dataset:
        svhn_mean = [0.4379, 0.4440, 0.4729]
        svhn_std = [0.1980, 0.2010, 0.1970]
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
        dataset_info['mean'] = svhn_mean
        dataset_info['std'] = svhn_std
        dataset_info['img_dims'] = (3, 32, 32)
        dataset_info['num_labels'] = 10
    elif 'imagenet' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224,244), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop((224,244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_info['mean'] = [0.485, 0.456, 0.406]
        dataset_info['std'] = [0.229, 0.224, 0.225]
        dataset_info['img_dims'] = (3,224,244)
        dataset_info['num_labels'] = 1000
    elif 'smallwidth_imagenet' in dataset or 'flowers' in dataset or 'caltech' in dataset or 'pets' in dataset or 'aircraft' in dataset or 'cars' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((448,448), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(512, interpolation=Image.BICUBIC),
            transforms.CenterCrop((448,448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_info['mean'] = [0.485, 0.456, 0.406]
        dataset_info['std'] = [0.229, 0.224, 0.225]
        dataset_info['img_dims'] = (3,224,224)
        dataset_info['num_labels'] = 1000
    elif 'ohl_pipeline_imagenet'  in dataset:
        pre_transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.,1.,1.])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.,1.,1.])
        ])
        dataset_info['mean'] = [0.485, 0.456, 0.406]
        dataset_info['std'] = [1.,1.,1.]
        dataset_info['img_dims'] = (3,224,224)
        dataset_info['num_labels'] = 1000

    else:
        raise ValueError('dataset=%s' % dataset)

    logger.debug('augmentation: %s' % C.get()['aug'])
    if C.get()['aug'] == 'randaugment':
        assert not C.get()['randaug'].get('corrected_sample_space') and not C.get()['randaug'].get('google_augmentations')
        transform_train.transforms.insert(0, get_randaugment(n=C.get()['randaug']['N'], m=C.get()['randaug']['M'],
                                                             weights=C.get()['randaug'].get('weights',None), bs=C.get()['batch']))
    elif C.get()['aug'] in ['default', 'inception', 'inception320']:
        pass
    elif C.get()['aug'] in ['noaug',]:
        del transform_train.transforms[0]
        del transform_train.transforms[0]
    else:
        raise ValueError('not found augmentations. %s' % C.get()['aug'])

    transform_train.transforms.insert(0, pre_transform_train)
    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if 'preprocessor' in C.get():
        if 'imagenet' in dataset:
            print("Only using cropping/centering transforms on dataset, since preprocessor active.")
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                PILImageToHWCByteTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(256, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                PILImageToHWCByteTensor(),
            ])
        else:
            print("Not using any transforms in dataset, since preprocessor is active.")
            transform_train = PILImageToHWCByteTensor()
            transform_test = PILImageToHWCByteTensor()

    else:
        if not try_heavy:
            if dataset in ('cifar10', 'pre_transform_cifar10'):
                total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
                testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
            elif dataset in ('cifar100', 'pre_transform_cifar100'):
                total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
                testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
            elif dataset == 'svhncore':
                total_trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True,
                                                           transform=transform_train)
                testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
            elif dataset == 'svhn':
                trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
                extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_train)
                total_trainset = ConcatDataset([trainset, extraset])
                testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
            elif dataset == 'flowers':
                total_trainset = torchvision.datasets.Flowers102(root=dataroot, split='train', download=True, transform=transform_train)

                testset = torchvision.datasets.Flowers102(root=dataroot, split='val', download=True,
                                                    transform=transform_test)
            elif dataset == 'caltech':
                total_trainset = Caltech101(root=dataroot, split='train', download=True,
                                                                 transform=transform_train)

                testset = Caltech101(root=dataroot, split='test', download=True,
                                                          transform=transform_test)
            elif dataset == 'pets':
                total_trainset = torchvision.datasets.OxfordIIITPet(root=dataroot, split='trainval', download=True,
                                                                 transform=transform_train)

                testset = torchvision.datasets.OxfordIIITPet(root=dataroot, split='test', download=True,
                                                          transform=transform_test)
            elif dataset == 'aircraft':
                total_trainset = torchvision.datasets.FGVCAircraft(root=dataroot, split='train', download=True,
                                                                 transform=transform_train)

                testset = torchvision.datasets.FGVCAircraft(root=dataroot, split='test', download=True,
                                                          transform=transform_test)
            elif dataset == 'cars':
                total_trainset = torchvision.datasets.StanfordCars(root=dataroot, split='train', download=False,
                                                                 transform=transform_train)

                testset = torchvision.datasets.StanfordCars(root=dataroot, split='test', download=False,
                                                          transform=transform_test)
            elif dataset in ('imagenet', 'ohl_pipeline_imagenet', 'smallwidth_imagenet'):
                # Ignore archive only means to not to try to extract the files again, because they already are and the zip files
                # are not there no more
                total_trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'train'), transform=transform_train, )
                testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'val'),  transform=transform_test)

                # compatibility
                total_trainset.targets = [lb for _, lb in total_trainset.samples]
            else:
                raise ValueError('invalid dataset name=%s' % dataset)
        else:
            if dataset in ('cifar10', 'pre_transform_cifar10'):
                total_trainset = CIFAR10(root=dataroot, train=True, download=True, heavy_transform=heavy_transform_train,
                                                              transform=transform_train)
                testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
            elif dataset in ('cifar100', 'pre_transform_cifar100'):
                total_trainset = CIFAR100(root=dataroot, train=True, download=True, heavy_transform=heavy_transform_train,
                                                               transform=transform_train)
                testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
            elif dataset == 'svhncore':
                total_trainset = SVHN(root=dataroot, split='train', download=True,
                                                           transform=transform_train, heavy_transform=heavy_transform_train)
                testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
            elif dataset == 'svhn':
                trainset = SVHN(root=dataroot, split='train', download=True, transform=transform_train, heavy_transform=heavy_transform_train)
                extraset = SVHN(root=dataroot, split='extra', download=True, transform=transform_train, heavy_transform=heavy_transform_train)
                total_trainset = ConcatDataset([trainset, extraset])
                testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)

            elif dataset in ('imagenet', 'ohl_pipeline_imagenet', 'smallwidth_imagenet'):
                # Ignore archive only means to not to try to extract the files again, because they already are and the zip files
                # are not there no more
                total_trainset = ImageFolder(root=os.path.join(dataroot, 'train'), transform=transform_train, heavy_transform= heavy_transform_train)

                testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=transform_test)

                # compatibility
                total_trainset.targets = [lb for _, lb in total_trainset.samples]
            else:
                raise ValueError('invalid dataset name=%s' % dataset)
        # try heavy dataset


    if 'throwaway_share_of_ds' in C.get():
        assert 'val_step_trainloader_val_share' not in C.get()
        share = C.get()['throwaway_share_of_ds']['throwaway_share']
        train_subset_inds, rest_inds = stratified_split(total_trainset.targets if hasattr(total_trainset, 'targets') else list(total_trainset.labels),share)
        if C.get()['throwaway_share_of_ds']['use_throwaway_as_val']:
            testset = copy_and_replace_transform(Subset(total_trainset, rest_inds), transform_test)
        total_trainset = Subset(total_trainset, train_subset_inds)

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    if distributed:
        assert split == 0.0, "Split not supported for distributed training."
        if C.get().get('all_workers_use_the_same_batches', False):
            train_sampler = DistributedSampler(total_trainset, num_replicas=1, rank=0)
        else:
            train_sampler = DistributedSampler(total_trainset)
        test_sampler = None
        test_train_sampler = None # if these are specified, acc/loss computation is wrong for results.
        # while one has to say, that this setting leads to the test sets being computed seperately on each gpu which
        # might be considered not-very-climate-friendly
    else:
        test_sampler = None
        test_train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=train_sampler is None, num_workers= os.cpu_count()//8 if distributed else 8, pin_memory=True,
        sampler=train_sampler, drop_last=True, persistent_workers=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=0 if started_with_spawn else 8, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=16 if started_with_spawn else 8, pin_memory=True,
        drop_last=False, sampler=test_sampler, persistent_workers=True
    )
    # We use this 'hacky' solution s.t. we do not need to keep the dataset twice in memory.
    test_total_trainset = copy_and_replace_transform(total_trainset, transform_test)
    test_trainloader = torch.utils.data.DataLoader(
        test_total_trainset, batch_size=batch, shuffle=False, num_workers=0 if started_with_spawn else 8, pin_memory=True,
        drop_last=False, sampler=test_train_sampler
    )
    test_trainloader.denorm = lambda x: denormalize(x, dataset_info['mean'], dataset_info['std'])
    return train_sampler, trainloader, validloader, testloader, test_trainloader, dataset_info




class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

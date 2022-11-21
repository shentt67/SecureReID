# encoding: utf-8

import math
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import warnings
from collections.abc import Sequence
import torch
import numbers
from torch import Tensor
from typing import Tuple, List


def build_transforms(cfg):
    normalize_transform_aligned = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transforms = {}
    if cfg.TEST.PARTIAL_REID == 'off':
        transforms['train'] = Compose([
            Resize(cfg.INPUT.IMG_SIZE),
            RandomHorizontalFlip(p=cfg.INPUT.PROB),
            Pad(cfg.INPUT.PADDING),
            RandomCrop(cfg.INPUT.IMG_SIZE),
            ToTensor(),
            normalize_transform_aligned,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transforms['train'] = Compose([
            Resize(cfg.INPUT.IMG_SIZE),
            RandomHorizontalFlip(p=cfg.INPUT.PROB),
            RandomResizedCrop(size=256, scale=(0.5, 1.0), ratio=(1.0, 3.0)),
            Resize(cfg.INPUT.IMG_SIZE),
            ToTensor(),
            normalize_transform_aligned,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    transforms['eval'] = Compose([
        Resize(cfg.INPUT.IMG_SIZE),
        ToTensor(),
        normalize_transform_aligned
    ])

    return transforms


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_a, img_b):
        for t in self.transforms:
            img_a, img_b = t(img_a, img_b)
        return img_a, img_b


class Resize(object):
    def __init__(self, size, interpolation=TF.InterpolationMode.BILINEAR):
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = TF._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation

    def __call__(self, img_a, img_b):
        return TF.resize(img_a, self.size, self.interpolation), TF.resize(img_b, self.size, self.interpolation)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_a, img_b):
        if torch.rand(1) < self.p:
            return TF.hflip(img_a), TF.hflip(img_b)
        return img_a, img_b


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = TF._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        width, height = TF.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img_a, img_b):
        i, j, h, w = self.get_params(img_a, self.scale, self.ratio)
        return TF.resized_crop(img_a, i, j, h, w, self.size, self.interpolation), TF.resized_crop(img_b, i, j, h, w,
                                                                                                  self.size,
                                                                                                  self.interpolation)


class Pad(object):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        if not isinstance(padding, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate padding arg")

        if not isinstance(fill, (numbers.Number, str, tuple)):
            raise TypeError("Got inappropriate fill arg")

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if isinstance(padding, Sequence) and len(padding) not in [1, 2, 4]:
            raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img_a, img_b):
        return TF.pad(img_a, self.padding, self.fill, self.padding_mode), TF.pad(img_b, self.padding, self.fill,
                                                                                 self.padding_mode)


class RandomCrop(object):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        w, h = TF.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img_a, img_b):
        if self.padding is not None:
            img_a = TF.pad(img_a, self.padding, self.fill, self.padding_mode)
            img_b = TF.pad(img_b, self.padding, self.fill, self.padding_mode)

        width, height = TF.get_image_size(img_a)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img_a = TF.pad(img_a, padding, self.fill, self.padding_mode)
            img_b = TF.pad(img_b, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img_a = TF.pad(img_a, padding, self.fill, self.padding_mode)
            img_b = TF.pad(img_b, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img_a, self.size)

        return TF.crop(img_a, i, j, h, w), TF.crop(img_b, i, j, h, w)


class ToTensor():
    def __call__(self, img_a, img_b):
        return TF.to_tensor(img_a), TF.to_tensor(img_b)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img_a, img_b):

        if random.uniform(0, 1) >= self.probability:
            return img_a, img_b

        for attempt in range(100):
            area = img_a.size()[1] * img_a.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_a.size()[2] and h < img_a.size()[1]:
                x1 = random.randint(0, img_a.size()[1] - h)
                y1 = random.randint(0, img_a.size()[2] - w)
                if img_a.size()[0] == 3:
                    img_a[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img_a[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img_a[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    img_b[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img_b[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img_b[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img_a[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img_b[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img_a, img_b

        return img_a, img_b


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor_a: Tensor, tensor_b: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.normalize(tensor_a, self.mean, self.std, self.inplace), TF.normalize(tensor_b, self.mean, self.std,
                                                                                       self.inplace)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

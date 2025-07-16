import random
import numpy as np
import torch
from monai.transforms import Transform, Compose, MapTransform
from monai.transforms import Flip, Resize, GaussianSmooth, SpatialCrop, NormalizeIntensity, SpatialPad
from monai.utils import ensure_tuple

class RandHFlip(Transform):
    """
    随机水平翻转图像和掩码。
    """
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        img, mask = data['img'], data['mask']
        if random.random() < self.prob:
            flip = Flip(spatial_axis=1)  # 水平翻转（沿宽度轴）
            img = flip(img)
            mask = flip(mask)
        data['img'] = img
        data['mask'] = mask
        # print('flip', data['img'].shape, data['mask'].shape)
        return {'img': img, 'mask': mask}


class ZscoreNormWithOptionClip(Transform):
    """
    Z 分数归一化图像，支持可选的裁剪（基于固定值或百分位数）。
    Args:
        prob: float, 应用归一化的概率（默认 1.0）
        clip: bool, 是否基于固定值裁剪
        clip_percentile: bool, 是否基于百分位数裁剪
        clip_min_value: int, 固定值裁剪的最小值
        clip_max_value: int, 固定值裁剪的最大值
        clip_min_percentile: float, 百分位数裁剪的最小百分位（0 到 1）
        clip_max_percentile: float, 百分位数裁剪的最大百分位（0 到 1）
    """
    def __init__(self, clip=False, clip_percentile=False, 
                 clip_min_value=None, clip_max_value=None, 
                 clip_min_percentile=None, clip_max_percentile=None):
        super().__init__()
        self.clip = clip
        self.clip_percentile = clip_percentile
        self.clip_min_value = clip_min_value
        self.clip_max_value = clip_max_value
        self.clip_min_percentile = clip_min_percentile
        self.clip_max_percentile = clip_max_percentile

    def __call__(self, data):
        """
        对图像进行 Z 分数归一化，并可选地裁剪。
        Args:
            data: dict, 包含 'img' 和 'mask' 的字典，img 是 torch.Tensor (C, H, W) 或 (C, H, W, D)
        Returns:
            dict: 处理后的数据字典
        """
        img = data['img']
        
        # 处理无穷值
        img = torch.where(torch.isinf(img), torch.tensor(0.0, device=img.device), img)
        
        # 固定值裁剪
        if self.clip and self.clip_min_value is not None and self.clip_max_value is not None:
            img = torch.clamp(img, min=self.clip_min_value, max=self.clip_max_value)
        
        # 百分位数裁剪
        if self.clip_percentile and self.clip_min_percentile is not None and self.clip_max_percentile is not None:
            min_percentile_value = torch.quantile(img, self.clip_min_percentile)
            max_percentile_value = torch.quantile(img, self.clip_max_percentile)
            img = torch.clamp(img, min=min_percentile_value, max=max_percentile_value)
        
        # Z 分数归一化
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (max(std, 1e-8))
        
        data['img'] = img
        return data

class RandResize(Transform):
    """
    按随机比率调整图像和掩码大小。
    """
    def __init__(self, ratio_range=(0.8, 1.2)):
        super().__init__()
        self.ratio_range = ratio_range

    def __call__(self, data):
        img, mask = data['img'], data['mask']
        h, w = img.shape[-2], img.shape[-1]
        long_side = random.randint(int(max(h, w) * self.ratio_range[0]), int(max(h, w) * self.ratio_range[1]))

        if h > w:
            oh = long_side
            ow = int(1.0 * w * long_side / h + 0.5)
        else:
            ow = long_side
            oh = int(1.0 * h * long_side / w + 0.5)

        resize_img = Resize(spatial_size=(oh, ow), mode='bilinear', align_corners=False)
        resize_mask = Resize(spatial_size=(oh, ow), mode='nearest')

        data['img'] = resize_img(img)
        data['mask'] = resize_mask(mask)
        # print('resize', data['img'].shape, data['mask'].shape)
        return data

class RandBlur(Transform):
    """
    随机应用高斯模糊。
    """
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        img = data['img']
        if random.random() < self.prob:
            sigma = np.random.uniform(0.1, 2.0)
            blur = GaussianSmooth(sigma=sigma)
            img = blur(img)
        data['img'] = img
        return data

class RandCrop(Transform):
    """
    对图像和掩码进行填充（如果需要）并随机裁剪到指定大小。
    """
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, data):
        img, mask = data['img'], data['mask']
        h, w = img.shape[-2], img.shape[-1]

        padw = self.size - w if w < self.size else 0
        padh = self.size - h if h < self.size else 0

        if padw > 0 or padh > 0:
            pad_transform = SpatialPad(
                spatial_size=(self.size, self.size),
                method='symmetric',
                mode="constant",
                constant_values=255
            )
            mask_pad_transform = SpatialPad(
                spatial_size=(self.size, self.size),
                method='end',
                mode="constant",
                constant_values=255
            )
            img = pad_transform(img)
            mask = mask_pad_transform(mask)

        h, w = img.shape[-2], img.shape[-1]
        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)

        crop_transform = SpatialCrop(
            roi_start=[y, x],
            roi_end=[y + self.size, x + self.size]
        )
        data['img'] = crop_transform(img)
        data['mask'] = crop_transform(mask)
        # print('crop', data['img'].shape, data['mask'].shape)
        return data

class RandColorJitter(Transform):
    """
    随机调整图像的亮度、对比度、饱和度和色调。
    Args:
        prob: float, 应用变换的概率
        brightness: float, 亮度调整范围
        contrast: float, 对比度调整范围
        saturation: float, 饱和度调整范围
        hue: float, 色调调整范围
    """
    def __init__(self, prob=0.8, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25):
        super().__init__()
        self.prob = prob
        self.brightness = ensure_tuple(brightness)
        self.contrast = ensure_tuple(contrast)
        self.saturation = ensure_tuple(saturation)
        self.hue = ensure_tuple(hue)

    def __call__(self, data):
        img = data['img']
        if random.random() < self.prob:
            # 手动实现颜色抖动
            if len(self.brightness) == 1:
                brightness_factor = random.uniform(1 - self.brightness[0], 1 + self.brightness[0])
            else:
                brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            
            if len(self.contrast) == 1:
                contrast_factor = random.uniform(1 - self.contrast[0], 1 + self.contrast[0])
            else:
                contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            
            # if len(self.saturation) == 1:
            #     saturation_factor = random.uniform(1 - self.saturation[0], 1 + self.saturation[0])
            # else:
            #     saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            
            # if len(self.hue) == 1:
            #     hue_factor = random.uniform(-self.hue[0], self.hue[0])
            # else:
            #     hue_factor = random.uniform(self.hue[0], self.hue[1])
            
            # 应用亮度和对比度
            img = img * brightness_factor
            img = (img - img.mean()) * contrast_factor + img.mean()
            
        data['img'] = img
        return data

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
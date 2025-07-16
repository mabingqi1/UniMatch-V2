from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from monai.transforms import LoadImage, Compose
import zstandard as zstd
from .meidcal_transform import (ZscoreNormWithOptionClip,
                                RandHFlip,
                                RandCrop,
                                RandResize,
                                RandBlur,
                                RandColorJitter,
                                obtain_cutmix_box,
                                )

LABEL_DICT = {
    "肺": [44, 43, 112],
    "食管": [22],
    "心包": [48],
    "气管": [57, 56, 46, 69, 68],
    "肺内血管": [45, 47, 104],
    "脾脏": [17],
    "胰腺": [72],
    "胆囊": [99],
    "动脉": [1, 23, 24, 25, 61, 64, 60, 86, 80, 65, 88, 90, 124, 118, 108, 109],
    "静脉": [63, 75, 2, 26, 27, 62, 74, 79, 81, 85, 98, 96, 120, 125, 110],
    "骨": [54, 12, 11, 9, 51, 50, 6, 5, 4, 49, 40, 55, 52, 66, 10, 13, 3, 8, 7, 15, 14, 89, 34, 30, 29, 33, 28, 31, 36, 42, 35, 32, 41, 67, 73, 39, 38, 82, 37, 58, 87, 70, 76, 102, 105, 119, 106, 116]
  }

def remap_mask(mask, label_dict):
   
    # 初始化新掩码
    remapped_mask = torch.zeros_like(mask, dtype=torch.int32)
    # 映射标签
    for i, (cls_name, old_labels) in enumerate(label_dict.items()):
        for old_label in old_labels:
            remapped_mask[mask == old_label] = i
    
    return remapped_mask
class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        
        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1])))) 
        
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)

class SemiYHDataset(Dataset):
    def __init__(self, data_json, mode, size=None, nsample=None):
        self.json = data_json
        self.mode = mode
        self.size = size

        self.load_image = LoadImage(image_only=True, ensure_channel_first=True)
        if mode == 'train_u':
            with open(self.json, 'r') as f:
                data = json.load(f)
                self.ids = [item['img_path'] for item in data.get('data_list', [])]
        elif mode == 'train_l':
            with open(self.json, 'r') as f:
                data = json.load(f)
            self.ids = [item['img_path'] for item in data.get('data_list', [])]
            self.labels = [item['seg_map_path'] for item in data.get('data_list', [])]
            if nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.labels *= math.ceil(nsample / len(self.labels))
                self.ids, self.labels = self.ids[:nsample], self.labels[:nsample]
        else:
            with open(self.json, 'r') as f:
                data = json.load(f)
                self.ids = [item['img_path'] for item in data.get('data_list', [])]
                self.labels = [item['seg_map_path'] for item in data.get('data_list', [])]
                # val 模式 存储切片
                self.slice_indices = []
                # self.data_cache = []
                for idx, (img_id, label_id) in enumerate(zip(self.ids, self.labels)):
                    img = self.load_image(img_id)
                    label = self.load_image(label_id) 
                    num_slices = img.shape[-1] if img.ndim == 4 else 1
                    # self.data_cache.append((img, label))
                    self.slice_indices.extend([(idx, slice_idx) for slice_idx in range(num_slices)])

        self.weak_transforms = Compose([
            RandResize(ratio_range=(0.5, 2.0)),
            RandCrop(size=self.size),
            RandHFlip(prob=0.5),
        ])
        self.strong_transforms = Compose([
            RandResize(ratio_range=(0.5, 2.0)),
            RandCrop(size=self.size),
            RandHFlip(prob=0.5),
            RandColorJitter(prob=0.8, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
            RandBlur(prob=0.5),        
        ])
        self.norm_transforms = Compose([
            ZscoreNormWithOptionClip(clip=True, 
                                    clip_percentile=False, 
                                    clip_min_value=-1024, 
                                    clip_max_value=2048,
                                    clip_min_percentile=0.01,
                                    clip_max_percentile=0.99),
        ])

    def __len__(self):
        if self.mode == 'val':
            return len(self.slice_indices)
        return len(self.ids)

    def __getitem__(self, item):
        if self.mode == 'val':
            data_idx, slice_idx = self.slice_indices[item]
            img_3d = self.data_cache[data_idx][0]
            mask_3d = self.data_cache[data_idx][1]
            mask_3d = remap_mask(mask_3d, LABEL_DICT)
            img = img_3d[..., slice_idx]
            mask = mask_3d[..., slice_idx]

            data = {'img': img, 'mask': mask}
            data = self.norm_transforms(data)
            
            return data['img'], data['mask'].long()
        
        elif self.mode == 'train_l':
            id = self.ids[item]
            img = self.load_image(id)
            slice_idx = random.randint(0, img.shape[-1] - 1)
            img = img[..., slice_idx]      

            mask_path = self.labels[item]
            mask_3d = self.load_image(mask_path)
            mask_3d = remap_mask(mask_3d, LABEL_DICT)
            mask = mask_3d[..., slice_idx] 
            
            data = {'img': img, 'mask': mask}
            data_w = self.weak_transforms(data)
            data_w = self.norm_transforms(data_w)

            # import SimpleITK as sitk
            # im = sitk.GetImageFromArray(data_w['img'].numpy()[0])
            # sitk.WriteImage(im, 'data_w.nii.gz')
            # mask = sitk.GetImageFromArray(data_w['mask'].numpy()[0])
            # sitk.WriteImage(mask, 'data_mask.nii.gz')
            # raise
            return data_w['img'], data_w['mask'].long()
        else:
            id = self.ids[item]
            with open(id, 'rb') as f:
                    shape_str = f.readline().decode('utf-8').strip()
                    dtype_str = f.readline().decode('utf-8').strip()
                    compressed_data = f.read()
            dctx = zstd.ZstdDecompressor()
            decompressed_data = dctx.decompress(compressed_data)
            shape = eval(shape_str)
            img = np.frombuffer(decompressed_data, dtype=dtype_str).reshape(shape)
            slice_idx = random.randint(0, img.shape[-1] - 1)
            img = img[..., slice_idx]
            mask = torch.zeros((img.shape), dtype=torch.uint8)
            ignore_mask = torch.zeros_like(mask) # [H, W]

            data = {'img': img[None], 'mask': ignore_mask[None]}
            data_w = self.weak_transforms(data)
            data_w = self.norm_transforms(data_w)
            data_s1 = deepcopy(data)
            data_s1 = self.strong_transforms(data_s1)
            data_s1 = self.norm_transforms(data_s1)
            cutmix_box1 = obtain_cutmix_box(data_s1['img'].shape[-1], p=1)
            data_s2 = deepcopy(data)
            data_s2 = self.strong_transforms(data_s2)
            data_s2 = self.norm_transforms(data_s2)
            cutmix_box2 = obtain_cutmix_box(data_s2['img'].shape[-1], p=1)

            ignore_mask[mask == 255] = 255
            # import cv2
            # print(data_w['img'].max(),data_w['img'].min())
            # import SimpleITK as sitk
            # im = sitk.GetImageFromArray(data_w['img'].numpy()[0])
            # sitk.WriteImage(im, 'data_w.nii.gz')
            # im1 = sitk.GetImageFromArray(data_s1['img'].numpy()[0])
            # sitk.WriteImage(im1, 'data_s1.nii.gz')
            # im2 = sitk.GetImageFromArray(data_s2['img'].numpy()[0])
            # sitk.WriteImage(im2, 'data_s2.nii.gz')            
            # print(cutmix_box1.sum(), cutmix_box2.sum())
            # raise

            return data_w['img'], data_s1['img'], data_s2['img'], ignore_mask.long(), cutmix_box1, cutmix_box2
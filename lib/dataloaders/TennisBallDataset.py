# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/06/13 05:39
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import glob
import random

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import lib.utils as utils


class TennisBallDataset(Dataset):
    """
    网球关键点数据集
    """

    def __init__(self, opt, mode="train"):
        """
        初始化网球关键点数据集

        :param opt: 参数字典
        :param mode: train/val
        """
        self.opt = opt
        self.mode = mode
        self.root_dir = os.path.join(self.opt["dataset_path"], mode)

        # 定义数据增强字典
        self.transforms_dict = {
            "train": A.Compose([
                A.RandomResizedCrop(width=self.opt["resize_width"], height=self.opt["resize_height"], scale=self.opt["resize_scale"], p=1.0),
                A.Perspective(scale=self.opt["perspective_scale"], keep_size=True, p=self.opt["p"], pad_mode=cv2.BORDER_CONSTANT, pad_val=0),
                # A.ElasticTransform(p=self.opt["p"], border_mode=cv2.BORDER_CONSTANT, value=0, same_dxdy=True),
                A.ColorJitter(brightness=self.opt["brightness"], contrast=self.opt["contrast"], saturation=self.opt["saturation"], hue=self.opt["hue"]),
                A.GaussNoise(p=self.opt["p"]),
                A.MotionBlur(p=self.opt["p"]),
                # A.Spatter(p=self.opt["p"]),
                A.Normalize(mean=self.opt["normalize_mean"], std=self.opt["normalize_std"]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True)),
            "valid": A.Compose([
                A.Resize(height=self.opt["resize_height"], width=self.opt["resize_width"], p=1.0),
                A.Normalize(mean=self.opt["normalize_mean"], std=self.opt["normalize_std"]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True))
        }

        # 读取数据集图像列表
        self.images_path_list = sorted(glob.glob(os.path.join(self.root_dir, "images", "*.jpg")))
        self.labels_path_list = sorted(glob.glob(os.path.join(self.root_dir, "labels", "*.txt")))

        # 随机选择一部分
        if self.mode == "train":
            self.images_path_list = random.sample(self.images_path_list, 4096)
            self.labels_path_list = [image_path.replace("images", "labels").replace(".jpg", ".txt") for image_path in self.images_path_list]
        else:
            self.images_path_list = random.sample(self.images_path_list, 512)
            self.labels_path_list = [image_path.replace("images", "labels").replace(".jpg", ".txt") for image_path in self.images_path_list]

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, index):
        # 获取当前图像路径
        image_path = self.images_path_list[index]
        label_path = self.labels_path_list[index]
        # 打开图像
        image = cv2.imread(image_path)
        ori_w, ori_h = image.shape[1], image.shape[0]
        label = self._load_label(label_path)
        # 构造数据增强的关键点格式
        keypoints = label[:, 1:]
        class_labels = label[:, 0]
        # 数据增强
        transformed = self.transforms_dict[self.mode](image=image, keypoints=keypoints, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_keypoints = transformed['keypoints']
        transformed_class_labels = transformed['class_labels']
        return transformed_image, transformed_keypoints, transformed_class_labels, (ori_w, ori_h)

    @staticmethod
    def _load_label(label_path):
        # 读取标签文件
        labels = open(label_path, "r").readlines()
        labels = list(map(lambda x: x.strip().split("  "), labels))
        kps = []
        # 遍历每个标签
        for lbs in labels:
            # 将标签字符串转换为浮点数，并去掉类别索引
            lb = list(map(float, lbs))
            # 存储关键点信息
            kps.append([0, lb[0], lb[1]])
        return np.array(kps)

    @staticmethod
    def collate_fn(batch):
        # 初始化一个batch的数据结构
        images_batch = []
        keypoints_batch = []
        class_labels_batch = []
        sizes_batch = []
        # 遍历当前batch
        for (image, keypoints, class_labels, size) in batch:
            images_batch.append(image)
            keypoints_batch.append(keypoints)
            class_labels_batch.append(class_labels)
            sizes_batch.append(size)
        return torch.stack(images_batch, dim=0), keypoints_batch, class_labels_batch, sizes_batch

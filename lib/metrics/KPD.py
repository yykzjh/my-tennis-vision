# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/06/14 15:28
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class KPD(object):

    def __init__(self, classes=19, size=(640, 384)):
        self.classes = classes
        self.size = size
        # 计算图像对角线距离
        self.diagonal = np.linalg.norm(np.array(self.size))

    def __call__(self, pred, keypoints_batch, class_labels_batch):
        # 获取维度信息
        bs, c, h, w = pred.size()
        # 判断batch维度是否一致
        assert bs == len(keypoints_batch) and bs == len(class_labels_batch), "batch维度不一致"
        # 判断通道维度是否正确
        assert c == self.classes, "通道维度不正确"
        # 将网络输出热力图转化为对数概率分布
        heatmap = F.softmax(pred.view(bs, c, -1), dim=-1)
        heatmap = heatmap.view_as(pred).detach().cpu().numpy()
        # 依次计算关键点距离
        count = 0
        dist_sum = 0.0
        for b in range(bs):
            # 判断关键点数量是否一致
            assert len(keypoints_batch[b]) == len(class_labels_batch[b]), "关键点数量不一致"
            for j in range(len(keypoints_batch[b])):
                single_heatmap = heatmap[b, int(class_labels_batch[b][j])]
                pos = np.unravel_index(np.argmax(single_heatmap), single_heatmap.shape)
                dist = np.linalg.norm(np.array(pos) - np.array(keypoints_batch[b][j][::-1])) / self.diagonal
                dist_sum += dist
                count += 1
        return dist_sum, count









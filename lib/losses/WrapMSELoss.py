# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/06/14 14:34
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WrapMSELoss(nn.Module):

    def __init__(self, classes=19, size=(640, 384), weight=None):
        super(WrapMSELoss, self).__init__()
        self.classes = classes
        self.weight = weight
        self.size = size
        self.loss_function = nn.MSELoss()

    @staticmethod
    def _generate_gaussian_heatmap(size, center, sigma):
        """
        Generate a Gaussian heatmap.
        :param size: tuple of the heatmap size (height, width)
        :param center: tuple of the center position (x, y)
        :param sigma: standard deviation of the Gaussian
        :return: Gaussian heatmap
        """
        x = torch.arange(0, size[0]).float()
        y = torch.arange(0, size[1]).float()[:, None]
        x0, y0 = center
        g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def forward(self, pred, keypoints_batch, class_labels_batch):
        # 获取维度信息
        bs, c, h, w = pred.size()
        # 判断batch维度是否一致
        assert bs == len(keypoints_batch) and bs == len(class_labels_batch), "batch维度不一致"
        # 判断通道维度是否正确
        assert c == self.classes, "通道维度不正确"
        # 将网络输出热力图转化为对数概率分布
        heatmap = F.sigmoid(pred)
        # 生成真实热力图
        batch_real_heatmap = torch.zeros((bs, c, h, w), device=pred.device)
        for b in range(bs):
            # 判断关键点数量是否一致
            assert len(keypoints_batch[b]) == len(class_labels_batch[b]), "关键点数量不一致"
            for j in range(len(keypoints_batch[b])):
                batch_real_heatmap[b, int(class_labels_batch[b][j])] = self._generate_gaussian_heatmap(self.size, keypoints_batch[b][j], 2.0)
        # 计算loss
        loss = self.loss_function(heatmap, batch_real_heatmap)
        return loss

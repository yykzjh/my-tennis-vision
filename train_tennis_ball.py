# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/06/08 21:52
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import nni
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lib import utils, dataloaders, models, losses, metrics, trainers

# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "1",  # 选择可用的GPU编号

    "seed": 114514,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    # 数据增强概率
    "p": 0.5,

    # RandomResizedCrop
    "resize_width": 640,
    "resize_height": 384,
    "resize_scale": (0.8, 1.0),

    # Perspective
    "perspective_scale": (0.05, 0.1),

    # ColorJitter
    "brightness": (0.6, 1),
    "contrast": (0.6, 1),
    "saturation": (0.6, 1),
    "hue": (-0.5, 0.5),

    # 标准化均值
    "normalize_mean": (0.485, 0.456, 0.406),
    "normalize_std": (0.229, 0.224, 0.225),

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "TennisBall",  # 数据集名称

    "dataset_path": r"./datasets/tennis_ball",  # 数据集路径

    "batch_size": 4,  # batch_size大小

    "num_workers": 2,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "MobileNetV2",  # 模型名称，可选["ResNet50_Revise", "MobileNetV2"]

    "in_channels": 3,  # 模型最开始输入的通道数,即模态数

    "classes": 1,  # 模型最后输出的通道数,即类别总数

    "resume": None,  # 是否重启之前某个训练节点，继续训练;如果需要则指定.state文件路径

    "pretrain": r"./pretrain/mobilenet_v2.pth.tar",  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径

    # ——————————————————————————————————————————————    优化器     ——————————————————————————————————————————————————————

    "optimizer_name": "AdamW",  # 优化器名称，可选["SGD", "Adagrad", "RMSprop", "Adam", "AdamW", "Adamax", "Adadelta"]

    "learning_rate": 0.00001,  # 学习率

    "weight_decay": 0.000001,  # 权重衰减系数,即更新网络参数时的L2正则化项的系数

    "momentum": 0.9,  # 动量大小

    # ———————————————————————————————————————————    学习率调度器     —————————————————————————————————————————————————————

    "lr_scheduler_name": "CosineAnnealingWarmRestarts",  # 学习率调度器名称，可选["ExponentialLR", "StepLR", "MultiStepLR",
    # "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "OneCycleLR", "ReduceLROnPlateau"]

    "gamma": 0.97,  # 学习率衰减系数

    "step_size": 5,  # StepLR的学习率衰减步长

    "milestones": [3, 5, 9, 13, 15, 17, 18, 19],  # MultiStepLR的学习率衰减节点列表

    "T_max": 5,  # CosineAnnealingLR的半周期

    "T_0": 5,  # CosineAnnealingWarmRestarts的周期

    "T_mult": 2,  # CosineAnnealingWarmRestarts的周期放大倍数

    "mode": "min",  # ReduceLROnPlateau的衡量指标变化方向

    "patience": 5,  # ReduceLROnPlateau的衡量指标可以停止优化的最长epoch

    "factor": 0.1,  # ReduceLROnPlateau的衰减系数

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "metric_names": ["KPD"],  # 采用的评价指标，可选["KPD"]

    "loss_function_name": "KLDivLoss",  # 损失函数名称，可选["KLDivLoss", "BCEWithLogitsLoss", "MSELoss"]

    "class_weight": None,  # 各类别计算损失值的加权权重

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    # —————————————————————————————————————————————   训练相关参数   ——————————————————————————————————————————————————————

    "optimize_params": False,  # 程序是否处于优化参数的模型，不需要保存训练的权重和中间结果

    "run_dir": r"./runs",  # 运行时产生的各类文件的存储根目录

    "start_epoch": 1,  # 训练时的起始epoch
    "end_epoch": 100,  # 训练时的结束epoch

    "best_metric": 1.0,  # 保存检查点的初始条件

    "terminal_show_freq": 4000,  # 终端打印统计信息的频率,以step为单位

    "save_epoch_freq": 25,  # 每多少个epoch保存一次训练状态和模型参数
}


def main():
    if params["optimize_params"]:
        # 获得下一组搜索空间中的参数
        tuner_params = nni.get_next_parameter()
        # 更新参数
        params.update(tuner_params)

    # 设置可用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # 随机种子、卷积算法优化
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # 获取GPU设备
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("完成初始化配置")

    # 初始化数据加载器
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("完成初始化数据加载器")

    # for batch_idx, (input_tensor, keypoints_batch, class_labels_batch, sizes_batch) in enumerate(train_loader):
    #     print(input_tensor.size())
    #     print(keypoints_batch)
    #     print(class_labels_batch)
    #     print(sizes_batch)
    #
    #     for j in range(32):
    #         image_np = np.ascontiguousarray(input_tensor[j].permute(1, 2, 0).numpy())
    #         image_np = (image_np - image_np.min()) * 255 / (image_np.max() - image_np.min())
    #         image_np = image_np.astype(np.uint8)
    #         for point in keypoints_batch[j]:
    #             cv2.circle(image_np, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    #         image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    #         # 展示图像
    #         plt.imshow(image_np)
    #         plt.axis('off')  # 隐藏坐标轴
    #         plt.show()

    # 初始化模型、优化器和学习率调整器
    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    print("完成初始化模型:{}、优化器:{}和学习率调整器:{}".format(params["model_name"], params["optimizer_name"], params["lr_scheduler_name"]))

    # 初始化损失函数
    loss_function = losses.get_loss_function(params)
    print("完成初始化损失函数")

    # 初始化各评价指标
    metric_function = metrics.get_metric(params)
    print("完成初始化评价指标")

    # 创建训练执行目录和文件
    if not params["optimize_params"]:
        if params["resume"] is None:
            params["execute_dir"] = os.path.join(params["run_dir"],
                                                 utils.datestr() +
                                                 "_" + params["model_name"] +
                                                 "_" + params["dataset_name"])
        else:
            params["execute_dir"] = os.path.dirname(os.path.dirname(params["resume"]))
        params["checkpoint_dir"] = os.path.join(params["execute_dir"], "checkpoints")
        params["tensorboard_dir"] = os.path.join(params["execute_dir"], "board")
        params["log_txt_path"] = os.path.join(params["execute_dir"], "log.txt")
        if params["resume"] is None:
            utils.make_dirs(params["checkpoint_dir"])
            utils.make_dirs(params["tensorboard_dir"])

    # 初始化训练器
    trainer = trainers.CourtTrainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric_function)

    # 如果需要继续训练或者加载预训练权重
    if (params["resume"] is not None) or (params["pretrain"] is not None):
        trainer.load()

    # 开始训练
    trainer.training()


if __name__ == '__main__':
    main()

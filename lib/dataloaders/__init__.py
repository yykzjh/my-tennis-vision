# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/19 11:40
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from torch.utils.data import DataLoader

from .CourtDataset import CourtDataset
from .TennisBallDataset import TennisBallDataset


def get_dataloader(opt):
    """
    获取数据加载器
    Args:
        opt: 参数字典
    Returns:
    """
    if opt["dataset_name"] == "Court":
        # 初始化数据集
        train_set = CourtDataset(opt, mode="train")
        valid_set = CourtDataset(opt, mode="valid")

        # 初始化数据加载器
        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True, collate_fn=CourtDataset.collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True, collate_fn=CourtDataset.collate_fn)

    if opt["dataset_name"] == "TennisBall":
        # 初始化数据集
        train_set = TennisBallDataset(opt, mode="train")
        valid_set = TennisBallDataset(opt, mode="valid")

        # 初始化数据加载器
        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True, collate_fn=CourtDataset.collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True, collate_fn=CourtDataset.collate_fn)

    else:
        raise RuntimeError(f"{opt['dataset_name']}是不支持的数据集！")

    return train_loader, valid_loader

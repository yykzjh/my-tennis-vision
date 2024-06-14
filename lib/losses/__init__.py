# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/19 11:40
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import torch.nn as nn

from .WrapKLDivLoss import WrapKLDivLoss
from .WrapMSELoss import WrapMSELoss
from .WrapBCEWithLogitsLoss import WrapBCEWithLogitsLoss


def get_loss_function(opt):
    if opt["loss_function_name"] == "KLDivLoss":
        loss_function = WrapKLDivLoss(classes=opt["classes"], size=(opt["resize_width"], opt["resize_height"]))

    elif opt["loss_function_name"] == "MSELoss":
        loss_function = WrapMSELoss(classes=opt["classes"], size=(opt["resize_width"], opt["resize_height"]))
    elif opt["loss_function_name"] == "BCEWithLogitsLoss":
        loss_function = WrapBCEWithLogitsLoss(classes=opt["classes"], size=(opt["resize_width"], opt["resize_height"]))
    else:
        raise RuntimeError(f"{opt['loss_function_name']}是不支持的损失函数！")

    return loss_function

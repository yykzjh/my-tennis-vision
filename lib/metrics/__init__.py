# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/19 11:40
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from .KPD import KPD


def get_metric(opt):
    # 初始化评价指标对象列表
    metrics = []
    for metric_name in opt["metric_names"]:
        if metric_name == "KPD":
            metrics.append(KPD(classes=opt["classes"], size=(opt["resize_width"], opt["resize_height"])))

        else:
            raise Exception(f"{metric_name}是不支持的评价指标！")

    return metrics

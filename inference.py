# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/06/14 22:03
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib import utils, models


class CourtDetector(object):

    def __init__(self, model_name="ResNet50_Revise", pretrain_path=r"./pretrain/best_ResNet50_Revise.pth", th=0.1):
        self.classes = 19
        self.model_name = model_name
        self.pretrain_path = pretrain_path
        self.th = th
        self.opt = {
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            "in_channels": 3,
            "classes": self.classes,
            "resize_height": 384,
            "resize_width": 640,
            "normalize_mean": (0.485, 0.456, 0.406),
            "normalize_std": (0.229, 0.224, 0.225),
            "model_name": self.model_name
        }
        self.transforms = A.Compose([
            A.Resize(height=self.opt["resize_height"], width=self.opt["resize_width"], p=1.0),
            A.Normalize(mean=self.opt["normalize_mean"], std=self.opt["normalize_std"]),
            ToTensorV2()
        ])
        pretrain_state_dict = torch.load(self.pretrain_path, map_location=self.opt["device"])
        self.model = models.get_model(self.opt)
        self.model.load_state_dict(pretrain_state_dict, strict=True)

    def predict(self, frame):
        self.ori_h, self.ori_w = frame.shape[0], frame.shape[1]
        input_tensor = self._pre_process(frame).to(self.opt["device"])
        output = self.model(input_tensor)
        kps = self._post_process(output)
        return kps

    def _pre_process(self, frame):
        transformed = self.transforms(image=frame)
        transformed_image = torch.unsqueeze(transformed['image'], dim=0)
        return transformed_image

    def _post_process(self, pred):
        # 获取维度信息
        pred = torch.squeeze(pred)
        c, h, w = pred.size()
        # 将网络输出热力图转化为对数概率分布
        heatmap = F.softmax(pred.view(c, -1), dim=-1)
        heatmap = heatmap.view_as(pred).detach().cpu().numpy()
        # 依次计算关键点位置
        kps = []
        for j in range(self.classes):
            single_heatmap = heatmap[j]
            pos = np.unravel_index(np.argmax(single_heatmap), single_heatmap.shape)
            max_val = single_heatmap[pos]
            if max_val > self.th:
                kps.append((int(j), int(pos[1] * (self.ori_w / self.opt["resize_width"])), int(pos[0] * (self.ori_h / self.opt["resize_height"]))))
        return kps


def inference_video_court(video_path=None, model_name="ResNet50_Revise", pretrain_path=r"./pretrain/best_ResNet50_Revise.pth", th=0.1):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)
    print(fps, total_frame_length, w, h)
    # 初始化场地线检测器
    court_detector = CourtDetector(model_name=model_name, pretrain_path=pretrain_path, th=th)

    # 初始化一些数据
    frame_ind = 0
    new_frames = []
    time_sum = 0.0
    time_cnt = 0
    # 遍历所有视频帧
    while True:
        # 读取一帧
        ret, frame = video.read()
        frame_ind += 1  # 帧数累计

        # 成功读取帧
        if ret:
            t1 = time.time()
            kps = court_detector.predict(frame.copy())
            t2 = time.time()
            time_sum += (t2 - t1) * 1000
            time_cnt += 1

            # 在当前帧画出球场关键点
            new_frame = frame.copy()
            for kp in kps:
                cv2.circle(new_frame, (kp[1], kp[2]), 5, (0, 0, 255), -1)
            # 将处理后的一帧添加到列表
            new_frames.append(new_frame)
        else:  # 视频结尾跳出循环
            break
    # 释放打开的视频
    video.release()

    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(video_path.replace("input", "output"), fourcc, fps, (w, h))
    # 遍历写入视频
    for frame in new_frames:
        output_video.write(frame)
    # 释放输出的视频
    output_video.release()

    # 输出平均用时
    print("跟踪网球场地线的平均用时为：{:.8f}".format(time_sum / time_cnt))



def inference_images_court(images_dir=None, model_name="ResNet50_Revise", pretrain_path=r"./pretrain/best_ResNet50_Revise.pth", th=0.1):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 初始化场地线检测器
    court_detector = CourtDetector(model_name=model_name, pretrain_path=pretrain_path, th=th)

    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        frame = cv2.imread(image_path)
        kps = court_detector.predict(frame.copy())
        # 在当前帧画出球场关键点
        new_frame = frame.copy()
        for kp in kps:
            cv2.circle(new_frame, (kp[1], kp[2]), 5, (0, 0, 255), -1)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(new_frame)
        plt.show()


if __name__ == '__main__':
    # 推理视频中的球场关键点
    inference_video_court(video_path=r"./static/video/video_input1.mp4",
                          model_name="MobileNetV2",
                          pretrain_path=r"./pretrain/best_MobileNetV2_BCEWithLogitsLoss_0.001557.pth",
                          th=0.01)

    # 推理图像集的球场关键点
    # inference_images_court(images_dir=r"./datasets/court/images",
    #                        model_name="MobileNetV2",
    #                        pretrain_path=r"./pretrain/best_MobileNetV2.pth",
    #                        th=0.01)

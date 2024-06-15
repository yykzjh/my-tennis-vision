# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/06/15 01:52
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import math
import threading
from tqdm import tqdm
import shutil

import torch

from lib import utils, models


class TennisBallDatasetTransformer(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 初始化各种目录
        self.raw_dir = os.path.join(root_dir, "raw")
        self.raw_videos_dir = os.path.join(self.raw_dir, "videos")
        self.raw_labels_dir = os.path.join(self.raw_dir, "labels")
        self.images_dir = os.path.join(root_dir, "images")
        self.labels_dir = os.path.join(root_dir, "labels")
        if os.path.exists(self.images_dir):
            shutil.rmtree(self.images_dir)
        if os.path.exists(self.labels_dir):
            shutil.rmtree(self.labels_dir)
        os.makedirs(self.images_dir)
        os.makedirs(self.labels_dir)
        # 多线程相关参数
        self.max_thread_num = 12

    def _transform(self, video_basenames):
        for video_basename in video_basenames:
            # 加载视频
            video_path = os.path.join(self.raw_videos_dir, video_basename + ".mp4")
            video_cap = cv2.VideoCapture(video_path)
            # 遍历当前视频的所有标注帧
            raw_labels_video_dir = os.path.join(self.raw_labels_dir, video_basename)
            for label_txt_name in tqdm(os.listdir(raw_labels_video_dir)):
                # 获取标注文件路径
                label_txt_path = os.path.join(raw_labels_video_dir, label_txt_name)
                # 读取文件内容
                line_content_str = open(label_txt_path, "r").readline().strip()
                str_nums_list = line_content_str.split(",")
                if str_nums_list[0] == "0":
                    continue
                # 获取关键点位置信息
                x, y = int(str_nums_list[1]), int(str_nums_list[2])
                # 获取帧号
                label_txt_basename, _ = os.path.splitext(label_txt_name)
                spans = label_txt_basename.split("_")
                frame_id = int(spans[1])
                # 获取新的txt标注文件名
                new_label_txt_basename = "{}_{:07d}".format(video_basename, frame_id)
                new_label_txt_name = new_label_txt_basename + ".txt"
                # 获取新的txt标注文件路径
                new_label_txt_path = os.path.join(self.labels_dir, new_label_txt_name)
                # 打开新的标注文件并写入关键点坐标
                with open(new_label_txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"{x}  {y}")
                # 读取当前帧图像
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = video_cap.read()
                # 获取图像帧的存储路径
                image_path = os.path.join(self.images_dir, new_label_txt_basename + ".jpg")
                # 存储图像帧
                cv2.imwrite(image_path, frame)
            video_cap.release()

    def run(self):
        # 标签子目录列表
        raw_labels_video_basename_list = list(os.listdir(self.raw_labels_dir))
        # 每个线程分配子目录数
        assign_num = math.ceil(len(raw_labels_video_basename_list) / self.max_thread_num)
        # 分配多线程
        threads = []
        for i in range(0, len(raw_labels_video_basename_list), assign_num):
            t = threading.Thread(target=self._transform, args=(raw_labels_video_basename_list[i: i + assign_num],))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()


def export_court_model(model_name=None, weight_path=None):
    # 一些参数
    opt = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "in_channels": 3,
        "classes": 19,
        "resize_height": 384,
        "resize_width": 640,
        "normalize_mean": (0.485, 0.456, 0.406),
        "normalize_std": (0.229, 0.224, 0.225),
        "model_name": model_name
    }
    # 加载模型和权重
    pretrain_state_dict = torch.load(weight_path, map_location=opt["device"])
    model = models.get_model(opt)
    model.load_state_dict(pretrain_state_dict, strict=True)
    model.eval()
    # 获取导出onnx目录
    onnx_dir = os.path.join(os.path.dirname(weight_path), "onnx")
    # 获取文件名
    pth_filename = os.path.basename(weight_path)
    pth_basename, _ = os.path.splitext(pth_filename)
    # 获取导出onnx文件目录
    onnx_path = os.path.join(onnx_dir, pth_basename + ".onnx")
    print(onnx_path)

    dummy_input = torch.randn(1, 3, opt["resize_height"], opt["resize_width"]).to(opt["device"])
    model(dummy_input)
    im = torch.zeros(1, 3, opt["resize_height"], opt["resize_width"]).to(opt["device"])
    torch.onnx.export(model, im,
                      onnx_path,
                      verbose=False,
                      opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      )


if __name__ == '__main__':
    # # 转换原生的网球关键点数据集
    # tennis_ball_dataset_transformer = TennisBallDatasetTransformer(r"./datasets/tennis_ball")
    # tennis_ball_dataset_transformer.run()

    # 将球场关键点模型导出为onnx
    export_court_model(model_name="MobileNetV2", weight_path=r"./pretrain/court/best_MobileNetV2_BCEWithLogitsLoss_0.001557.pth")

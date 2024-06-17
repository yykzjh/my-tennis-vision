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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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


def split_tennis_ball_dataset(root_dir):
    # 初始化目录
    ori_images_dir = os.path.join(root_dir, "images")
    ori_labels_dir = os.path.join(root_dir, "labels")
    train_dir = os.path.join(root_dir, "train")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    valid_dir = os.path.join(root_dir, "valid")
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)
    train_images_dir = os.path.join(train_dir, "images")
    os.makedirs(train_images_dir)
    train_labels_dir = os.path.join(train_dir, "labels")
    os.makedirs(train_labels_dir)
    valid_images_dir = os.path.join(valid_dir, "images")
    os.makedirs(valid_images_dir)
    valid_labels_dir = os.path.join(valid_dir, "labels")
    os.makedirs(valid_labels_dir)
    # 划分训练集和验证集
    ori_images_path_list = np.array(os.listdir(ori_images_dir))
    train_images_path_list, valid_images_path_list = train_test_split(ori_images_path_list, test_size=0.2, random_state=42)
    # 遍历训练集，复制图像和标注文件
    for ori_image_name in tqdm(train_images_path_list):
        # 获取原始图像路径
        ori_image_path = os.path.join(ori_images_dir, ori_image_name)
        # 获取图像新存储路径
        new_image_path = os.path.join(train_images_dir, ori_image_name)
        # 获取原始标注文件名
        ori_label_name = ori_image_name.replace(".jpg", ".txt")
        # 获取原始标注文件路径
        ori_label_path = os.path.join(ori_labels_dir, ori_label_name)
        # 获取标注文件新存储路径
        new_label_path = os.path.join(train_labels_dir, ori_label_name)
        # 复制存储图像和标注文件
        shutil.copy(ori_image_path, new_image_path)
        shutil.copy(ori_label_path, new_label_path)
    # 遍历验证集，复制图像和标注文件
    for ori_image_name in tqdm(valid_images_path_list):
        # 获取原始图像路径
        ori_image_path = os.path.join(ori_images_dir, ori_image_name)
        # 获取图像新存储路径
        new_image_path = os.path.join(valid_images_dir, ori_image_name)
        # 获取原始标注文件名
        ori_label_name = ori_image_name.replace(".jpg", ".txt")
        # 获取原始标注文件路径
        ori_label_path = os.path.join(ori_labels_dir, ori_label_name)
        # 获取标注文件新存储路径
        new_label_path = os.path.join(valid_labels_dir, ori_label_name)
        # 复制存储图像和标注文件
        shutil.copy(ori_image_path, new_image_path)
        shutil.copy(ori_label_path, new_label_path)



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
                      opset_version=11,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      )


class CourtReference(object):
    """
    标准球场
    """

    def __init__(self):
        self.single_court_width = 823
        self.single_court_height = 2377
        self.double_court_width = 1097
        self.double_court_height = 2377
        self.accessibility_area_width = 366
        self.accessibility_area_height = 640.5
        self.tee_area_width = 411.5
        self.tee_area_height = 640
        self.backfield_height = 548.5
        self.sideline_width = 137
        self.whole_width = 1829
        self.whole_height = 3658

        self.court_conf_keypoints = {
            1: (-548.5, 1188.5),
            2: (-411.5, 1188.5),
            3: (411.5, 1188.5),
            4: (548.5, 1188.5),
            5: (-411.5, 640),
            6: (0, 640),
            7: (411.5, 640),
            8: (-548.5, 0),
            9: (-411.5, 0),
            10: (0, 0),
            11: (411.5, 0),
            12: (548.5, 0),
            13: (-411.5, -640),
            14: (0, -640),
            15: (411.5, -640),
            16: (-548.5, -1188.5),
            17: (-411.5, -1188.5),
            18: (411.5, -1188.5),
            19: (548.5, -1188.5)
        }

        self.origin_coord = (-(self.whole_width / 2 - 0.5), self.whole_height / 2 - 0.5)

        self.center_pos = (-self.origin_coord[0], self.origin_coord[1])

        self.court_conf_keypoints_position = {
            key: [pos[0] - self.origin_coord[0], -(pos[1] - self.origin_coord[1])]
            for key, pos in self.court_conf_keypoints.items()
        }

        # conference_court_image_path = os.path.join(os.path.dirname(__file__), "lib/resource/court_reference.jpg")
        # self.court = cv2.cvtColor(cv2.imread(conference_court_image_path), cv2.COLOR_BGR2GRAY)

    def build_court_reference_image(self):
        kps_dict = {
            k: v
            for k, v in self.court_conf_keypoints_position.items()
        }

        for key in kps_dict.keys():
            if not kps_dict[key][0].is_integer():
                if kps_dict[key][0] > self.center_pos[0]:
                    kps_dict[key][0] = math.floor(kps_dict[key][0])
                elif kps_dict[key][0] < self.center_pos[0]:
                    kps_dict[key][0] = math.ceil(kps_dict[key][0])
            if not kps_dict[key][1].is_integer():
                if kps_dict[key][1] > self.center_pos[1]:
                    kps_dict[key][1] = math.floor(kps_dict[key][1])
                elif kps_dict[key][1] < self.center_pos[1]:
                    kps_dict[key][1] = math.ceil(kps_dict[key][1])
            kps_dict[key] = (int(kps_dict[key][0]), int(kps_dict[key][1]))

        court = np.zeros((self.whole_height, self.whole_width), dtype=np.uint8)
        cv2.line(court, kps_dict[1], kps_dict[16], 1, 1)
        cv2.line(court, kps_dict[2], kps_dict[17], 1, 1)
        cv2.line(court, kps_dict[6], kps_dict[14], 1, 1)
        cv2.line(court, kps_dict[3], kps_dict[18], 1, 1)
        cv2.line(court, kps_dict[4], kps_dict[19], 1, 1)
        cv2.line(court, kps_dict[1], kps_dict[4], 1, 1)
        cv2.line(court, kps_dict[5], kps_dict[7], 1, 1)
        cv2.line(court, kps_dict[8], kps_dict[12], 1, 1)
        cv2.line(court, kps_dict[13], kps_dict[15], 1, 1)
        cv2.line(court, kps_dict[16], kps_dict[19], 1, 1)

        plt.imsave(r"./static/image/court_reference.jpg", court, cmap='gray')
        self.court = court
        return court




if __name__ == '__main__':
    # # 转换原生的网球关键点数据集
    # tennis_ball_dataset_transformer = TennisBallDatasetTransformer(r"./datasets/tennis_ball")
    # tennis_ball_dataset_transformer.run()

    # 将网球关键点数据集划分为训练集和验证集
    split_tennis_ball_dataset(r"./datasets/tennis_ball")

    # 将球场关键点模型导出为onnx
    # export_court_model(model_name="MobileNetV2", weight_path=r"./pretrain/court/best_MobileNetV2_BCEWithLogitsLoss_0.001557.pth")

    # 生成标准网球场图像
    # court_reference = CourtReference()
    # court_reference.build_court_reference_image()


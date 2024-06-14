# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/19 19:10
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import onnx
import onnxruntime

from lib import utils


def trans_onnx_to_trt():
    # 定义要转化的文件名称
    weight_names = ["yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    weight_names = ["yolov8n"]
    flop_precisions = ["FP32", "FP16"]
    for weight_name in weight_names:
        for fp in flop_precisions:
            # 初始化一些参数
            img_size = (640, 640)
            yolov8_onnx_path = os.path.join(r"./pretrain/yolov8_onnx_static", weight_name + ".onnx")
            yolov8_trt_path = os.path.join(r"./pretrain/yolov8_trt_static", weight_name + "_" + fp + ".trt")
            # 初始化TRT类
            trt_util = utils.TRTUtil(yolov8_onnx_path, yolov8_trt_path, fp, img_size)
            # 将onnx模型转换为序列化的TRT模型
            engine = trt_util.onnx_to_TRT_model(yolov8_onnx_path, yolov8_trt_path, fp)
            assert engine, "从 onnx 文件中转换的 engine 为 None ! "


def test_trt_yolov8():
    # 初始化一些参数
    fp = "FP16"
    yolov8_onnx_path = os.path.join(r"./pretrain/yolov8_onnx_static", "yolov8n.onnx")
    yolov8_trt_path = os.path.join(r"./pretrain/yolov8_trt_static", "yolov8n_" + fp + ".trt")
    image_path = r"./static/image/frame_20.jpg"
    img_size = (640, 640)
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # 初始化TRT类
    trt_util = utils.TRTUtil(yolov8_onnx_path, yolov8_trt_path, fp, img_size)
    trt_util.init_model()

    input_image, ratio = trt_util.pre_process(image_np)
    engine_infer_output = trt_util.inference(input_image)
    postprocess_output = trt_util.post_process(engine_infer_output, 0.25, 0.65, image_np, ratio)

    PERSON_LABEL = 0
    person_min_score = 0.2
    for box, label, score in zip(postprocess_output[0]['boxes'][:], postprocess_output[0]['labels'], postprocess_output[0]['scores']):
        if label == PERSON_LABEL and score > person_min_score:
            box_image = image_np.copy()
            cv2.rectangle(box_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
            plt.imshow(box_image)
            plt.show()

    print(postprocess_output)


def test_detect_court(video_path, quantize=False, precision="FP32"):
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)
    print(fps, total_frame_length, w, h)
    # 初始化场地线检测器
    court_detector = utils.CourtDetector()

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

        if frame_ind % 20 == 0:
            cv2.imwrite(r"./static/image/frame_{}.jpg".format(frame_ind), frame)

        # 成功读取帧
        if ret:
            # 检测第一帧的场地线
            if frame_ind == 1:
                lines = court_detector.detect(frame)
            else:  # 其他帧跟踪场地线
                t1 = time.time()
                lines = court_detector.track_court(frame)
                t2 = time.time()
                time_sum += (t2 - t1) * 1000
                time_cnt += 1

            # 在当前帧画出场地线
            for i in range(0, len(lines), 4):
                x1, y1, x2, y2 = lines[i:i + 4]
                new_frame = cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
            # 缩放图像尺寸
            new_frame = cv2.resize(new_frame, (w, h))
            # 将处理后的一帧添加到列表
            new_frames.append(new_frame)
        else:  # 视频结尾跳出循环
            break
    # 释放打开的视频
    video.release()

    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(r"./static/video/Output_Court.mp4", fourcc, fps, (w, h))
    # 遍历写入视频
    for frame in new_frames:
        output_video.write(frame)
    # 释放输出的视频
    output_video.release()

    # 输出平均用时
    print("跟踪网球场地线的平均用时为：{:.8f}".format(time_sum / time_cnt))


def test_detect_player(video_path, model_name="faster_rcnn", quantize=False, fp="FP32"):
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)
    print(fps, total_frame_length, w, h)
    # 初始化球员检测器
    player_detector = utils.PlayerDetector(utils.get_dtype(), model_name, quantize, fp)
    # 初始化场地线检测器
    court_detector = utils.CourtDetector()

    # 初始化一些数据
    frame_ind = 0
    new_frames = []
    time_sum_1 = 0.0
    time_cnt_1 = 0
    time_sum_2 = 0.0
    time_cnt_2 = 0
    time_sum_3 = 0.0
    time_cnt_3 = 0
    time_sum_4 = 0.0
    time_cnt_4 = 0
    # 遍历所有视频帧
    while True:
        # 读取一帧
        ret, frame = video.read()
        frame_ind += 1  # 帧数累计
        # 成功读取帧
        if ret:
            # 检测第一帧的场地线
            if frame_ind == 1:
                lines = court_detector.detect(frame)
            else:  # 其他帧跟踪场地线
                lines = court_detector.track_court(frame)

            # 检测下半场球员
            t1 = time.time()
            _, t3 = player_detector.detect_player_1(frame, court_detector)
            t2 = time.time()
            time_sum_1 += (t2 - t1) * 1000
            time_cnt_1 += 1
            time_sum_2 += t3
            time_cnt_2 += 1

            # 检测上半场球员
            t1 = time.time()
            _, t4 = player_detector.detect_top_persons(frame, court_detector, frame_ind)
            t2 = time.time()
            time_sum_3 += (t2 - t1) * 1000
            time_cnt_3 += 1
            time_sum_4 += t4
            time_cnt_4 += 1

            # 在当前帧画出场地线
            for i in range(0, len(lines), 4):
                x1, y1, x2, y2 = lines[i:i + 4]
                new_frame = cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
            # 缩放图像尺寸
            new_frame = cv2.resize(new_frame, (w, h))
            # 将处理后的一帧添加到列表
            new_frames.append(new_frame)
        else:  # 视频结尾跳出循环
            break
    # 释放打开的视频
    video.release()

    # 获得上半场球员的目标框
    t1 = time.time()
    player_detector.find_player_2_box()
    t2 = time.time()
    print("获得上半场球员的目标框用时为：{:.8f}".format((t2 - t1) * 1000))
    player1_boxes = player_detector.player_1_boxes
    player2_boxes = player_detector.player_2_boxes

    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(r"./static/video/Output_Player.mp4", fourcc, fps, (w, h))
    # 遍历写入视频
    for i, frame in enumerate(new_frames):
        frame = utils.mark_player_box(frame, player1_boxes, i - 1)
        frame = utils.mark_player_box(frame, player2_boxes, i - 1)
        output_video.write(frame)
    # 释放输出的视频
    output_video.release()

    # 输出平均用时
    print("检测下半场球员的平均用时为：{:.8f}".format(time_sum_1 / time_cnt_1))
    print("检测下半场球员中的目标检测的平均用时为：{:.8f}".format(time_sum_2 / time_cnt_2))
    print("检测上半场球员的平均用时为：{:.8f}".format(time_sum_3 / time_cnt_3))
    print("检测上半场球员中的目标检测的平均用时为：{:.8f}".format(time_sum_4 / time_cnt_4))


if __name__ == '__main__':
    # 批量将yolov8的onnx转成trt
    # trans_onnx_to_trt()

    # 测试量化模型
    # test_trt_yolov8()

    # 测试检测场地线
    # test_detect_court(r"./static/video/video_input3.mp4", quantize=False, precision="FP32")

    # 测试检测球员
    test_detect_player(r"./static/video/video_input3.mp4", model_name="yolov8x", quantize=True, fp="FP32")

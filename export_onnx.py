# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/21 22:11
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
# from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# import tf2onnx
import onnx
import onnxruntime
# import tensorflow as tf

from lib.models.tracknet import trackNet
from lib.models.TrackNetV3 import TrackNetV2


def export_faster_rcnn():
    # 一些参数
    img_size = (640, 640)
    faster_rcnn_onnx_path = r"./pretrain/faster_rcnn.onnx"

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to("cuda")
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
    model.eval()
    model(dummy_input)
    im = torch.zeros(1, 3, img_size[0], img_size[1]).to("cuda")
    torch.onnx.export(model, im,
                      faster_rcnn_onnx_path,
                      verbose=False,
                      opset_version=11,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}}
                      )


# def h5_to_pb():
#     # 图像读取和预处理
#     def load_preprocess_image(path):
#         width, height = 640, 360  # 初始化尺寸
#         img = cv2.imread(path)  # 读取图片
#         img = cv2.resize(img, (width, height))
#         img = img.astype("float32")
#         X = np.rollaxis(img, 2, 0)
#         return np.array([X])
#
#     # 加载模型
#     n_classes = 256
#     width, height = 640, 360
#     save_weights_path = r"./pretrain/tracknet_h5/tracknet.h5"
#     model = trackNet(n_classes, input_height=height, input_width=width)
#     model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#     model.load_weights(save_weights_path)
#
#     # 推理h5模型
#     preds = model.predict(load_preprocess_image(r"./static/image/frame_20.jpg"))
#
#     # 定义模型转onnx的参数
#     spec = (tf.TensorSpec((1, 3, height, width), tf.float32, name="input"),)  # 输入签名参数，(None, 128, 128, 3)决定输入的size
#     output_path = r"./pretrain/tracknet_onnx/tracknet.onnx"  # 输出路径
#
#     # 转换并保存onnx模型，opset决定选用的算子集合
#     model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11, output_path=output_path)
#     output_names = [n.name for n in model_proto.graph.output]
#     print(output_names)  # 查看输出名称，后面推理用的到:  activation_18   (1, 360*640, 256)
#
#     onnx_model = onnx.load(output_path)
#     onnx.checker.check_model(onnx_model)


def export_TrackNetV3():
    # 一些参数
    img_size = (288, 512)
    TrackNetV3_checkpoint_file_path = r"./pretrain/TrackNetV3/model_best.pt"
    TrackNetV3_onnx_path = r"./pretrain/TrackNetV3/TrackNetV3.onnx"
    checkpoint = torch.load(TrackNetV3_checkpoint_file_path)
    param_dict = checkpoint['param_dict']
    num_frame = param_dict['num_frame']

    model = TrackNetV2(in_dim=num_frame*3, out_dim=num_frame).to("cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 9, img_size[0], img_size[1]).to("cuda")
    model.eval()
    model(dummy_input)
    im = torch.zeros(1, 9, img_size[0], img_size[1]).to("cuda")
    torch.onnx.export(model, im,
                      TrackNetV3_onnx_path,
                      verbose=False,
                      opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      )


if __name__ == '__main__':
    # 转换faster_rcnn模型
    # export_faster_rcnn()

    # 将h5转换成onnx
    # h5_to_pb()

    # 转换TrackNetV3模型
    export_TrackNetV3()

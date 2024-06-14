# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/21 02:12
@Version  :   1.0
@License  :   (C)Copyright 2024
"""

import argparse
from ast import arg, parse
from genericpath import isfile
import os
import sys
import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTUtil(object):
    def __init__(self, onnx_file_path=None, engine_file_path=None, precision_flop=None, img_size=(640, 640)) -> None:
        self.onnx_file_path = onnx_file_path
        self.engine_file_path = engine_file_path
        self.precision_flop = precision_flop
        self.img_size = img_size

        self.inputs = []
        self.outputs = []
        self.bindings = []

        self.logger = trt.Logger(trt.Logger.WARNING)

    def init_model(self):
        """加载 TRT 模型, 并加载一些多次推理过程共用的参数。
            情况 1、TRT 模型不存在，会先从输入的 onnx 模型创建一个 TRT 模型，并保存，再进行推导；
            情况 2、TRT 模型存在，直接进行推导
        """
        # 1、加载 logger 等级
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 2、加载 TRT 模型
        if os.path.isfile(self.engine_file_path):
            self.engine = self._read_TRT_file(self.engine_file_path)
            assert self.engine, "从 TRT 文件中读取的 engine 为 None ! "
        else:
            self.engine = self.onnx_to_TRT_model(self.onnx_file_path, self.engine_file_path, self.precision_flop)
            assert self.engine, "从 onnx 文件中转换的 engine 为 None ! "

        # 3、创建上下管理器，后面进行推导使用
        self.context = self.engine.create_execution_context()
        assert self.context, "创建的上下文管理器 context 为空，请检查相应的操作"

        # 4、创建数据传输流，在 cpu <--> gpu 之间传输数据的时候使用。
        self.stream = cuda.Stream()

        # 5、在 cpu 和 gpu 上申请内存
        for binding in self.engine:
            # 对应的输入输出内容的 个数，！！！注意是个数，不是内存的大小，
            size = trt.volume(self.engine.get_binding_shape(binding))
            # 内存的类型，如 int， bool。单个数据所占据的内存大小
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # 个数 * 单个内存的大小 = 内存的真实大小，先申请 cpu 上的内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配 gpu 上的内存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            print("size: {}, dtype: {}, device_mem: {}".format(size, dtype, device_mem))
            # 区分输入的和输出 申请的内存
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def inference(self, img):
        """对单张图片进行推理

        Args:
            img: 输入的图片

        Returns:
            返回 trt 推理的结果
        """

        # 1、对输入的数据进行处理
        self.inputs[0]['host'] = img  # 目前数据是放在 cpu 上
        # 2、将输入的数据同步到 gpu 上面 , 从 host -> device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 3、执行推理（Execute / Executev2）
        # execute_async_v2  ： 对批处理异步执行推理。此方法仅适用于从没有隐式批处理维度的网络构建的执行上下文。
        # execute_v2：      ： 在批次上同步执行推理。此方法仅适用于从没有隐式批处理维度的网络构建的执行上下文。
        # 同步和异步的差异    ： 在同一个上下文管理器中，程序的执行是否严格按照从上到下的过程。
        #                     如，连续输入多张图片，同步 会等处理完结果再去获得下一张，异步会开启多线程，提前处理数据
        self.context.execute_async_v2(
            bindings=self.bindings,  # 要进行推理的数据，放进去的时候，只有输入，出来输入、输出都有了
            stream_handle=self.stream.handle  # 将在其上执行推理内核的 CUDA 流的句柄。
        )
        # 4、Buffer 拷贝操作	Device to Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # 5、将 stream 中的数据进行梳理
        self.stream.synchronize()

        # 6、整理输出
        engine_infer_output = []
        for out in self.outputs:
            # out['host'] = np.reshape(out['host'], (-1, 85))
            engine_infer_output.append(out['host'])
        # engine_infer_output = np.concatenate(engine_infer_output, 0)

        return engine_infer_output

    def pre_process(self, img):
        """
        "对输入的图像进行预处理
        :param img: 原始的读取的图像
        :return: 返回处理好的图像
        """
        # 获取图像变换前后大小
        oh, ow, _ = img.shape
        h, w = self.img_size
        # 初始化固定输入大小的张量
        input_image = np.ones((h, w, 3)) * 128
        # 计算宽高比不变的缩放因子
        ratio = min(h / oh, w / ow)
        # 缩放图像
        nh, nw = int(oh * ratio), int(ow * ratio)
        resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # 填充图像
        input_image[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2: (w - nw) // 2 + nw, :] = resized_img
        # 变换维度和数值范围
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image / 255
        input_image = np.expand_dims(input_image, axis=0)
        input_image = np.ascontiguousarray(input_image, dtype=np.float32)

        return input_image, ratio

    def post_process(self, engine_infer_output, conf_thres, iou_thres, origin_img, ratio):
        """
        对网络输出的结果进行后处理

        :param engine_infer_output: 网络输出的结果，-> [(705600,)]
        :param conf_thres: 置信度阈值
        :param iou_thres: iou阈值
        :param origin_img: 送入网络之前的原始图片
        :param ratio: 送入网络的图片大小 / 原始图片的大小
        :return:
        """
        # 变换输出数据格式
        engine_infer_output = engine_infer_output[0]
        engine_infer_output = engine_infer_output.reshape((84, 8400))
        # 将(84, 8400)处理成(8400, 85)  85= box:4  conf:1 cls:80
        pred = np.transpose(engine_infer_output, (1, 0))  # (8400, 84)
        pred_class = pred[..., 4:]
        pred_conf = np.max(pred_class, axis=-1)
        pred = np.insert(pred, 4, pred_conf, axis=-1)  # (8400, 85)
        # 进行非极大值抑制nms，(N, 6) 6->[x,y,w,h,conf(最大类别概率),class]
        nms_result = self._nms(pred, conf_thres, iou_thres)
        # 转换数据最终输出的格式
        return self._cod_trf(nms_result, origin_img, ratio)

    @staticmethod
    def yolov8_pytorch_post_process(result):
        result_dict = result[0].boxes
        ret = [{
            "boxes": result_dict.xyxy,
            "labels": result_dict.cls,
            "scores": result_dict.conf
        }]
        return ret

    def faster_rcnn_post_process(self, result, origin_img, ratio):
        """
        因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上

        :param result: [[], [], []]
        :param origin_img: 送入网络之前的原始图片
        :param ratio: 送入网络的图片大小 / 原始图片的大小
        :return: [{"boxes": , "labels": , "scores": }]
        """
        # 获取图像大小
        h, w = self.img_size
        oh, ow, _ = origin_img.shape
        # 获取目标框数组
        boxes = result[0]
        # 转换坐标表示形式
        x1, y1, x2, y2 = boxes.transpose((1, 0))
        # 计算原图在等比例缩放后的尺寸
        nh, nw = oh * ratio, ow * ratio
        # 计算平移的量
        w_move, h_move = abs(w - nw) // 2, abs(h - nh) // 2
        ret_x1, ret_x2 = (x1 - w_move) / ratio, (x2 - w_move) / ratio
        ret_y1, ret_y2 = (y1 - h_move) / ratio, (y2 - h_move) / ratio
        # 返回指定的格式
        ret = [{
            "boxes": np.array([ret_x1, ret_y1, ret_x2, ret_y2]).transpose((1, 0)),
            "labels": result[1],
            "scores": result[2]
        }]
        return ret

    def _cod_trf(self, result, origin_img, ratio):
        """
        因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上

        :param result: (N, 6) 6->[x,y,w,h,conf(最大类别概率),class]
        :param origin_img: 送入网络之前的原始图片
        :param ratio: 送入网络的图片大小 / 原始图片的大小
        :return: [{"boxes": , "labels": , "scores": }]
        """
        # 获取图像大小
        h, w = self.img_size
        oh, ow, _ = origin_img.shape
        # 按列分开进行处理
        res = np.array(result)
        cx, cy, bw, bh, conf, cls = res.transpose((1, 0))
        # 转换坐标表示形式
        x1, y1, x2, y2 = self._xywh2xyxy(cx, cy, bw, bh)
        # 计算原图在等比例缩放后的尺寸
        nh, nw = oh * ratio, ow * ratio
        # 计算平移的量
        w_move, h_move = abs(w - nw) // 2, abs(h - nh) // 2
        ret_x1, ret_x2 = (x1 - w_move) / ratio, (x2 - w_move) / ratio
        ret_y1, ret_y2 = (y1 - h_move) / ratio, (y2 - h_move) / ratio
        # 返回指定的格式
        ret = [{
            "boxes": np.array([ret_x1, ret_y1, ret_x2, ret_y2]).transpose((1, 0)),
            "labels": cls,
            "scores": conf
        }]
        return ret

    def _nms(self, pred, conf_thres, iou_thres):
        """
        非极大值抑制nms

        :param pred: 模型输出并处理后的结果(8400, 85)
        :param conf_thres:置信度阈值
        :param iou_thres:iou阈值
        :return:
        """
        box = pred[pred[..., 4] > conf_thres]  # 置信度筛选
        cls_conf = box[..., 5:]
        cls = []
        for i in range(len(cls_conf)):
            cls.append(int(np.argmax(cls_conf[i])))

        total_cls = list(set(cls))  # 记录图像内共出现几种物体
        output_box = []
        # 每个预测类别分开考虑
        for i in range(len(total_cls)):
            clss = total_cls[i]
            cls_box = []
            temp = box[:, :6]
            for j in range(len(cls)):
                # 记录[x,y,w,h,conf(最大类别概率),class]值
                if cls[j] == clss:
                    temp[j][5] = clss
                    cls_box.append(temp[j][:6])
            #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]
            cls_box = np.array(cls_box)
            sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序
            # box_conf_sort = np.argsort(-box_conf)
            # 得到置信度最大的预测框
            max_conf_box = sort_cls_box[0]
            output_box.append(max_conf_box)
            sort_cls_box = np.delete(sort_cls_box, 0, 0)
            # 对除max_conf_box外其他的框进行非极大值抑制
            while len(sort_cls_box) > 0:
                # 得到当前最大的框
                max_conf_box = output_box[-1]
                del_index = []
                for j in range(len(sort_cls_box)):
                    current_box = sort_cls_box[j]
                    iou = self._get_iou(max_conf_box, current_box)
                    if iou > iou_thres:
                        # 筛选出与当前最大框Iou大于阈值的框的索引
                        del_index.append(j)
                # 删除这些索引
                sort_cls_box = np.delete(sort_cls_box, del_index, 0)
                if len(sort_cls_box) > 0:
                    output_box.append(sort_cls_box[0])
                    sort_cls_box = np.delete(sort_cls_box, 0, 0)
        return output_box

    def _xywh2xyxy(self, *box):
        """
        将xywh转换为左上角点和左下角点
        Args:
            box:
        Returns: x1y1x2y2
        """
        ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
               box[0] + box[2] // 2, box[1] + box[3] // 2]
        return ret

    def _get_inter(self, box1, box2):
        """
        计算相交部分面积
        Args:
            box1: 第一个框
            box2: 第二个框
        Returns: 相交部分的面积
        """
        x1, y1, x2, y2 = self._xywh2xyxy(*box1)
        x3, y3, x4, y4 = self._xywh2xyxy(*box2)
        # 验证是否存在交集
        if x1 >= x4 or x2 <= x3:
            return 0
        if y1 >= y4 or y2 <= y3:
            return 0
        # 将x1,x2,x3,x4排序，因为已经验证了两个框相交，所以x3-x2就是交集的宽
        x_list = sorted([x1, x2, x3, x4])
        x_inter = x_list[2] - x_list[1]
        # 将y1,y2,y3,y4排序，因为已经验证了两个框相交，所以y3-y2就是交集的宽
        y_list = sorted([y1, y2, y3, y4])
        y_inter = y_list[2] - y_list[1]
        # 计算交集的面积
        inter = x_inter * y_inter
        return inter

    def _get_iou(self, box1, box2):
        """
        计算交并比： (A n B)/(A + B - A n B)
        Args:
            box1: 第一个框
            box2: 第二个框
        Returns:  # 返回交并比的值
        """
        box1_area = box1[2] * box1[3]  # 计算第一个框的面积
        box2_area = box2[2] * box2[3]  # 计算第二个框的面积
        inter_area = self._get_inter(box1, box2)
        union = box1_area + box2_area - inter_area  # (A n B)/(A + B - A n B)
        iou = inter_area / union
        return iou

    def _read_TRT_file(self, engine_file_path):
        """从已经存在的文件中读取 TRT 模型

        Args:
            engine_file_path: 已经存在的 TRT 模型的路径

        Returns:
            加载完成的 engine
        """
        # 将路径转换为绝对路径防止出错
        engine_file_path = os.path.realpath(engine_file_path)
        # 建立一个反序列化器
        runtime = trt.Runtime(self.logger)
        # 判断TRT模型是否存在
        if not os.path.isfile(engine_file_path):
            print("模型文件：{}不存在".format(engine_file_path))
            return None
        # 反序列化TRT模型
        with open(engine_file_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine, "反序列化之后的引擎为空，确保转换过程的正确性."
        print("从{}成功载入引擎 . ".format(engine_file_path))

        return engine

    def onnx_to_TRT_model(self, onnx_file_path, engine_file_path, precision_flop):
        """构建期 -> 转换网络模型为 TRT 模型

        Args:
            onnx_file_path  : 要转换的 onnx 模型的路径
            engine_file_path: 转换之后的 TRT engine 的路径
            precision_flop  : 转换过程中所使用的精度

        Returns:
            转化成功: engine
            转换失败: None
        """
        # ---------------------------------#
        # 准备全局信息
        # ---------------------------------#
        # 构建一个 构建器
        builder = trt.Builder(self.logger)
        builder.max_batch_size = 1

        # ---------------------------------#
        # 第一步，读取 onnx
        # ---------------------------------#
        # 1-1、设置网络读取的 flag
        # EXPLICIT_BATCH 相教于 IMPLICIT_BATCH 模式，会显示的将 batch 的维度包含在张量维度当中，
        # 有了 batch大小的，我们就可以进行一些必须包含 batch 大小的操作了，如 Layer Normalization。
        # 不然在推理阶段，应当指定推理的 batch 的大小。目前主流的使用的 EXPLICIT_BATCH 模式
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # 1-3、构建一个空的网络计算图
        network = builder.create_network(network_flags)
        # 1-4、将空的网络计算图和相应的 logger 设置装载进一个 解析器里面
        parser = trt.OnnxParser(network, self.logger)
        # 1-5、打开 onnx 压缩文件，进行模型的解析工作。
        # 解析器 工作完成之后，网络计算图的内容为我们所解析的网络的内容。
        onnx_file_path = os.path.realpath(onnx_file_path)  # 将路径转换为绝对路径防止出错
        if not os.path.isfile(onnx_file_path):
            print("ONNX file not exist. Please check the onnx file path is right ? ")
            return None
        else:
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the onnx file {} . ".format(onnx_file_path))
                    # 出错了，将相关错误的地方打印出来，进行可视化处理`-`
                    for error in range(parser.num_errors):
                        print(parser.num_errors)
                        print(parser.get_error(error))
                    return None
            print("Completed parsing ONNX file . ")
        # 6、将转换之后的模型的输入输出的对应的大小进行打印，从而进行验证
        for i in range(network.num_outputs):
            print(i)
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        print("Network Description")
        batch_size = 0
        for inp in inputs:
            # 获取当前转化之前的 输入的 batch_size
            batch_size = inp.shape[0]
            print("Input '{}' with shape {} and dtype {} . ".format(inp.name, inp.shape, inp.dtype))
        for outp in outputs:
            print("Output '{}' with shape {} and dtype {} . ".format(outp.name, outp.shape, outp.dtype))
        # 确保 输入的 batch_size 不为零
        # assert batch_size > 0, "输入的 batch_size < 0, 请确定输入的参数是否满足要求. "

        # ---------------------------------#
        # 第二步，转换为 TRT 模型
        # ---------------------------------#
        # 2-1、设置 构建器 的 相关配置器
        # 应当丢弃老版本的 builder. 进行设置的操作
        config = builder.create_builder_config()
        # 2-2、设置 可以为 TensorRT 提供策略的策略源。如CUBLAS、CUDNN 等
        # 也就是在矩阵计算和内存拷贝的过程中选择不同的策略
        # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
        # 2-3、给出模型中任一层能使用的内存上限，这里是 2^30,为 2GB
        # 每一层需要多少内存系统分配多少，并不是每次都分 2 GB
        config.max_workspace_size = 1 << 30
        # 2-4、设置 模型的转化精度
        if precision_flop == "FP32":
            # config.set_flag(trt.BuilderFlag.FP32)
            pass
        elif precision_flop == "FP16":
            if not builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device . ")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
        elif precision_flop == "INT8":
            config.set_flag(trt.BuilderFlag.INT8)
        # 2-5，从构建器 构建引擎
        engine = builder.build_engine(network, config)

        # ---------------------------------#
        # 第三步，生成 SerializedNetwork
        # ---------------------------------#
        # 3-1、删除已经已经存在的版本
        engine_file_path = os.path.realpath(engine_file_path)  # 将路径转换为绝对路径防止出错
        if os.path.isfile(engine_file_path):
            try:
                os.remove(engine_file_path)
            except Exception:
                print("Cannot removing existing file: {} ".format(engine_file_path))
        print("Creating Tensorrt Engine: {}".format(engine_file_path))
        # 3-2、打开要写入的 TRT engine，利用引擎写入
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("ONNX -> TRT Success。 Serialized Engine Saved at: {} . ".format(engine_file_path))

        return engine

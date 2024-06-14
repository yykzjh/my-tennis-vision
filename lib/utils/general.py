# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/19 20:02
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import cv2
import argparse
import numpy as np

import json
import os
import random
import shutil
import time
import pickle

import torch
import torch.backends.cudnn as cudnn


def get_dtype():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    if dev == 'cuda':
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    print(f'Using device {device}')
    return dtype


def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # get videos properties
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height


def reproducibility(seed, deterministic, benchmark):
    random.seed(seed)  # 为python设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    torch.backends.cudnn.enabled = True

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(False)


def save_arguments(args, path):
    with open(path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()


def pre_write_txt(pred, file):
    f = open(file, 'a', encoding='utf-8')
    f.write(str(pred))
    f.write('\n')
    f.close()


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def datestr():
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))


def save_list(name, list):
    with open(name, "wb") as fp:
        pickle.dump(list, fp)


def load_list(name):
    with open(name, "rb") as fp:
        list_file = pickle.load(fp)

    return list_file


def load_json_file(file_path=r"./3DTooth.json"):
    def key_2_int(x):
        return {int(k): v for k, v in x.items()}

    assert os.path.exists(file_path), "{} file not exist.".format(file_path)
    json_file = open(file_path, 'r')
    dict = json.load(json_file, object_hook=key_2_int)
    json_file.close()

    return dict

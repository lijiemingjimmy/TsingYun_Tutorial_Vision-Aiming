"""MNIST digit model scaffold for Task 2.

Detector code should call the inference function in this module. Training code
lives in train.py so detector.py stays focused on board detection, corner
geometry, and PnP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import torch
import torch.nn.functional as F

RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_classifier.npz"


def preprocess_mnist_crop(board_crop: ImageLike) -> np.ndarray:
    """
    图像预处理，将扣出的装甲板处理成模型能吃进去的格式。
    """
    # 1. 确保输入是 NumPy 数组，以防测试中传入列表
    if not isinstance(board_crop, np.ndarray):
        board_crop = np.array(board_crop, dtype=np.uint8)

    # 2. 提取数字并二值化
    if len(board_crop.shape) == 3:
        # 游戏中数字是白色的(R,G,B都较高)，边框是红色的(R高，G,B低)
        # 取通道最小值，红框会变为0，只保留发白的数字部分
        min_c = np.min(board_crop, axis=2)
        # 考虑到画面可能偏暗，最大像素值可能只有90左右，使用30作为阈值
        _, binary = cv2.threshold(min_c, 30, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(board_crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 3. 调整图像大小为 28x28 像素 (MNIST数据集和我们模型的标准尺寸)
    resized = cv2.resize(binary, (28, 28))
    
    # 4. 将像素值从 0~255 归一化到 0.0~1.0 的浮点数
    normalized = resized.astype(np.float32) / 255.0
    
    # 4. 对齐 train.py 里的 Normalize((0.1307,), (0.3081,)) 进行减均值除以方差操作
    normalized = (normalized - 0.1307) / 0.3081
    
    # 5. 调整维度以适配 PyTorch 的要求：增加 Batch 和 Channel 维度，最后变为 (1, 1, 28, 28)
    tensor_input = np.expand_dims(normalized, axis=(0, 1))
    
    return tensor_input


def load_mnist_model(model_path: Path = DEFAULT_MODEL_PATH) -> torch.nn.Module:
    """
    加载模型架构及预训练好的权重文件。
    """
    # 导入 train.py 中定义的模型架构类
    from train import MNISTClassifier
    
    # 实例化模型
    model = MNISTClassifier()
    
    # 加载保存在 model_path 的权重。
    # 设置 map_location="cpu" 可以防止因为你在一台没显卡的机器上跑而发生报错
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    # 切换为评估模式（eval），以此关闭训练时特有的操作（如Dropout随机失活），保证预测稳定输出
    model.eval()
    
    return model


def predict_mnist_digit(model: torch.nn.Module, model_input: np.ndarray) -> tuple[int, float]:
    """
    用模型进行预测，得到最后分类的数字和概率
    """
    # 1. 把预处理好的 NumPy 数组转成 PyTorch 的 Tensor
    input_tensor = torch.from_numpy(model_input)
    
    # 2. 放在 torch.no_grad() 下推理，因为只是前向预测，不需要记录梯度反向传播，这样省显存且速度快
    with torch.no_grad():
        # 获取原始网络输出值 (Logits)
        logits = model(input_tensor)
        
        # 模型输出的 Logits 需要通过 Softmax 函数转换为各个类别的概率值 (加起来等于1)
        probabilities = F.softmax(logits, dim=1)
        
    # 3. 拿到概率最高的那一项所在的索引，就是预测出的数字
    digit = int(torch.argmax(probabilities, dim=1).item())
    
    # 4. 获取刚刚算出的那个类别的概率值作为置信度返回
    confidence = float(probabilities[0, digit].item())
    
    return digit, confidence


def classify_mnist_digit(board_crop: ImageLike, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[int, float]:
    """
    整合上面三个函数的流水线包装。
    """
    # 1. 预处理提取出的像素块
    model_input = preprocess_mnist_crop(board_crop)
    
    # 2. 加载模型（每次推理都加载一次其实比较浪费性能，但在这里契合整体的函数切分设计）
    model = load_mnist_model(model_path)
    
    # 3. 实施推理
    digit, confidence = predict_mnist_digit(model, model_input)
    
    return digit, confidence

import numpy as np
import cv2
from typing import Union

def one_dim_kmeans(inputs):
    """
    一维K-means聚类算法，用于二值化提取的水印
    :param inputs: 输入数据
    :return: 二值化结果
    """
    # 检查输入数组是否为空或只有一个元素
    if inputs.size == 0:
        return np.array([], dtype=bool)
    if inputs.size == 1:
        return np.array([inputs[0] > 0], dtype=bool)
    
    # 检查所有元素是否相同
    if np.all(inputs == inputs[0]):
        # 如果所有元素都相同，根据值返回全0或全1
        return np.ones_like(inputs, dtype=bool) if inputs[0] > 0 else np.zeros_like(inputs, dtype=bool)
    
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  # 1. 初始化中心点
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  # 2. 检查所有点与这k个点之间的距离，每个点归类到最近的中心
        
        # 安全地计算均值，避免空数组
        if np.any(~is_class01):
            center_0 = inputs[~is_class01].mean()
        else:
            center_0 = inputs.min()
            
        if np.any(is_class01):
            center_1 = inputs[is_class01].mean()
        else:
            center_1 = inputs.max()
            
        center = [center_0, center_1]  # 3. 重新找中心点
        
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:  # 4. 停止条件
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01

def random_strategy1(seed, size, block_shape):
    """
    随机策略1：为每个块生成唯一的置乱序列
    :param seed: 随机种子
    :param size: 序列数量
    :param block_shape: 块大小
    :return: 置乱索引数组
    """
    return np.random.RandomState(seed) \
        .random(size=(size, block_shape)) \
        .argsort(axis=1)

def random_strategy2(seed, size, block_shape):
    """
    随机策略2：所有块使用相同的置乱序列
    :param seed: 随机种子
    :param size: 序列数量
    :param block_shape: 块大小
    :return: 置乱索引数组
    """
    one_line = np.random.RandomState(seed) \
        .random(size=(1, block_shape)) \
        .argsort(axis=1)

    return np.repeat(one_line, repeats=size, axis=0)

def image_to_wm_bit(image: Union[str, np.ndarray], threshold: int = 128) -> np.ndarray:
    """
    将图像转换为水印位图
    :param image: 输入图像（灰度或RGB）
    :param threshold: 二值化阈值
    :return: 一维位图数组（uint8类型，值为0或1）
    """
    if isinstance(image, str):
        # 从文件加载
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法加载图像文件: {image}")
    elif isinstance(image, np.ndarray):
        # 确保输入是正确的数据类型
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise TypeError(f"不支持的图像数据类型: {image.dtype}")
        
        # 转换为灰度图
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()
        
        # 确保数据类型为uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
    else:
        raise TypeError("输入必须是文件路径或numpy数组")

    # 二值化处理
    _, binary = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    # 确保返回uint8类型的0/1数组
    return binary.flatten().astype(np.uint8)

def wm_bit_to_image(wm_bit, shape):
    """
    将水印位图转换为图像
    :param wm_bit: 一维位图数组
    :param shape: 目标图像形状 (height, width)
    :return: 二值图像
    """
    return (wm_bit.reshape(shape) * 255).astype(np.uint8)
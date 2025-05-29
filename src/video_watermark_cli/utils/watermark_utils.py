import numpy as np
import cv2

def one_dim_kmeans(inputs):
    """
    一维K-means聚类算法，用于二值化提取的水印
    :param inputs: 输入数据
    :return: 二值化结果
    """
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  # 1. 初始化中心点
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  # 2. 检查所有点与这k个点之间的距离，每个点归类到最近的中心
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]  # 3. 重新找中心点
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

def image_to_wm_bit(image, threshold=128):
    """
    将图像转换为水印位图
    :param image: 输入图像（灰度或RGB）
    :param threshold: 二值化阈值
    :return: 一维位图数组
    """
    if isinstance(image, str):
        # 从文件加载
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, np.ndarray):
        # 直接使用数组
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    else:
        raise TypeError("输入必须是文件路径或numpy数组")

    _, binary = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    return binary.flatten()

def wm_bit_to_image(wm_bit, shape):
    """
    将水印位图转换为图像
    :param wm_bit: 一维位图数组
    :param shape: 目标图像形状 (height, width)
    :return: 二值图像
    """
    return (wm_bit.reshape(shape) * 255).astype(np.uint8)
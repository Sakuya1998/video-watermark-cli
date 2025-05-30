import qrcode
import numpy as np
import logging
import cv2
from typing import Optional, Union, Tuple
from PIL.Image import Image as PILImage
import pyzbar.pyzbar as pyzbar
from video_watermark_cli.config import QR_SIZE

logger = logging.getLogger(__name__)

def generate_qrcode(data: str, size: int = QR_SIZE) -> np.ndarray:
    """
    生成二维码
    :param data: 要编码的数据
    :param size: 二维码大小
    :return: 二维码图像
    """
    if not data:
        raise ValueError("Data cannot be empty.")
    if len(data) > 2953:  # QR Code Version 40的最大容量
        raise ValueError("Data too long for QR code (max 2953 chars)")

    try:
        qr = qrcode.QRCode(
            version=None,  # 自动选择版本
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        # 生成二维码图像
        base_img = qr.make_image(fill_color="black", back_color="white")
        # 调整大小
        base_array = np.array(base_img)
        base_resized = cv2.resize(base_array.astype(np.uint8) * 255, (size, size))

        if base_resized is None:
            raise ValueError("生成二维码失败")
        else:
            return base_resized

    except Exception as e:
        logger.error(f"生成二维码失败: {e}")
        raise ValueError(f"生成二维码失败：{e}")

def decode_qrcode(image: Union[PILImage, np.ndarray]) -> str:
    """
    解码二维码
    :param image: 二维码图像
    :return: 解码后的数据
    """
    if image is None:
        raise ValueError("Image cannot be None")
    try:
        # 统一处理不同输入类型
        if isinstance(image, PILImage):
            image = np.array(image)
        elif not isinstance(image, np.ndarray):
            raise ValueError("Unsupported image type")

        decoded_objects = pyzbar.decode(image)
        if not decoded_objects:
            raise ValueError("No QR code detected")
        # 打印解码结果  
        result = decoded_objects[0].data.decode("utf-8")
        logger.info(f"解码结果: {result}")
        return result
    except UnicodeDecodeError:
        logger.warning("QR code contains binary data")
        return decoded_objects[0].data
    except Exception as e:
        logger.error(f"解码二维码失败: {e}")
        raise ValueError(f"解码二维码失败：{e}")
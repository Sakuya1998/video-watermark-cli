import numpy as np
from numpy.linalg import svd
import copy
import cv2
from typing import Optional, Tuple, Union
from cv2 import dct, idct
from pywt import dwt2, idwt2
import logging
from video_watermark_cli.core.pool import AutoPool
from video_watermark_cli.utils.watermark_utils import one_dim_kmeans, random_strategy1, random_strategy2

logger = logging.getLogger(__name__)

class WaterMarkCore:
    def __init__(self, password_img=1, mode='common', processes=None):
        self.block_shape = np.array([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = 36, 20  # d1/d2 越大鲁棒性越强,但输出图片的失真越大

        # init data
        self.img, self.img_YUV = None, None  # self.img 是原图，self.img_YUV 对像素做了加白偶数化
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # 每个通道 dct 的结果
        self.ca_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ca_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

        self.wm_size, self.block_num = 0, 0  # 水印的长度，原图片可插入信息的个数
        self.pool = AutoPool(mode=mode, processes=processes)

        self.fast_mode = False
        self.alpha = None  # 用于处理透明图

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img):
        # 处理透明图
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        # 读入图片->YUV化->加白边使像素变偶数->四维分块
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        # 如果不是偶数，那么补上白边，Y（明亮度）UV（颜色）
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def block_add_wm(self, arg):
        if self.fast_mode:
            return self.block_add_wm_fast(arg)
        else:
            return self.block_add_wm_slow(arg)

    def block_add_wm_slow(self, arg):
        block, shuffler, i = arg
        # dct->(flatten->加密->逆flatten)->svd->打水印->逆svd->(flatten->解密->逆flatten)->逆dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = dct(block)

        # 加密（打乱顺序）
        block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)
        u, s, v = svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2

        block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
        block_dct_flatten[shuffler] = block_dct_flatten.copy()
        return idct(block_dct_flatten.reshape(self.block_shape))

    def block_add_wm_fast(self, arg):
        # dct->svd->打水印->逆svd->逆dct
        block, shuffler, i = arg
        wm_1 = self.wm_bit[i % self.wm_size]

        u, s, v = svd(dct(block))
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1

        return idct(np.dot(u, np.dot(np.diag(s), v)))

    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3

        self.idx_shuffle = random_strategy1(self.password_img, self.block_num,
                                            self.block_shape[0] * self.block_shape[1])
        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]

            # 4维分块变回2维
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # 逆变换回去
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # 合并3通道
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 之前如果不是2的整数，增加了白边，这里去除掉
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def block_get_wm(self, args):
        if self.fast_mode:
            return self.block_get_wm_fast(args)
        else:
            return self.block_get_wm_slow(args)

    def block_get_wm_slow(self, args):
        block, shuffler = args
        # dct->flatten->加密->逆flatten->svd->解水印
        block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def block_get_wm_fast(self, args):
        block, shuffler = args
        # dct->svd->解水印
        u, s, v = svd(dct(block))
        wm = (s[0] % self.d1 > self.d1 / 2) * 1

        return wm

    def extract_raw(self, img):
        # 每个分块提取 1 bit 信息
        self.read_img_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3个channel，length 个分块提取的水印，全都记录下来

        self.idx_shuffle = random_strategy1(seed=self.password_img,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )
        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.block_get_wm,
                                                     [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # 对循环嵌入+3个 channel 求平均
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()

        # 提取每个分块埋入的 bit：
        wm_block_bit = self.extract_raw(img=img)
        # 做平均：
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        wm_avg = self.extract(img=img, wm_shape=wm_shape)

        return one_dim_kmeans(wm_avg)

class VideoWatermarker:
    def __init__(self, password_img=1, mode='common', processes=None):
        """
        视频水印处理器
        :param password_img: 随机种子（用于置乱）
        :param mode: 处理模式（'common' 或 'fast'）
        :param processes: 并行进程数
        """
        self.core = WaterMarkCore(password_img, mode, processes)
        self.frame_count = 0
        self.wm_bit = None
        self.wm_shape = None
        
    def set_watermark(self, wm_bit, wm_shape=None):
        """
        设置水印
        :param wm_bit: 水印位图（一维数组）
        :param wm_shape: 水印形状（可选，用于提取时）
        """
        if not isinstance(wm_bit, np.ndarray) or wm_bit.dtype != np.uint8:
            raise ValueError("水印必须为uint8类型的numpy数组")        
        self.core.read_wm(wm_bit)
        self.wm_bit = wm_bit
        self.wm_shape = wm_shape

    def _validate_frame_range(self, start_frame: int, end_frame: Optional[int], total_frames: int) -> Tuple[int, int]:
        """验证并修正帧范围"""
        if total_frames <= 0:
            raise ValueError("视频帧数无效（可能不是视频文件）")

        # 处理结束帧
        if end_frame is None:
            end_frame = total_frames - 1
        else:
            end_frame = min(end_frame, total_frames - 1)

        # 验证范围有效性
        if start_frame < 0:
            raise ValueError(f"起始帧({start_frame})不能为负数")
        if start_frame >= total_frames:
            raise ValueError(f"起始帧({start_frame})超出视频总帧数({total_frames})")
        if start_frame > end_frame:
            raise ValueError(f"起始帧({start_frame})不能大于结束帧({end_frame})")

        return start_frame, end_frame

    def process_frame(self, frame, operation='embed'):
        """
        处理单个视频帧
        :param frame: 输入帧（BGR格式）
        :param operation: 'embed' 嵌入水印, 'extract' 提取水印
        :return: 处理后的帧（嵌入操作）或提取的水印位（提取操作）
        """
        if operation == 'embed':
            # 嵌入水印
            return self.core.embed(frame)
        elif operation == 'extract':
            # 提取水印
            if self.wm_shape is None:
                raise ValueError("Watermark shape must be set for extraction")
            return self.core.extract(frame, self.wm_shape)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def process_video(
        self,
        input_path: str,
        output_path: Optional[str],
        operation: str = 'embed',
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        progress_interval: int = 50
    ) -> Optional[np.ndarray]:
        """
        处理视频文件（支持帧范围控制）
        
        参数:
            input_path: 输入视频路径
            output_path: 输出路径（提取模式可设为None）
            operation: 'embed'嵌入/'extract'提取
            start_frame: 起始帧号（从0开始）
            end_frame: 结束帧号（None表示到视频末尾）
            progress_interval: 进度打印间隔帧数
            
        返回:
            提取模式返回水印数据，嵌入模式返回None
        """
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {input_path}")

        try:
            # 获取视频属性
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 验证帧范围
            start_frame, end_frame = self._validate_frame_range(start_frame, end_frame, total_frames)
            logger.info(f"处理范围: 第{start_frame}-{end_frame}帧 (共{end_frame-start_frame+1}帧)")

            # 初始化输出（仅嵌入模式）
            if operation == 'embed':
                if not output_path:
                    raise ValueError("嵌入模式必须指定输出路径")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            else:
                out = None
                wm_accumulator = None
                valid_frames = 0

            # 跳转到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame

            # 逐帧处理
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"第{current_frame}帧读取失败，提前终止")
                    break

                try:
                    if operation == 'embed':
                        processed = self.core.embed(frame)
                        out.write(processed)
                    else:
                        extracted = self.core.extract(frame, self.wm_shape)
                        if wm_accumulator is None:
                            wm_accumulator = np.zeros_like(extracted)
                        wm_accumulator += extracted
                        valid_frames += 1
                except Exception as e:
                    logger.error(f"第{current_frame}帧处理失败: {str(e)}")
                    if operation == 'extract':
                        continue  # 提取模式允许跳过错误帧
                    else:
                        raise

                # 进度显示
                if current_frame % progress_interval == 0:
                    logger.info(f"已处理 {current_frame-start_frame}/{end_frame-start_frame} 帧")

                current_frame += 1

            # 提取模式返回结果
            if operation == 'extract' and valid_frames > 0:
                return one_dim_kmeans(wm_accumulator / valid_frames)
            return None

        finally:
            # 确保资源释放
            cap.release()
            if operation == 'embed' and 'extract' in locals():
                out.release()
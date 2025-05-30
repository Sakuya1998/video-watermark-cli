import numpy as np
from scipy.linalg import svd
import copy
import cv2
from typing import Optional, Tuple, Union
from scipy.fftpack import dct, idct
from pywt import dwt2, idwt2
import logging
from video_watermark_cli.config import D1, D2
from video_watermark_cli.core.pool import AutoPool
from video_watermark_cli.utils.watermark_utils import one_dim_kmeans, random_strategy1, random_strategy2

logger = logging.getLogger(__name__)

class WaterMarkCore:
    def __init__(self, password_img: int = 1, mode: str = 'common', processes: Optional[int] = None) -> None:
        self.block_shape: np.ndarray = np.array([4, 4])
        self.password_img: int = password_img
        self.d1: int = D1  # 从配置文件读取，避免硬编码高值导致失真
        self.d2: int = D2  # 从配置文件读取，d1/d2 越大鲁棒性越强,但输出图片的失真越大

        # init data
        self.img: Optional[np.ndarray] = None  # 原图
        self.img_YUV: Optional[np.ndarray] = None  # 对像素做了加白偶数化
        self.ca: list = [np.array([])] * 3  # 每个通道 dct 的结果
        self.hvd: list = [np.array([])] * 3
        self.ca_block: list = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ca_part: list = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

        self.wm_size: int = 0  # 水印的长度
        self.block_num: int = 0  # 原图片可插入信息的个数
        self.pool = AutoPool(mode=mode, processes=processes)

        self.fast_mode: bool = False
        self.alpha: Optional[np.ndarray] = None  # 用于处理透明图

    def init_block_index(self) -> None:
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img: np.ndarray) -> None:
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
        # 在YUV色彩空间中，(128, 128, 128)对应中性灰色，避免色彩偏移
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(128, 128, 128))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit: np.ndarray) -> None:
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

    def embed(self) -> np.ndarray:
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
        
        # 确保输出图像为uint8类型，避免视频编码问题
        embed_img = embed_img.astype(np.uint8)
        
        if self.alpha is not None:
            embed_img = cv2.merge([embed_img, self.alpha])
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
        # SVD返回的s是一维数组，取第一个元素作为标量
        s0_val = s[0]
        wm = int(s0_val % self.d1 > self.d1 / 2)
        if self.d2:
            s1_val = s[1]
            tmp = int(s1_val % self.d2 > self.d2 / 2)
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def block_get_wm_fast(self, args):
        block, shuffler = args
        # dct->svd->解水印
        u, s, v = svd(dct(block))
        # SVD返回的s是一维数组，取第一个元素作为标量
        s0_val = s[0]
        wm = int(s0_val % self.d1 > self.d1 / 2)

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

    def extract(self, img: np.ndarray, wm_shape: Tuple[int, ...]) -> np.ndarray:
        self.wm_size = np.array(wm_shape).prod()

        # 提取每个分块埋入的 bit：
        wm_block_bit = self.extract_raw(img=img)
        # 做平均：
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img: np.ndarray, wm_shape: Tuple[int, ...]) -> np.ndarray:
        wm_avg = self.extract(img=img, wm_shape=wm_shape)

        return one_dim_kmeans(wm_avg)

class VideoWatermarker:
    def __init__(self, password_img: int = 1, mode: str = 'common', processes: Optional[int] = None) -> None:
        """
        视频水印处理器
        :param password_img: 随机种子（用于置乱）
        :param mode: 处理模式（'common' 或 'fast'）
        :param processes: 并行进程数
        """
        self.core = WaterMarkCore(password_img, mode, processes)
        self.frame_count: int = 0
        self.wm_bit: Optional[np.ndarray] = None
        self.wm_shape: Optional[Tuple[int, ...]] = None
        
    def set_watermark(self, wm_bit: Optional[np.ndarray], wm_shape: Optional[Tuple[int, ...]] = None) -> None:
        """
        设置水印
        :param wm_bit: 水印位图（一维数组），提取模式下可为None
        :param wm_shape: 水印形状（可选，用于提取时）
        """
        if wm_bit is None:
            # 提取模式下，水印可以为None
            self.wm_bit = None
            self.wm_shape = wm_shape
            # 不调用core.read_wm，因为提取模式不需要水印数据
        else:
            # 嵌入模式下，验证水印格式
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

    def process_frame(self, frame: np.ndarray, operation: str = 'embed') -> Union[np.ndarray, np.ndarray]:
        """
        处理单个视频帧
        :param frame: 输入帧（BGR格式）
        :param operation: 'embed' 嵌入水印, 'extract' 提取水印
        :return: 处理后的帧（嵌入操作）或提取的水印位（提取操作）
        """
        if operation == 'embed':
            # 嵌入水印
            self.core.read_img_arr(frame)
            return self.core.embed()
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

        # 初始化输出变量
        out = None
        
        try:
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 读取第一帧来获取真实的帧尺寸（避免元数据不准确的问题）
            ret, first_frame = cap.read()
            if not ret:
                raise IOError(f"无法读取视频的第一帧: {input_path}")
            
            # 从实际帧数据获取真实尺寸
            frame_height, frame_width = first_frame.shape[:2]
            # 直接使用源视频帧的长和宽：VideoWriter的(width, height)对应帧数据的(frame_width, frame_height)
            width, height = frame_width, frame_height
            # 注意：OpenCV帧数据shape为(height, width, channels)，VideoWriter期望size为(width, height)
            
            # 重置视频到开始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 验证帧范围
            start_frame, end_frame = self._validate_frame_range(start_frame, end_frame, total_frames)
            logger.info(f"处理范围: 第{start_frame}-{end_frame}帧 (共{end_frame-start_frame+1}帧)")

            # 初始化输出（仅嵌入模式）
            if operation == 'embed':
                if not output_path:
                    raise ValueError("嵌入模式必须指定输出路径")
                
                # 检查输出目录是否存在
                import os
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    logger.info(f"创建输出目录: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"视频参数: 实际帧尺寸=({frame_width}, {frame_height}), VideoWriter尺寸=({frame_width}, {frame_height}), 帧率={fps}, 输出路径={output_path}")
                
                # 使用H.264编码器提供更好的质量和压缩比
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                logger.info("尝试使用H.264编码器")
                # OpenCV的VideoWriter期望的尺寸格式是(width, height)，对应帧数据的(frame_width, frame_height)
                # 帧数据shape为(frame_height, frame_width, channels)，所以VideoWriter应该使用(frame_width, frame_height)
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                # 如果H.264不可用，尝试其他编码器
                if not out.isOpened():
                    logger.warning("H.264编码器不可用，尝试XVID编码器")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                    
                    if not out.isOpened():
                        logger.warning("XVID编码器不可用，使用mp4v编码器")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        if not out.isOpened():
                            # 尝试最后的备选方案
                            logger.warning("mp4v编码器不可用，尝试MJPG编码器")
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            
                            if not out.isOpened():
                                logger.error(f"所有编码器都不可用，系统信息:")
                                logger.error(f"OpenCV版本: {cv2.__version__}")
                                logger.error(f"输出路径: {output_path}")
                                logger.error(f"视频参数: {width}x{height}@{fps}fps")
                                raise RuntimeError(f"无法初始化任何视频编码器，输出路径: {output_path}")
                
                logger.info(f"成功初始化VideoWriter，使用编码器: {fourcc}")
            else:
                out = None
                wm_accumulator = None
                valid_frames = 0

            # 从第0帧开始处理整个视频
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = 0

            # 逐帧处理整个视频
            while current_frame < total_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"第{current_frame}帧读取失败，提前终止")
                    break

                try:
                    if operation == 'embed':
                        # 检查VideoWriter状态
                        if not out.isOpened():
                            raise RuntimeError(f"VideoWriter未正确初始化，无法写入第{current_frame}帧")
                        
                        # 判断当前帧是否在处理范围内
                        if start_frame <= current_frame <= end_frame:
                            # 在范围内：进行水印处理
                            self.core.read_img_arr(frame)
                            processed = self.core.embed()
                            
                            # 验证处理后的帧
                            if processed is None or processed.size == 0:
                                raise ValueError(f"第{current_frame}帧处理结果为空")
                            
                            # 确保帧尺寸正确 - OpenCV帧格式为(height, width, channels)，VideoWriter期望(width, height)
                            # 帧数据shape应该是(frame_height, frame_width, channels)，对应VideoWriter的(width, height)
                            expected_shape = (frame_height, frame_width)  # 帧数据的实际shape格式
                            if processed.shape[:2] != expected_shape:
                                logger.warning(f"第{current_frame}帧尺寸不匹配，调整中: {processed.shape[:2]} -> {expected_shape}")
                                processed = cv2.resize(processed, (frame_width, frame_height))
                        else:
                            # 在范围外：直接使用原始帧
                            processed = frame
                            
                            # 确保帧尺寸正确 - OpenCV帧格式为(height, width, channels)，VideoWriter期望(width, height)
                            # 帧数据shape应该是(frame_height, frame_width, channels)，对应VideoWriter的(width, height)
                            expected_shape = (frame_height, frame_width)  # 帧数据的实际shape格式
                            if processed.shape[:2] != expected_shape:
                                processed = cv2.resize(processed, (frame_width, frame_height))
                        
                        # 详细的帧数据检查
                        logger.debug(f"第{current_frame}帧数据: shape={processed.shape}, dtype={processed.dtype}, min={processed.min()}, max={processed.max()}")
                        
                        # 确保数据类型为uint8
                        if processed.dtype != np.uint8:
                            logger.warning(f"第{current_frame}帧数据类型不正确: {processed.dtype}, 转换为uint8")
                            processed = processed.astype(np.uint8)
                        
                        # 确保值范围在0-255
                        if processed.min() < 0 or processed.max() > 255:
                            logger.warning(f"第{current_frame}帧像素值超出范围: [{processed.min()}, {processed.max()}], 进行裁剪")
                            processed = np.clip(processed, 0, 255).astype(np.uint8)
                        
                        # 检查VideoWriter状态
                        if not out.isOpened():
                            logger.error(f"VideoWriter已关闭，无法写入第{current_frame}帧")
                            raise RuntimeError("VideoWriter意外关闭")
                        
                        out.write(processed)
                        # OpenCV的write方法无返回值，无法直接判断写入是否失败，如有异常会抛出
                        # 若后续帧输出异常，可通过日志和调试帧定位问题
                    else:
                        # 提取模式：只处理指定范围内的帧
                        if start_frame <= current_frame <= end_frame:
                            try:
                                # 确保wm_shape不为None
                                if self.wm_shape is None:
                                    raise ValueError("提取水印需要指定水印形状(wm_shape)")
                                extracted = self.core.extract(frame, self.wm_shape)
                                if wm_accumulator is None:
                                    wm_accumulator = np.zeros_like(extracted)
                                wm_accumulator += extracted
                                valid_frames += 1
                            except Exception as e:
                                logger.error(f"提取水印失败: {str(e)}")
                                # 在提取模式下，如果处理单帧失败，记录错误但继续处理其他帧
                except Exception as e:
                    logger.error(f"第{current_frame}帧处理失败: {str(e)}")
                    if operation == 'extract':
                        continue  # 提取模式允许跳过错误帧
                    else:
                        raise

                # 进度显示
                if current_frame % progress_interval == 0:
                    processed_frames = min(current_frame - start_frame + 1, end_frame - start_frame + 1)
                    total_frames_to_process = end_frame - start_frame + 1
                    logger.info(f"已处理 {processed_frames}/{total_frames_to_process} 帧 (当前帧: {current_frame})")

                current_frame += 1

            # 提取模式返回结果
            if operation == 'extract' and valid_frames > 0:
                try:
                    # 计算平均值并进行二值化
                    avg_wm = wm_accumulator / valid_frames
                    # 使用安全的方式调用one_dim_kmeans
                    return one_dim_kmeans(avg_wm)
                except Exception as e:
                    logger.error(f"二值化水印失败: {str(e)}")
                    # 如果二值化失败，尝试直接返回原始数据
                    if wm_accumulator is not None:
                        # 简单阈值处理
                        threshold = np.mean(wm_accumulator)
                        return wm_accumulator > threshold
            return None

        finally:
            # 确保资源释放
            cap.release()
            if operation == 'embed' and out is not None:
                out.release()
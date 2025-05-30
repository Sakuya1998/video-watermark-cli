
"""
配置文件
"""
# === Logging parameters (日志参数) === 
LOG_PATH = "logs"  # 日志文件夹路径
LOG_LEVEL = "DEBUG"  # 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）

# === FFmpeg parameters (FFmpeg 参数) ===
FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"

# === QR Code parameters (二维码参数) ===
QR_SIZE = 64  # 二维码图像尺寸（像素

# === Watermark parameters (水印参数) ===
PASSWORD = 123456  # 水印密码（整数类型，用作随机种子）

# 水印强度参数（控制鲁棒性vs视觉质量的平衡）
# 推荐设置：
# - 高质量模式：D1=10, D2=5  （失真最小，但鲁棒性较低）
# - 平衡模式：  D1=15, D2=8  （默认，平衡失真和鲁棒性）
# - 鲁棒模式：  D1=25, D2=15 （鲁棒性强，但失真较明显）
D1 = 15  # 主要强度参数（原默认36会导致严重失真）
D2 = 8   # 次要强度参数（原默认20会导致严重失真）
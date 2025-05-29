import subprocess
import json
from typing import Optional, Dict
from pathlib import Path
import os
import logging
from shutil import which
from video_watermark_cli.config import FFMPEG_PATH as CONFIG_FFMPEG_PATH, FFPROBE_PATH as CONFIG_FFPROBE_PATH

logger = logging.getLogger(__name__)

# 优雅 fallback 到系统 ffmpeg/ffprobe
FFMPEG_PATH = CONFIG_FFMPEG_PATH or which("ffmpeg")
FFPROBE_PATH = CONFIG_FFPROBE_PATH or which("ffprobe")

if not FFMPEG_PATH or not FFPROBE_PATH:
    logger.warning("FFmpeg 或 FFprobe 未正确配置，将依赖系统默认路径")


def extract_video_info(video_path: str) -> Dict:
    """
    提取视频和音频的完整编码参数
    
    Args:
        video_path: 视频路径
    
    Returns:
        dict: 包含视频流和音频流的参数字典
    """
    if not Path(video_path).exists():
        logger.error("视频文件不存在: %s", video_path)
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-show_entries",
        "stream=codec_type,codec_name,bit_rate,profile,width,height,r_frame_rate,pix_fmt,sample_rate,channels,level,gop_size",
        "-of", "json",
        video_path
    ]

    logger.debug("执行 ffprobe 命令: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("ffprobe 提取失败: %s", e.stderr)
        raise RuntimeError(f"ffprobe 提取失败: {e.stderr}") from e

    info = json.loads(result.stdout)
    
    video_info = {}
    audio_info = {}
    
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            fr_str = stream.get("r_frame_rate", "30/1")
            try:
                num, denom = map(int, fr_str.split("/"))
                frame_rate = round(num / denom, 2)
            except Exception:
                frame_rate = 30.0

            video_info = {
                "codec": stream.get("codec_name"),
                "bitrate": stream.get("bit_rate"),
                "profile": stream.get("profile"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "frame_rate": frame_rate,
                "pix_fmt": stream.get("pix_fmt"),
                "level": stream.get("level"),
                "gop_size": stream.get("gop_size"),
            }

        elif stream.get("codec_type") == "audio":
            audio_info = {
                "codec": stream.get("codec_name"),
                "bitrate": stream.get("bit_rate"),
                "sample_rate": stream.get("sample_rate"),
                "channels": stream.get("channels"),
            }

    logger.debug("提取到的视频信息: %s", video_info)
    logger.debug("提取到的音频信息: %s", audio_info)

    return {
        "video": video_info,
        "audio": audio_info
    }


def transcode_video(temp_video_path: str, original_video_path: str, output_path: str) -> bool:
    """
    使用ffmpeg将临时视频和原视频音频合成，保持视频参数一致，输出到output_path

    Args:
        temp_video_path: 无音频的临时视频路径
        original_video_path: 原始视频路径（用于提取音频和参数）
        output_path: 转码输出文件路径

    Returns:
        bool: 成功返回 True，失败返回 False
    """
    try:
        logger.info("开始转码：源视频：%s，临时视频：%s，输出路径：%s",
                    original_video_path, temp_video_path, output_path)

        if not Path(temp_video_path).exists():
            logger.error("临时视频文件不存在: %s", temp_video_path)
            raise FileNotFoundError(f"临时视频文件不存在: {temp_video_path}")
        if not Path(original_video_path).exists():
            logger.error("原始视频文件不存在: %s", original_video_path)
            raise FileNotFoundError(f"原始视频文件不存在: {original_video_path}")

        info = extract_video_info(original_video_path)
        video_info = info.get("video", {})
        audio_info = info.get("audio", {})

        if not video_info:
            logger.error("无法获取原视频的视频流信息")
            raise RuntimeError("无法获取原视频的视频流信息")

        codec_map = {
            "h264": "libx264",
            "hevc": "libx265",
            "vp9": "libvpx-vp9",
            "av1": "libaom-av1",
        }
        src_codec = video_info.get("codec", "h264").lower()
        encoder = codec_map.get(src_codec, "libx264")

        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", temp_video_path,
            "-i", original_video_path,
            "-map", "0:v:0"
        ]

        if audio_info:
            cmd += ["-map", "1:a:0", "-c:a", "copy"]
        else:
            logger.warning("未检测到音频流，将移除音轨")
            cmd += ["-an"]

        cmd += ["-c:v", encoder]

        frame_rate = str(video_info.get("frame_rate", 30))
        cmd += ["-r", frame_rate]

        width = video_info.get("width", 1280)
        height = video_info.get("height", 720)
        cmd += ["-s", f"{width}x{height}"]

        pix_fmt = video_info.get("pix_fmt", "yuv420p")
        cmd += ["-pix_fmt", pix_fmt]

        gop = video_info.get("gop_size")
        if gop is not None:
            cmd += ["-g", str(gop)]

        profile = video_info.get("profile")
        if profile:
            cmd += ["-profile:v", profile]

        level = video_info.get("level")
        if level:
            cmd += ["-level", str(level)]

        bitrate = video_info.get("bitrate")
        if bitrate:
            cmd += ["-b:v", bitrate]
        else:
            cmd += ["-crf", "18"]

        cmd += ["-movflags", "+faststart"]

        cmd += ["-f", "mp4", output_path]

        logger.debug("执行 ffmpeg 命令: %s", " ".join(cmd))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg 转码失败: %s", e.stderr)
            return False

        logger.info("转码成功，已生成输出文件：%s", output_path)
        return True

    except Exception as e:
        logger.exception("转码异常: %s", e)
        return False
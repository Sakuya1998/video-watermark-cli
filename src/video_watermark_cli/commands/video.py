import typer
import cv2
from video_watermark_cli.core.watermark_core import VideoWatermarker, WaterMarkCore
from video_watermark_cli.utils.ffmpeg import extract_video_info,transcode_video
from video_watermark_cli.utils.qrcode import generate_qrcode, decode_qrcode
from video_watermark_cli.utils.watermark_utils import image_to_wm_bit, wm_bit_to_image, one_dim_kmeans, random_strategy1, random_strategy2
from video_watermark_cli.config import QR_SIZE, PASSWORD, FFMPEG_PATH, FFPROBE_PATH, LOG_PATH, LOG_LEVEL

import json
import os

app = typer.Typer()

def print_video_info(info: dict, output_format: str = "json") -> None:
    """
    打印视频信息，支持多种输出格式
    """
    if output_format == "json":
        typer.echo(json.dumps(info, indent=2, ensure_ascii=False))
    elif output_format == "text":
        video_info = info.get("video", {})
        audio_info = info.get("audio", {})
        
        typer.echo("视频信息:")
        typer.echo(f"  编码格式: {video_info.get('codec', 'N/A')}")
        typer.echo(f"  配置：{video_info.get('profile', 'N/A')}")
        typer.echo(f"  分辨率: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}")
        typer.echo(f"  帧率: {video_info.get('frame_rate', 'N/A')}")
        typer.echo(f"  码率: {video_info.get('bitrate', 'N/A')}")
        typer.echo(f"  像素格式: {video_info.get('pix_fmt', 'N/A')}")
        typer.echo(f"  GOP大小: {video_info.get('gop_size', 'N/A')}")
        typer.echo(f"  级别: {video_info.get('level', 'N/A')}")
        
        typer.echo("音频信息:")
        typer.echo(f"  编码格式: {audio_info.get('codec', 'N/A')}")
        typer.echo(f"  采样率: {audio_info.get('sample_rate', 'N/A')}")
        typer.echo(f"  通道数: {audio_info.get('channels', 'N/A')}")
        typer.echo(f"  码率: {audio_info.get('bitrate', 'N/A')}")

@app.command()
def info(
    input_path: str = typer.Option(..., "--input", "-i", help="输入视频文件路径")
) -> None:
    """
    显示视频文件的编码参数（分辨率、帧率、码率、音频等）
    """
    if not os.path.exists(input_path):
        typer.echo("❌ 输入视频文件不存在")
        raise typer.Exit(code=1)
    try:
        info = extract_video_info(input_path)
        typer.echo(json.dumps(info, indent=2, ensure_ascii=False))
    except Exception as e:
        typer.echo(f"❌ 获取视频信息失败: {e}")

@app.command()
def embed(
    input_path: str = typer.Option(..., "--input", "-i", help="输入视频文件路径"),
    output_path: str = typer.Option(..., "--output", "-o", help="输出视频文件路径"),
    watermark: str = typer.Option(..., "--watermark", "-w", help="水印内容"),
    start_frame: int = typer.Option(0, "--start-frame", "-s", help="开始帧号"),
    end_frame: int = typer.Option(-1, "--end-frame", "-e", help="结束帧号"),
) -> None:
    """
    嵌入水印到视频文件
    """
    if not os.path.exists(input_path):
        typer.echo("❌ 输入视频文件不存在")
        raise typer.Exit(code=1)
    if not watermark:
        typer.echo("❌ 水印内容不能为空")
        raise typer.Exit(code=1)
    try:
        os.makedirs("output", exist_ok=True)
        #  生成二维码
        qr_img = generate_qrcode(watermark, QR_SIZE) 
        qr_path = "output/qrcode.png"
        cv2.imwrite(qr_path, qr_img)
        
        #  初始化水印处理器
        wm_logo = image_to_wm_bit(qr_img)
        wm_shape = (QR_SIZE, QR_SIZE)

        video_processor = VideoWatermarker(
            password_img=PASSWORD,
            mode='common',
            processes=2
        )
        video_processor.set_watermark(wm_logo, wm_shape)
        #  嵌入水印
        temp_path = "output/temp.mp4"
        video_processor.process_video(
            input_path, 
            temp_path, 
            operation='embed',
            start_frame=start_frame, 
            end_frame=end_frame
        )
 
        time.sleep(2)   #  等待水印嵌入完成
        typer.echo("✅ 水印嵌入成功")
        #  转码
        success = transcode_video(temp_path, input_path, output_path)
        if success and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        else:
            typer.echo(f"转码失败或临时文件未删除: {temp_video_path}")
        if not success:
            return False
        typer.echo("✅ 转码成功")
        return True  

    except Exception as e:
        typer.echo(f"❌ 嵌入水印失败: {e}")
        raise typer.Exit(code=1)      

@app.command()
def extract(
    input_path: str = typer.Option(..., "--input", "-i", help="输入视频文件路径"),
    output_path: str = typer.Option(..., "--output", "-o", help="输出提取水印文件路径"),
    start_frame: int = typer.Option(0, "--start-frame", "-s", help="开始帧号"),
    end_frame: int = typer.Option(-1, "--end-frame", "-e", help="结束帧号"),    
) -> None:
    """
    从视频文件中提取水印

    """
    if not os.path.exists(input_path):
        typer.echo("❌ 输入视频文件不存在")
        raise typer.Exit(code=1)
    try:
        #  初始化水印处理器
        video_processor = VideoWatermarker(
            password_img=PASSWORD,
            mode='common',
            processes=2
        )
        wm_shape = (QR_SIZE, QR_SIZE)
        video_processor.set_watermark(None, wm_shape)

        extracted = video_processor.process_video(
            input_path,
            output_path,
            operation='extract',
            start_frame=start_frame,
            end_frame=end_frame
        )
        if extracted:
            extracted_img = wm_bit_to_image(extracted, wm_shape)
            cv2.imwrite("output/extracted.png", extracted_img)
            typer.echo("✅ 水印提取成功")
        else:
            typer.echo("❌ 水印提取失败")

        #  解码二维码        
        qr_img = cv2.imread("output/extracted.png")
        decoded_text = decode_qrcode(qr_img)

        if not decoded_text:
            typer.echo("❌ 解码二维码失败")
            raise typer.Exit(code=1)
        else:
            typer.echo(f"✅ 提取到的水印内容: {decoded_text}")

    except Exception as e:
        typer.echo(f"❌ 提取水印失败: {e}")
        raise typer.Exit(code=1)


@app.command()
def help():
    """
    显示帮助信息
    """
    typer.echo("这是一个用于提取视频水印的工具")
    typer.echo("使用方法:")
    typer.echo("  video_watermark_cli info -i <input_path>")
    typer.echo("  video_watermark_cli embed -i <input_path> -o <output_path> -w <watermark>  -s <start_frame> -e <end_frame>")
    typer.echo("  video_watermark_cli extract -i <input_path> -o <output_path> -s <start_frame> -e <end_frame>")
    typer.echo("参数说明:")
    typer.echo("  -i, --input: 输入视频文件路径")
    typer.echo("  -o, --output: 输出视频文件路径")
    typer.echo("  -w, --watermark: 水印内容")
    typer.echo("  -s, --start-frame: 开始帧号")
    typer.echo("  -e, --end-frame: 结束帧号")
    typer.echo("  -h, --help: 显示帮助信息")
    typer.echo("示例:")
    typer.echo("  video_watermark_cli info -i input.mp4")
    typer.echo("  video_watermark_cli embed -i input.mp4 -o output.mp4 -w 'hello world' -s 0 -e 100")
    typer.echo("  video_watermark_cli extract -i input.mp4 -o output.mp4 -s 0 -e 100")
    



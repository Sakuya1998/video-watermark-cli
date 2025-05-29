import typer
from video_watermark_cli.__about__ import __version__

from video_watermark_cli.commands import video
from video_watermark_cli.utils.logger import setup_logger






logger = setup_logger("video_watermark_cli")

app = typer.Typer(help="🎥 视频二维码水印 CLI 工具")

app.add_typer(video.app, name="video", help="视频水印处理")

@app.command()
def version():
    """
    显示版本信息
    """
    typer.echo(f"video_watermark_cli {__version__}")


if __name__ == "__main__":
    app()
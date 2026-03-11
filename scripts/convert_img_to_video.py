import os
import sys
from typing import Union

import ffmpeg
import argparse
from PIL import Image
from pathlib import Path

# Add parent to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_preprocessor import load_upscaler_model, preprocess_image
from utils import get_image_files


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "input_dir", type=str, help="The input directory (usually dataset/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory (usually dataset/../out)",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=16,
        help="Converts all video files to the specified framerate",
    )

    return parser


def img_to_video_with_interpolation(
    image_path: Union[str, Image.Image],
    output_path: str,
    img_width: int,
    img_height: int,
    framerate: int = 16,
    duration: int = 3,
    interpolate: bool = False,
):
    """
    Convert a single image into a video, with optional scaling and frame interpolation.

    Args:
        image_path: Path to input image
        output_path: Path to output video
        framerate: Base framerate for the video
        duration: Duration of the output video in seconds
        width: Optional width for scaling
        height: Optional height for scaling
        interpolate: Whether to apply minterpolate to reach target framerate
    """

    if isinstance(image_path, Image.Image):
        stream = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{img_width}x{img_height}",
            framerate=framerate,
        )
    else:
        stream = ffmpeg.input(image_path, loop=1, framerate=framerate)

    stream = stream.filter("scale", "trunc(iw/2)*2", "trunc(ih/2)*2")

    # Optional interpolation
    if interpolate:
        stream = stream.filter("minterpolate", fps=framerate)

    # Output video
    (
        stream.output(output_path, vcodec="libx264", pix_fmt="yuv420p", t=duration)
        .overwrite_output()
        .run()
    )


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    framerate = args.framerate

    if framerate < 1:
        raise ValueError("Framerate cannot be smaller than 1")

    print(f"Scanning for images in {input_dir}...")
    files = get_image_files(input_dir)

    if not files:
        print("No image files found in the source directory.")
        sys.exit(1)

    upscale_processor, upscale_model = load_upscaler_model()

    for file in files:
        output_dir_path = str(file.parent / "out") if output_dir is None else output_dir
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
        output = (output_dir_path + "/" + file.stem + ".mp4").replace("\\", "/")

        img = Image.open(str(file)).convert("RGB")
        img = preprocess_image(
            img,
            upscale_processor,
            upscale_model,
        )
        img_to_video_with_interpolation(
            image_path=str(file),
            output_path=output,
            framerate=framerate,
            img_width=img.width,
            img_height=img.height,
        )


if __name__ == "__main__":
    main()

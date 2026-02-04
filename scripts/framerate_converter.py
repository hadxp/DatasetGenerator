import os
import sys
import ffmpeg
import argparse
from pathlib import Path

# Add parent to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_video_files

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("directory", help="The operating directory")
    parser.add_argument(
        "--framerate", type=int, default=24, help="Converts all video files to 30 FPS"
    )

    return parser


def interpolate_and_scale(input: str, output: str, framerate: int, width: int = 640, height: int = 640):
    """
    Interpolate video to target framerate and optionally scale.

    Args:
        input: Input video file path
        output: Output video file path
        framerate: Target frames per second
        width: Target width (optional)
        height: Target height (optional)
    """
    stream = ffmpeg.input(input)

    # Apply scaling if dimensions provided
    if width is not None or height is not None:
        # Use scale filter
        if width is not None and height is not None:
            # Both dimensions specified
            stream = stream.filter('scale', width, height)
        elif width is not None:
            # Scale to width, keep aspect ratio
            stream = stream.filter('scale', width, -1)
        elif height is not None:
            # Scale to height, keep aspect ratio
            stream = stream.filter('scale', -1, height)

    # Apply interpolation
    stream = stream.filter('minterpolate', fps=framerate)

    # Output
    stream.output(output).overwrite_output().run()


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    source_dir = Path(args.directory)
    framerate = args.framerate

    if framerate < 1:
        raise ValueError("Framerate cannot be smaller than 1")

    print(f"Scanning for images in {source_dir}...")
    files = get_video_files(source_dir)

    if not files:
        print("No video files found in the source directory.")
        sys.exit(1)

    for file in files:
        output_dir_path = str(file.parent)
        output = (output_dir_path + "/" + file.stem + "_" + file.suffix).replace(
            "\\", "/"
        )
        interpolate_and_scale(input=str(file), output=output, framerate=framerate)


if __name__ == "__main__":
    main()

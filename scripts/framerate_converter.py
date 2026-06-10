import sys
import ffmpeg
import argparse
from pathlib import Path
from utils import get_video_files


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("directory", help="The operating directory")
    parser.add_argument(
        "--framerate", type=int, default=19, help=""
    )
    parser.add_argument(
        "--duration", type=int, default=24, help=""
    )

    return parser


def interpolate_and_scale(
    input: str,
    output: str,
    max_width: int = 512,
    max_height: int = 512,
    framerate: int = 19,
):
    """
    Interpolate video to target framerate and optionally scale.

    Args:
        input: Input video file path
        output: Output video file path
        max_width: Target width (optional)
        max_height: Target height (optional)
        framerate: Target frames per second
    """
    stream = ffmpeg.input(input)

    # force_original_aspect_ratio="decrease" ensures the video fits inside 
    # the max_width/max_height box without ever upscaling or stretching.
    # We then use setsar to ensure the pixel aspect ratio remains standard.
    stream = stream.filter(
        "scale",
        f"min({max_width},iw)",
        f"min({max_height},ih)",
        force_original_aspect_ratio="decrease"
    ).filter("setsar", "1/1")

    # Ensure final dimensions are divisible by 2 (required for many codecs)
    stream = stream.filter("scale", "trunc(iw/2)*2", "trunc(ih/2)*2")

    # Apply interpolation
    stream = stream.filter("minterpolate", fps=framerate)

    # Output
    stream.output(output).overwrite_output().run(quiet=True)


def interpolate_and_fix_duration(
    input_path: str,
    output_path: str,
    framerate: int = 19,
    duration: int = 3
):
    # 1. Use stream_loop on input to ensure there is plenty of data
    input_node = ffmpeg.input(input_path, stream_loop=10)
    video = input_node.video

    # 2. Scale for codec safety (even dimensions)
    video = video.filter('scale', 'iw', 'ih', force_divisible_by=2)

    # 3. Apply interpolation
    # We do this before the trim to ensure smooth transitions between loops
    video = video.filter("minterpolate", fps=framerate)

    # 4. Force the output to be exactly 3 seconds
    # 'r=framerate' forces the output header to match the interpolated rate
    # 't=duration' is the hard cutoff for the file
    (
        video
        .output(
            output_path, 
            an=None, 
            t=duration, 
            r=framerate,
            vcodec='libx264',
            pix_fmt='yuv420p' # Ensures compatibility with all players
        )
        .overwrite_output()
        .run(quiet=True)
    )


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    source_dir = Path(args.directory)
    framerate = args.framerate
    duration = args.duration

    if framerate < 1:
        raise ValueError("Framerate cannot be smaller than 1")

    print(f"Scanning for videos in {source_dir}...")
    files = get_video_files(source_dir)

    if not files:
        print("No video files found in the source directory.")
        sys.exit(1)

    # sort files
    sorted_files = sorted(files, key=lambda x: int(x.stem.split('_')[0]))
    for file in sorted_files:
        output_dir_path = str(file.parent)
        output = (output_dir_path + "/out/" + file.stem + file.suffix).replace(
            "\\", "/"
        )
        print(f"Processing {file.stem} save to {output}...")
        #interpolate_and_scale(input=str(file), output=output, framerate=framerate)
        interpolate_and_fix_duration(input_path=str(file), output_path=output, framerate=framerate, duration=duration)


if __name__ == "__main__":
    main()

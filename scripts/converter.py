import os
import sys
import json
import argparse
from typing import List
from pathlib import Path

# Add parent to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    image_extensions,
    video_extensions,
    caption_extensions,
    get_image_files,
    get_video_files,
    SaveImageEntry,
    SaveVideoEntry,
)


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("directory", help="The operating directory")
    parser.add_argument(
        "--jsonl",
        action="store_true",
        default=True,
        help="Converts all files to jsonl format",
    )
    parser.add_argument(
        "--img_and_caption",
        action="store_true",
        default=False,
        help="Converts all files from jsonl to file with caption format",
    )

    return parser


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    source_dir = Path(args.directory)
    convert_to_jsonl = args.jsonl
    if args.img_and_caption:
        convert_to_jsonl = False

    print(f"Scanning for images in {source_dir}...")
    files = get_image_files(source_dir)
    if len(files) <= 0:
        files = get_video_files(source_dir)

    if not files:
        print("No image or video files found in the source directory.")
        sys.exit(1)

    jsonl_path = source_dir / "0_dataset.jsonl"

    if jsonl_path.exists() and convert_to_jsonl is True:
        print("jsonl file already exists")
        return

    results: List[SaveImageEntry | SaveVideoEntry] = []

    for file_path in files:
        if convert_to_jsonl:
            is_video = file_path.suffix in video_extensions
            is_image = file_path.suffix in image_extensions
            caption_file_path = file_path.with_suffix(caption_extensions[0])
            file_path_str = str(file_path).replace("\\", "/")
            with open(caption_file_path, "r", encoding="utf-8") as f:
                content = f.read().replace("\n", "")
                if is_image:
                    results.append(
                        SaveImageEntry(
                            image_path=file_path_str,
                            control_path=file_path_str,
                            caption=content,
                        )
                    )
                elif is_video:
                    results.append(
                        SaveVideoEntry(
                            video_path=file_path_str,
                            control_path=file_path_str,
                            caption=content,
                        )
                    )
        else:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

    if convert_to_jsonl is True and results:
        # create jsonl
        print(f"Writing {len(results)} results to JSONL: {jsonl_path}")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            dump = "\n".join(
                json.dumps(result, ensure_ascii=False) for result in results
            )
            f.write(dump)
    if convert_to_jsonl is False and results:
        for result in results:
            # the image must already exist
            file = result["file"]
            caption = result["caption"]
            filename = os.path.splitext(os.path.basename(file))[0]
            caption_file_path = source_dir / f"{filename}.txt"
            if not caption_file_path.exists():
                with open(caption_file_path, "w", encoding="utf-8") as f:
                    f.write(caption)


if __name__ == "__main__":
    main()

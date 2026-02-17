"""
DatasetGeneration Script
including image preprocessing via the "mohsin-riad/upscaler-ultra" upscaler model and
Florence-2 for caption generation, text replacement with triggerword in generated caption,

usage:
python main.py [dataset_names]
"""

import os
import sys
import json
import argparse

import transformers
from PIL import Image
from packaging.version import Version
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

from caption_generator import (
    generate_caption,
    process_caption_text,
    load_cation_model_florence2,
    load_caption_model_qwen3,
)
from parquet import create_parquet, upload_to_hf
from utils import (
    ResultEntry,
    get_image_files,
    get_video_files,
    save,
    image_extensions,
    video_extensions,
)
from image_preprocessor import load_upscaler_model, preprocess_image
from VideoFrameExtractor import VideoFrameExtractor, VideoInfo


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhance, scale and generate captions (using Florence2) for images"
    )
    parser.add_argument(
        "dataset",
        type=str,
        default=None,
        help="The name of the folder the dataset(s) to parse is in (comma separated)",
    )
    parser.add_argument(
        "--triggerword",
        type=str,
        default="",
        help="The Triggerword to replace gender terms in generated captions",
    )
    parser.add_argument(
        "--task",
        default="more_detailed_caption",
        choices=[
            "region_caption",
            "dense_region_caption",
            "region_proposal",
            "caption",
            "detailed_caption",
            "more_detailed_caption",
            "caption_to_phrase_grounding",
            "referring_expression_segmentation",
            "ocr",
            "ocr_with_region",
            "docvqa",
            "prompt_gen_tags",
            "prompt_gen_mixed_caption",
            "prompt_gen_analyze",
            "prompt_gen_mixed_caption_plus",
        ],
        help="Task type for Florence-2 model",
    )
    parser.add_argument(
        "--text-input",
        default="",
        help="Additional text input for specific tasks in caption generation",
    )
    parser.add_argument(
        "--jsonl", action="store_true", default=False, help="Enable jsonl generation"
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        default=False,
        help="Enable parquet generation (no images or captions will be saved)",
    )
    parser.add_argument(
        "--huggingface_repoid",
        type=str,
        default="hadxp/datasets",
        help="Huggingface repoid to upload the parquet file",
    )
    parser.add_argument(
        "--huggingface_token", type=str, default=None, help="Huggingface token"
    )
    parser.add_argument(
        "--search_dir",
        type=str,
        default=None,
        help="The folder to search the datasets in",
    )
    parser.add_argument(
        "--no_check", action="store_true", default=False, help="Skips gen.txt check"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help='Replaces the default prompt, words like "{prompt}" or "{template}" or "{person_template}" will be replaced',
    )
    return parser


video: Tuple[List[Image.Image], VideoInfo] = None
img: Image.Image = None


def main():
    global video
    global img

    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # get the arguments
    trigger_word: str = args.triggerword.strip()
    task: str = args.task
    text_input: str = args.text_input
    jsonl: bool = args.jsonl
    parquet: bool = args.parquet
    huggingface_repoid: str = args.huggingface_repoid
    huggingface_token: str = args.huggingface_token
    dataset_names_arg: str = args.dataset
    search_dir: str = args.search_dir
    no_check: bool = args.no_check
    prompt: bool = args.prompt

    can_upload_to_huggingface = (
        huggingface_token is not None and huggingface_repoid != ""
    )

    if dataset_names_arg is None:
        print("No dataset(s) specified, cannot continue")
        sys.exit(1)

    if search_dir is None:
        datasets_path = Path.cwd() / "datasets"
    else:
        datasets_path = Path(search_dir)

    dataset_names_arg_arr = args.dataset.split(",")

    print(f"Using dataset search dir: {datasets_path}")

    for dataset_name in dataset_names_arg_arr:
        if not datasets_path.exists():
            datasets_path.mkdir()
            sys.exit(0)

        dataset_name_folder: Path = None

        for folder in datasets_path.iterdir():
            if folder.is_dir() and dataset_name in folder.name:
                dataset_name_folder = folder

        dataset_dir = datasets_path / dataset_name_folder

        print(f"Dataset folder is: {dataset_name_folder}")

        source_dir = dataset_dir / "dataset"  # set the source image directory path

        if not source_dir.exists():
            print(f"Error: Source dataset directory '{source_dir}' does not exist.")
            sys.exit(1)

        # "out" as target_dir (if one is needed)
        target_dir = dataset_dir / "out"

        # Get image files
        print(f"Scanning for images in {source_dir}...")
        files = get_image_files(source_dir)
        if len(files) <= 0:
            files = get_video_files(source_dir)

        if not files:
            print("No image or video files found in the source directory.")
            sys.exit(1)

        # check the generation file

        generation_file_path = target_dir / "gen.txt"

        regernerate_dataset = True

        if not no_check:
            if generation_file_path.exists():
                # read the file
                with open(generation_file_path, "r", encoding="utf-8") as f:
                    saved_count = int(f.read().strip())
                # check if its content equals the number of images -> if yes skip processing
                if saved_count == len(files):
                    regernerate_dataset = False
            else:
                # create the file
                with open(generation_file_path, "w", encoding="utf-8") as f:
                    f.write(str(len(files)))

        # regenerate the dataset if needed
        if regernerate_dataset:
            print(f"Found {len(files)} files.")
            # print(f"Using task: {args.task}")
            # print(f"Using trigger word: {trigger_word}")
            if args.text_input:
                print(f"Using text input: {args.text_input}")

            results: List[ResultEntry] = []

            print("\nStarting image / video processing and caption generation...")

            version = Version(transformers.__version__)

            # Load caption model
            if version <= Version("4.53.1"):
                model, processor = load_cation_model_florence2()
            elif version >= Version("4.57.6"):
                model, processor = load_caption_model_qwen3()
            else:
                raise Exception(
                    f"No valid transformers version present, cannot load caption model, version was {version} valid versions are 4.53.1 or 4.57.6"
                )

            # Load upscale model
            upscale_processor, upscale_model = load_upscaler_model()

            successful_processing = 0
            for i, file_path in enumerate(tqdm(files), 1):
                is_video = False
                is_image = False
                if file_path.suffix in video_extensions:
                    is_video = True
                if file_path.suffix in image_extensions:
                    is_image = True

                if not is_video and not is_image:
                    print(f"Skipping file {file_path} - not a image or video file")
                    continue

                target_file_path = (
                    target_dir / file_path.name
                )  # target_dir + filepath with extension

                # load the source
                if is_video:
                    from scripts.framerate_converter import interpolate_and_scale

                    if not target_file_path.exists():
                        target_file_path_str = str(target_file_path)
                        interpolate_and_scale(
                            file_path, target_file_path_str, framerate=16
                        )
                    vfe = VideoFrameExtractor(
                        target_file_path,
                        upscale_model,
                        upscale_processor,
                    )
                    video = vfe.get_video()
                elif is_image:
                    img = Image.open(file_path).convert("RGB")
                    img = preprocess_image(
                        img,
                        upscale_processor,
                        upscale_model,
                    )

                # Generate caption
                caption = generate_caption(
                    model=model,
                    processor=processor,
                    source_object=video if is_video else img if is_image else None,
                    task=task,
                    text_input=text_input,
                    prompt=prompt,
                )

                if caption:
                    # Process caption with trigger word replacement
                    if trigger_word != "":
                        caption = process_caption_text(caption, trigger_word)

                    result_entry: ResultEntry = {
                        "target_file_path": target_file_path,
                        "video": video,
                        "image": img,
                        "control": img,
                        "caption": caption,
                    }

                    results.append(result_entry)
                    successful_processing += 1
                    # print(f"  ✓ Processed")
                else:
                    print(f"  ✗ Failed to generate caption for {file_path}")

            write_task_handled: bool = False

            if can_upload_to_huggingface and write_task_handled is False:
                write_task_handled = True
                parquet_path = create_parquet(dataset_dir, results)
                upload_to_hf(parquet_path, huggingface_repoid, huggingface_token)

            if parquet and write_task_handled is False:
                write_task_handled = True
                create_parquet(dataset_dir, results)

            # save_upscaled_img needs to be true for jsonl, since the jsonl references it
            if jsonl and write_task_handled is False:
                write_task_handled = True
                if results:
                    print(f"Saving {len(results)} images to: {target_dir}")
                    # save the image files in the results list
                    save_entries = save(results, target_dir)
                    # create jsonl
                    jsonl_path = target_dir / "0_dataset.jsonl"
                    print(f"Writing {len(results)} results to JSONL: {jsonl_path}")
                    with open(jsonl_path, "w", encoding="utf-8") as f:
                        dump = "\n".join(
                            json.dumps(save_entry, ensure_ascii=False)
                            for save_entry in save_entries
                        )
                        f.write(dump)

            if not write_task_handled:
                if results:
                    # save the image files in the results list
                    save_entries = save(results, target_dir)
                    # create text files
                    print(
                        f"Creating text (caption) files from {len(results)} results..."
                    )
                    for save_entry in save_entries:
                        text_filename = (
                            Path(
                                save_entry["image_path"]
                                if "image_path" in save_entry
                                else save_entry["video_path"]
                            ).stem
                            + ".txt"
                        )
                        text_filepath = os.path.join(target_dir, text_filename)
                        Path(text_filepath).write_text(
                            save_entry["caption"], encoding="utf-8"
                        )
                        print(f"  Created caption file: {text_filename}")

            # Print summary
            print("\nProcessing complete!")
            print(f"Successfully processed {successful_processing}/{len(files)} files")


if __name__ == "__main__":
    main()

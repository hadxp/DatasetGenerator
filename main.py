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
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

import utils
from utils import (
    ResultEntry,
    get_image_files,
    get_video_files,
    save,
    video_extensions,
    split_list_into_batches,
)
from caption_generator import (
    batch_generate_captions,
    load_caption_model_qwen3,
    process_caption_text,
    generate_caption_prompt,
)
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from parquet import create_parquet, upload_to_hf
from image_preprocessor import preprocess_image
from VideoFrameExtractor import VideoFrameExtractor, VideoInfo
from scripts.framerate_converter import interpolate_and_scale

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
        "--prompt",
        type=str,
        default="",
        help='Replaces the default prompt, words like "{prompt}" or "{template}" or "{person_template}" will be replaced',
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        default=False,
        help="Shows the prompt before generating",
    )
    parser.add_argument(
        "--person_lora",
        action="store_true",
        default=True,
        help="Determines if the generated caption should contain a person description (not in caption = learn)",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="",
        help="A token which is already known by the model, to properly associate the triggerword (eg. if your image shows a girl, the class prompt will be girl)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="The batch size for caption generation (default: 1)",
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
    triggerword: str = args.triggerword.strip()
    jsonl: bool = args.jsonl
    parquet: bool = args.parquet
    huggingface_repoid: str = args.huggingface_repoid
    huggingface_token: str = args.huggingface_token
    dataset_names_arg: str = args.dataset
    search_dir: str = args.search_dir
    prompt: str = args.prompt
    show_prompt: bool = args.show_prompt
    person_lora: bool = args.person_lora
    class_prompt: str = args.class_prompt
    batch_size: int = args.batch

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

        if dataset_name_folder is None:
            raise Exception(f"No dataset named '{dataset_name}' found in directory '{datasets_path}'")

        dataset_dir = datasets_path / dataset_name_folder

        print(f"Dataset folder is: {dataset_name_folder}")

        source_dir = dataset_dir / "dataset"  # set the source image directory path

        if not source_dir.exists():
            print(f"Error: Source dataset directory '{source_dir}' does not exist.")
            sys.exit(1)

        # "out" as target_dir (if one is needed)
        target_dir = dataset_dir / "out"

        # Get image files
        print(f"Scanning for files in {source_dir}...")
        image_files = get_image_files(source_dir)
        video_files = get_video_files(source_dir)
            
        files: List[Path] = []
        
        if len(image_files) > 0 and len(video_files) > 0:
            print("Cannot have mix image and video files in one dataset")
            sys.exit(1)
        elif len(image_files) > 0:
            files = image_files
        elif len(video_files) > 0:
            files = get_video_files(source_dir)
        
        is_video_dataset = len(video_files) > 0
            
        if not files:
            print("No image or video files found in the source directory.")
            sys.exit(1)
            
        print(f"Found {len(files)} files.")

        print("\nStarting image / video processing and caption generation...")

        model, processor = load_caption_model_qwen3()

        # set prompt to generate the caption
        prompt = generate_caption_prompt(
            prompt,
            triggerword = triggerword if triggerword else "ohwx",
            class_prompt = class_prompt if class_prompt else "person" if person_lora else None,
            person_lora = person_lora,
        )

        if show_prompt:
            print("-----------------------------------")
            print(prompt)
            print("-----------------------------------")
            
        results: List[ResultEntry] = []
        
        print("Proccessing all files...", end="")
        
        # sort files
        sorted_files = sorted(files, key=lambda x: int(x.stem.split('_')[0]))
        for image_index, file_path in enumerate(sorted_files):
            is_video, file_path_in_target_dir, media = process_media_file(target_dir, file_path)
            result_entry: ResultEntry = {
                "file_path_in_target_dir": file_path_in_target_dir,
                "video": media if is_video else None,
                "image": media if not is_video else None,
                "control": media if not is_video else None,
                "caption": None,
            }
            results.append(result_entry)
            
        print("DONE")
        
        print(f"Generating captions with batch {batch_size}...")
        batch_caption_generation(results, model, processor, prompt, is_video_dataset, batch_size)
        
        ex: bool = False
        
        for result_entry in results:
            caption = result_entry["caption"]
            if caption is None or caption == "":
                print(f"No caption found for '{result_entry['file_path_in_target_dir'].stem}'")
                
        if ex:
            sys.exit(0)

        write_captions(dataset_dir, target_dir, results, huggingface_repoid, huggingface_token, parquet, jsonl)
        

def process_media_file(target_dir: Path, file_path: Path) -> Tuple[bool, str, Tuple[List[np.ndarray], VideoInfo]] | Tuple[bool, str, Image.Image]:
    is_video = True if file_path.suffix.lower() in video_extensions else False

    file_path_in_target_dir: Path = (
        target_dir / file_path.name
    )  # target_dir + filepath with extension

    # load the source
    if is_video:
        if not file_path_in_target_dir.exists():
            os.makedirs(target_dir, exist_ok=True)
            target_file_path_str = str(file_path_in_target_dir)
            interpolate_and_scale(
                file_path, target_file_path_str, framerate=19
            )
        vfe = VideoFrameExtractor(
            file_path=file_path_in_target_dir,
            # upscale=False,
            #upsample=True,
        )
        video = vfe.get_video()
        return is_video, file_path_in_target_dir, video
    elif not is_video:
        img = Image.open(file_path).convert("RGB")
        img = preprocess_image(
            img,
            upscale=True,
            upsample=True,
        )
        return is_video, file_path_in_target_dir, img
    else:
        print(f"ERROR file {file_path} - not an image or video")
        sys.exit(0)
        
def batch_caption_generation(results: List[ResultEntry], model: Qwen3VLForConditionalGeneration, processor: Qwen3VLProcessor, prompt: str, is_video_dataset: bool, batch_size: int) -> None:
    batches = split_list_into_batches(results, batch_size)

    total_batches = len(results) // batch_size + (1 if len(results) % batch_size != 0 else 0)
    
    for idx, batch in tqdm(enumerate(batches), total=total_batches):
        # Generate caption (after "generate_caption" returns, the "caption" field in each resultentry in the results, will be filled)
        batch_generate_captions(
            model=model,
            processor=processor,
            results=batch,
            prompt=prompt,
            is_video_dataset=is_video_dataset,
        )
        

def write_captions(dataset_dir: Path, target_dir: Path, results: List[ResultEntry], huggingface_repoid: str, huggingface_token: str, parquet: bool, jsonl: bool):
    can_upload_to_huggingface = (
        huggingface_token is not None and huggingface_repoid != ""
    )
    
    if can_upload_to_huggingface:
        parquet_path = create_parquet(dataset_dir, results)
        upload_to_hf(parquet_path, huggingface_repoid, huggingface_token)
        
    elif parquet:
        create_parquet(dataset_dir, results)

    elif jsonl:
        if results:
            # save the image files in the results list
            save_entries = save(results, target_dir)
            # create jsonl
            jsonl_path = target_dir / "0_dataset.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                dump = "\n".join(
                    json.dumps(save_entry, ensure_ascii=False)
                    for save_entry in save_entries
                )
                f.write(dump)
            print(f"Wrote {len(results)} results to JSONL: {jsonl_path}")

    else:
        if results:
            # save the image files in the results list
            save_entries = save(results, target_dir)
            # create text files
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
        
if __name__ == "__main__":
    main()

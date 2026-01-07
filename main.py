"""
DatasetGeneration Script
including image preprocessing via the "mohsin-riad/upscaler-ultra" upscaler model and
Florence-2 for caption generation, text replacement with triggerword in generated caption,

usage:
python main.py [triggerword] [dataset_names]
"""

import os
import sys
import json
import argparse

from tqdm import tqdm
from pathlib import Path
from typing import List
from caption_generator import load_cation_model, generate_caption, process_caption_text
from create import create_parquet, upload_to_hf
from utils import ResultEntry, InMemoryResultEntry, get_image_files
from image_preprocessor import load_upscaler_model, preprocess_image

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Enhance, scale and generate captions (using Florence2) for images'
    )
    parser.add_argument('triggerword', help='The Triggerword to replace gender terms in generated captions')
    parser.add_argument('--task', default='more_detailed_caption',
                        choices=[
                            'region_caption', 'dense_region_caption', 'region_proposal',
                            'caption', 'detailed_caption', 'more_detailed_caption',
                            'caption_to_phrase_grounding', 'referring_expression_segmentation',
                            'ocr', 'ocr_with_region', 'docvqa', 'prompt_gen_tags',
                            'prompt_gen_mixed_caption', 'prompt_gen_analyze', 'prompt_gen_mixed_caption_plus'
                        ],
                        help='Task type for Florence-2 model')
    parser.add_argument('--text-input', default='', help='Additional text input for specific tasks in caption generation')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Maximum new tokens for caption generation')
    parser.add_argument('--num-beams', type=int, default=3, help='Number of beams for caption generation')
    parser.add_argument('--imgwidth', type=int, default=1024, help='The width the images are scaled to')
    parser.add_argument('--imgheight', type=int, default=1024, help='The height the images are scaled to')
    parser.add_argument('--jsonl', action='store_true', default=False, help='Enable jsonl generation')
    parser.add_argument('--parquet', action='store_true', default=False, help='Enable parquet generation (no images or captions will be saved)')
    parser.add_argument('--huggingface_repoid', type=str, default="hadxp/datasets", help='Huggingface repoid to upload the parquet file')
    parser.add_argument('--huggingface_token', type=str, default=None, help='Huggingface token')
    parser.add_argument('dataset', type=str, default=None, help='The name of the folder the dataset(s) to parse is in (comma separated)')
    parser.add_argument('--search_dir', type=str, default=None, help='The folder to search the datasets in')
    
    return parser

def main():
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    trigger_word: str = args.triggerword.strip()
    task: str = args.task
    text_input: str = args.text_input
    max_new_tokens: int = args.max_new_tokens
    num_beams: int = args.num_beams
    imgwidth: str = args.imgwidth
    imgheight: str = args.imgheight
    jsonl: bool = args.jsonl
    parquet: bool = args.parquet
    huggingface_repoid: str = args.huggingface_repoid
    huggingface_token: str = args.huggingface_token
    dataset_names_arg: str = args.dataset
    search_dir: str = args.search_dir
    
    can_upload_to_huggingface = huggingface_token is not None and huggingface_repoid != ""
    save_upscaled_img = parquet is False and not can_upload_to_huggingface
    
    if not trigger_word:
        print("Error: Trigger word cannot be empty.")
        sys.exit(1)
    
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

        source_dir = dataset_dir / "dataset" # set the source image directory path
        
        if not source_dir.exists():
            print(f"Error: Source dataset directory '{source_dir}' does not exist.")
            sys.exit(1)
        
        # "out" as target_dir (if one is needed)
        target_dir = dataset_dir / "out"
        
        # create the directory
        target_dir.mkdir(parents=True, exist_ok=True)  # Create target directory if it does not exist
    
        # Get image files
        print(f"Scanning for images in {source_dir}...")
        image_files = get_image_files(source_dir)
    
        if not image_files:
            print("No image files found in the source directory.")
            sys.exit(1)

        # check the generation file

        generation_file_path = target_dir / "gen.txt"

        regernerate_dataset = True

        if generation_file_path.exists():
            # read the file
            with open(generation_file_path, "r", encoding="utf-8") as f:
                saved_count = int(f.read().strip())
            # check if its content equals the number of images -> if yes skip procesing
            if saved_count == len(image_files):
                regernerate_dataset = False
        else:
            # create the file
            with open(generation_file_path, "w", encoding="utf-8") as f:
                f.write(str(len(image_files)))
    
        # regenerate the dataset if needed
        if regernerate_dataset:
            print(f"Found {len(image_files)} image files.")
            print(f"Using task: {args.task}")
            print(f"Using trigger word: {trigger_word}")
            if args.text_input:
                print(f"Using text input: {args.text_input}")
        
            results: List[ResultEntry] | List[InMemoryResultEntry] = []
            jsonl_path = target_dir / "0_dataset.jsonl"
        
            print(f"\nStarting image processing and caption generation...")

            # Load caption model
            model, processor = load_cation_model()

            # Load upscale upscale model
            upscale_processor, upscale_model = load_upscaler_model()
        
            successful_processing = 0
            for i, image_path in enumerate(tqdm(image_files), 1):
                #print(f"Processing {i}/{len(image_files)}: {image_path.name}")
        
                target_filename = f"{i}.png"
        
                # path of the copied image
                target_path = target_dir / target_filename
                
                processed_img = preprocess_image(image_path, upscale_processor, upscale_model, imgwidth, imgheight)
                
                if save_upscaled_img:
                    # Check if the target image already exists
                    if target_path.exists():
                        #print(f"  Warning: {target_filename} already exists, skipping copy")
                        pass
                    else:
                        # Save as PNG
                        processed_img.save(target_path, 'PNG')
                        #print(f"  ✓ Copied image {target_path.name}")
        
                # Generate caption
                caption = generate_caption(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    task=task,
                    text_input=text_input,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams
                )
        
                if caption:
                    # Process caption with trigger word replacement
                    processed_caption = process_caption_text(caption, trigger_word)
                    
                    if save_upscaled_img:
                        # Create (path) result entry
                        result_entry: ResultEntry = {
                            "image_path": str(target_path),
                            "control_path": str(target_path),
                            "caption": processed_caption,
                        }
                    else:
                        # Create (image) result entry
                        result_entry: InMemoryResultEntry = {
                            "image": processed_img,
                            "control_image": processed_img,
                            "caption": processed_caption,
                        }
        
                    results.append(result_entry)
                    successful_processing += 1
                    #print(f"  ✓ Processed")
                else:
                    #print(f"  ✗ Failed to generate caption for {image_path.name}")
                    pass
        
            write_task_handled: bool = False
        
            if can_upload_to_huggingface and not save_upscaled_img and write_task_handled is False:
                write_task_handled = True
                parquet_path = create_parquet(dataset_dir, results)
                upload_to_hf(parquet_path, huggingface_repoid, huggingface_token)
                
            if parquet and not save_upscaled_img and write_task_handled is False:
                write_task_handled = True
                create_parquet(dataset_dir, results)
                
            # save_upscaled_img needs to be true for jsonl, since the jsonl references it
            if jsonl and save_upscaled_img and write_task_handled is False:
                write_task_handled = True
                if results:
                    print(f"\nWriting {len(results)} results to JSONL: {jsonl_path}")
                    with open(jsonl_path, 'w', encoding='utf-8') as f:
                        dump = '\n'.join(json.dumps(result, ensure_ascii=False) for result in results)
                        f.write(dump)
            
            if not write_task_handled and save_upscaled_img is True:
                # normal write operation create text files
                print("\nCreating text (caption) files from results...")
                for i, result in enumerate(results, 1):
                    text_filename = f"{i}.txt"
                    text_filepath = os.path.join(target_dir, text_filename)
                    Path(text_filepath).write_text(result["caption"], encoding='utf-8')
                    print(f"  Created caption file: {text_filename}")
        
            # Print summary
            print(f"\nProcessing complete!")
            print(f"Successfully processed {successful_processing}/{len(image_files)} images")

if __name__ == "__main__":
    main()

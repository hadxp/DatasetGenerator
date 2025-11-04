"""
DatasetGeneration Script
including image preprocessing via the "mohsin-riad/upscaler-ultra" upscaler model and
Florence-2 for caption generation, text replacement with triggerword in generated caption,

usage:
python main.py [source_dir] [target_dir] [triggerword]
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from diffusers import DiffusionPipeline 
from transformers import AutoModelForCausalLM, AutoProcessor, Swin2SRForImageSuperResolution, Swin2SRImageProcessor
import numpy as np

import torch
# These imports are used by the florence2 model, and kept for completeness, as the code would crash without them being avaliable
import transformers 
import accelerate
import einops
import timm

torch_dtype = torch.float32
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Enhance, scale and generate captions (using Florence2) for images'
    )
    parser.add_argument('source_dir', help='Source directory containing images')
    parser.add_argument('target_dir', help='Target directory to save processed images and captions')
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
    parser.add_argument('--no_jsonl', type=bool, default=False, help='Disable jsonl generation')
    
    return parser


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    return sorted(image_files)


def get_task_prompt(task: str) -> str:
    """Get the appropriate prompt for the given task."""
    prompts = {
        'region_caption': '<OD>',
        'dense_region_caption': '<DENSE_REGION_CAPTION>',
        'region_proposal': '<REGION_PROPOSAL>',
        'caption': '<CAPTION>',
        'detailed_caption': '<DETAILED_CAPTION>',
        'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
        'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
        'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
        'ocr': '<OCR>',
        'ocr_with_region': '<OCR_WITH_REGION>',
        'docvqa': '<DocVQA>',
        'prompt_gen_tags': '<GENERATE_TAGS>',
        'prompt_gen_mixed_caption': '<MIXED_CAPTION>',
        'prompt_gen_analyze': '<ANALYZE>',
        'prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
    }

    if task not in prompts:
        raise ValueError(f"Unknown task: {task}. Available tasks: {list(prompts.keys())}")

    return prompts[task]


def validate_task_input(task: str, text_input: str) -> None:
    """Validate if text input is allowed for the given task."""
    if text_input and task not in ['referring_expression_segmentation', 'caption_to_phrase_grounding', 'docvqa']:
        raise ValueError(
            "Text input is only supported for 'referring_expression_segmentation', "
            "'caption_to_phrase_grounding', and 'docvqa' tasks"
        )


def generate_caption(
        model,
        processor,
        image_path: Path,
        task: str = "more_detailed_caption",
        text_input: str = "",
        max_new_tokens: int = 256,
        num_beams: int = 3
) -> Optional[str]:
    """Generate caption for an image using Florence2 model."""

    # Validate task and input
    validate_task_input(task, text_input)
    task_prompt = get_task_prompt(task)

    # Construct final prompt
    if text_input:
        prompt = f"{task_prompt} {text_input}"
    else:
        prompt = task_prompt

    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Process image and text
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        # Move all inputs to the same device and data type as model
        model_dtype = next(model.parameters()).dtype

        processed_inputs = {}
        for key, value in inputs.items():
            if value.dtype.is_floating_point:
                processed_inputs[key] = value.to(device=torch_device, dtype=model_dtype)
            else:
                processed_inputs[key] = value.to(device=torch_device)

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=processed_inputs["input_ids"],
                pixel_values=processed_inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

        # Decode the generated text
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Clean up the generated text - remove the task prompt if present
        if generated_text.startswith(task_prompt):
            generated_text = generated_text[len(task_prompt):].strip()

        # Remove any remaining special tokens
        generated_text = generated_text.replace("<|endoftext|>", "").strip()

        return generated_text

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_caption_text(caption: str, trigger_word: str) -> str:
    """Process and clean the caption text with trigger word replacement."""
    if not caption:
        return caption

    # Gender term replacements
    replacements = {
        "woman": trigger_word,
        "man": trigger_word,
        "female": trigger_word,
        "male": trigger_word,
        "lady": trigger_word,
        "gentleman": trigger_word,
        "girl": trigger_word,
        "boy": trigger_word,
    }

    pronoun_replacements = {
        " the ": f" the {trigger_word} ", # Added spaces for safety
        " The ": f" The {trigger_word} ", 
        " she ": f" the {trigger_word} ", 
        " She ": f" The {trigger_word} ", 
        " her ": f" the {trigger_word}'s ", 
        " Her ": f" The {trigger_word}'s ", 
        " hers ": f" the {trigger_word}'s ", 
        " Hers ": f" The {trigger_word}'s ", 
        " he ": f" the {trigger_word} ", 
        " He ": f" The {trigger_word} ", 
        " him ": f" the {trigger_word} ", 
        " Him ": f" The {trigger_word} ", 
        " his ": f" the {trigger_word}'s ", 
        " His ": f" The {trigger_word}'s ", 
        f" {trigger_word}{trigger_word} ": f" {trigger_word} ", # Cleanup for double trigger words
        " leatthe ": " leather ",
    }

    # Apply gender term replacements first
    processed_caption = caption
    for old, new in replacements.items():
        processed_caption = processed_caption.replace(old, new)

    # Apply pronoun replacements
    for old, new in pronoun_replacements.items():
        processed_caption = processed_caption.replace(old, new)

    # Remove portrait-related phrases
    portrait_phrases = [
        "portrait of a",
        "portrait of the",
        "portrait of",
        "portrait",
        "photo of a",
        "photo of the",
        "photo of",
        "image of a",
        "image of the",
        "image of",
        "picture of a",
        "picture of the",
        "picture of",
    ]

    for phrase in portrait_phrases:
        processed_caption = processed_caption.replace(phrase, "")

    # Clean up extra spaces
    processed_caption = " ".join(processed_caption.split())

    return processed_caption.strip()

def load_florence2_model() -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Florence2 model and processor with proper data type handling."""
    print("Loading Florence2 model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        model = model.to(torch_device)
        model.eval()  # Set to evaluation mode

        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True
        )

        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")
        return model, processor

    except Exception as e:
        print(f"Error loading Florence2 model: {e}")
        sys.exit(1)

def load_upscaler_model() -> Optional[Tuple[Swin2SRImageProcessor, Swin2SRForImageSuperResolution]]:
    """Load the DiffusionPipeline for upscaling."""
    print("Loading Upscaler-Ultra model...")
    
    try:
        upscale_processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64", trust_remote_code=True)
        upscale_model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64", 
            trust_remote_code=True, 
            torch_dtype=torch_dtype 
        )
        upscale_model = upscale_model.to(torch_device)
        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")
        return upscale_processor, upscale_model
    except Exception as e:
        print(f"Warning: Could not load upscaler model. Skipping upscaling. Error: {e}")
        # Return a mock object or None if upscaler is critical
        return None 


def sharpen_image(source_path: Path, upscale_processor: Swin2SRImageProcessor, upscale_model: Swin2SRForImageSuperResolution) -> Image:
    """Sharpen image using Upscaler-Ultra before further processing."""
    if upscale_processor or upscale_model is None:
        return Image.open(source_path).convert("RGB")
        
    input_image = Image.open(source_path).convert("RGB")

    inputs = upscale_processor(images=input_image, return_tensors="pt")
    pixel_values = inputs['pixel_values']

    # 3. Run Model
    with torch.no_grad():
        outputs = upscale_model(pixel_values=pixel_values)

    # Correct Post-processing
    # The upscaled tensor is accessed via the 'reconstruction' attribute
    upscaled_tensor = outputs.reconstruction

    # Move to CPU, remove batch dim (squeeze), clip, and convert to NumPy
    # The tensor is in (1, C, H, W) format, normalized to [0, 1].
    upscaled_np = upscaled_tensor.squeeze(0).cpu().clamp(0, 1).numpy()

    # Scale to [0, 255] and change data type to 8-bit integer
    upscaled_np = (upscaled_np * 255.0).astype(np.uint8)

    # Permute dimensions from (C, H, W) to (H, W, C) for PIL
    upscaled_np = np.transpose(upscaled_np, (1, 2, 0))

    # Convert NumPy array to PIL Image object
    upscaled_image = Image.fromarray(upscaled_np)
    
    return upscaled_image

def copy_image(source_path: Path, target_dir: Path, filename: str, upscale_processor: Swin2SRImageProcessor, upscale_model: Swin2SRForImageSuperResolution, imgwidth: int, imgheight: int) -> Optional[str]:
    """Copy image to target directory with format conversion to PNG and padding to fit dimensions."""
    try:
        target_path = target_dir / filename

        # Check if target already exists
        if target_path.exists():
            print(f"  Warning: {filename} already exists, skipping copy")
            return str(target_path)  # Return path even if file exists

        # Open and process image (sharpening happens inside sharpen_image)
        img = sharpen_image(source_path, upscale_processor, upscale_model)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Calculate scaling factor to fit within target dimensions while preserving aspect ratio
        original_width, original_height = img.size
        width_ratio = imgwidth / original_width
        height_ratio = imgheight / original_height
        scale_factor = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Resize image maintaining aspect ratio
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target dimensions and white background
        padded_img = Image.new('RGB', (imgwidth, imgheight), (255, 255, 255))  # White background

        # Calculate position to center the image
        x_offset = (imgwidth - new_width) // 2
        y_offset = (imgheight - new_height) // 2

        # Paste resized image onto centered position
        padded_img.paste(resized_img, (x_offset, y_offset))

        # Save as PNG
        padded_img.save(target_path, 'PNG')

        return str(target_path)
    except Exception as e:
        print(f"Error copying {source_path}: {e}")
        return None

def main():
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # Validate directories
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    trigger_word = args.triggerword.strip()
    task = args.task
    text_input = args.text_input
    max_new_tokens = args.max_new_tokens
    num_beams = args.num_beams
    imgwidth = args.imgwidth
    imgheight = args.imgheight
    no_jsonl = args.no_jsonl

    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    if not trigger_word:
        print("Error: Trigger word cannot be empty.")
        sys.exit(1)

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Get image files
    print(f"Scanning for images in {source_dir}...")
    image_files = get_image_files(source_dir)

    if not image_files:
        print("No image files found in the source directory.")
        sys.exit(1)

    print(f"Found {len(image_files)} image files.")
    print(f"Using task: {args.task}")
    print(f"Using trigger word: {trigger_word}")
    if args.text_input:
        print(f"Using text input: {args.text_input}")

    # Load Florence2 model
    model, processor = load_florence2_model()
    
    # Load upscale DiffusionPipeline
    upscale_processor, upscale_model = load_upscaler_model()

    # Process images
    results = []
    jsonl_path = target_dir / "0_dataset.jsonl"

    print(f"\nStarting image processing and caption generation...")

    successful_processing = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_path.name}")

        target_filename = f"{i}.png"
        # Copy image to target directory with sequential naming
        copied_path = copy_image(image_path, target_dir, target_filename, upscale_processor, upscale_model, imgwidth, imgheight)
        if not copied_path:
            print(f"  ✗ Failed to copy image {image_path.name}")
            continue

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

            # Create result entry
            result_entry = {
                "image_path": copied_path,
                "control_path": copied_path,
                "caption": processed_caption,
            }
            
            results.append(result_entry)
            successful_processing += 1
            print(f"  ✓ Processed")
        else:
            print(f"  ✗ Failed to generate caption for {image_path.name}")

    if no_jsonl:
        # Create text files from results
        print("\nCreating text files from results...")
        for i, result in enumerate(results, 1):
            text_filename = f"{i}.txt"
            text_filepath = os.path.join(target_dir, text_filename)
            Path(text_filepath).write_text(result["caption"], encoding='utf-8')
            print(f"  Created caption file: {text_filename}")
    else:
        # JSONL write operation
        if results:
            print(f"\nWriting {len(results)} results to JSONL: {jsonl_path}")
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                dump = '\n'.join(json.dumps(result, ensure_ascii=False) for result in results)
                f.write(dump)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful_processing}/{len(image_files)} images")
    print(f"Images saved to: {target_dir}")
    print(f"JSONL file saved to: {jsonl_path}")

if __name__ == "__main__":
    main()

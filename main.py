#!/usr/bin/env python3
"""
Image Processing Script
Enhanced version for Florence-2 with better prompt handling and text replacement
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import torch
import transformers
import accelerate
import einops
import timm

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Process images and generate captions using Florence2 model'
    )
    parser.add_argument('source_dir', help='Source directory containing images')
    parser.add_argument('target_dir', help='Target directory to save processed images and captions and JSONL file')
    parser.add_argument('triggerword', help='The Triggerword to replace gender terms in captions')
    parser.add_argument('--task', default='more_detailed_caption',
                        choices=[
                            'region_caption', 'dense_region_caption', 'region_proposal',
                            'caption', 'detailed_caption', 'more_detailed_caption',
                            'caption_to_phrase_grounding', 'referring_expression_segmentation',
                            'ocr', 'ocr_with_region', 'docvqa', 'prompt_gen_tags',
                            'prompt_gen_mixed_caption', 'prompt_gen_analyze', 'prompt_gen_mixed_caption_plus'
                        ],
                        help='Task type for Florence-2 model')
    parser.add_argument('--text-input', default='', help='Additional text input for specific tasks')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Maximum new tokens for generation')
    parser.add_argument('--num-beams', type=int, default=3, help='Number of beams for beam search')
    
    return parser

def load_florence2_model():
    """Load Florence2 model and processor with proper data type handling."""
    print("Loading Florence2 model...")
    try:
        # Use float32 for stability
        torch_dtype = torch.float32  # Using float32 for better compatibility

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()  # Set to evaluation mode

        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True
        )

        print(f"Model loaded on {device} with dtype {torch_dtype}")
        return model, processor, device

    except Exception as e:
        print(f"Error loading Florence2 model: {e}")
        sys.exit(1)


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
        device: str,
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
                processed_inputs[key] = value.to(device=device, dtype=model_dtype)
            else:
                processed_inputs[key] = value.to(device=device)

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

    # Pronoun replacements
    pronoun_replacements = {
        "the ": f"the {trigger_word}",
        "The ": f"The {trigger_word}",
        "she ": f"the {trigger_word}",
        "She ": f"The {trigger_word}",
        "her ": f"the {trigger_word}",
        "Her ": f"The {trigger_word}",
        "hers ": f"the {trigger_word}'s",
        "Hers ": f"The {trigger_word}'s",
        "he ": f"the {trigger_word}",
        "He ": f"The {trigger_word}",
        "him ": f"the {trigger_word}",
        "Him ": f"The {trigger_word}",
        "his ": f"the {trigger_word}'s",
        "His ": f"The {trigger_word}'s",
        "Tthe": "The",
        "tthe": "the",
        f"{trigger_word}{trigger_word}": f" {trigger_word} ",
    }

    # Apply replacements
    processed_caption = caption
    for old, new in replacements.items():
        processed_caption = processed_caption.replace(old, new)

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


def copy_image(source_path: Path, target_dir: Path, filename: str, width: int = 1024, height: int = 1024) -> Optional[str]:
    """Copy image to target directory with format conversion to PNG and padding to fit dimensions."""
    try:
        target_path = target_dir / filename

        # Check if target already exists
        if target_path.exists():
            print(f"  Warning: {filename} already exists, skipping copy")
            return str(target_path)  # Return path even if file exists

        # Open and process image with padding
        with Image.open(source_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Calculate scaling factor to fit within target dimensions while preserving aspect ratio
            original_width, original_height = img.size
            width_ratio = width / original_width
            height_ratio = height / original_height
            scale_factor = min(width_ratio, height_ratio)

            # Calculate new dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize image maintaining aspect ratio
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create new image with target dimensions and white background
            padded_img = Image.new('RGB', (width, height), (255, 255, 255))  # White background

            # Calculate position to center the image
            x_offset = (width - new_width) // 2
            y_offset = (height - new_height) // 2

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
    model, processor, device = load_florence2_model()

    # Process images
    results = []
    jsonl_path = target_dir / "0_dataset.jsonl"

    print(f"\nStarting image processing and caption generation...")

    successful_processing = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_path.name}")

        target_filename = f"{i}.png"
        # Copy image to target directory with sequential naming
        copied_path = copy_image(image_path, target_dir, target_filename)
        if not copied_path:
            print(f"  ✗ Failed to copy image {image_path.name}")
            continue

        # Generate caption
        caption = generate_caption(
            model=model,
            processor=processor,
            image_path=image_path,
            device=device,
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

            if results.__contains__(result_entry):
                print(f"  Entry for image {image_path.name} already exist")
                pass

            results.append(result_entry)
            successful_processing += 1
            #print(f"  ✓ Original: {caption[:80]}...")
            #print(f"  ✓ Processed: {processed_caption[:80]}...")
            print(f"  ✓ Processed")
        else:
            print(f"  ✗ Failed to generate caption for {image_path.name}")

        # write to JSONL file
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            dump = '\n'.join(json.dumps(result, ensure_ascii=False) for result in results)
            f.write(dump)
            print("Wrote to JSONL file")

    # Create text files from results
    print("\nCreating text files from results...")
    for i, result in enumerate(results, 1):
        text_filename = f"{i}.txt"
        text_filepath = os.path.join(target_dir, text_filename)
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(result["caption"])
            print(f"  Created caption file: {text_filename}")

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful_processing}/{len(image_files)} images")
    print(f"Images saved to: {target_dir}")
    print(f"JSONL file saved to: {jsonl_path}")

if __name__ == "__main__":
    main()

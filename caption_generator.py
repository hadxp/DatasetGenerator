import re

import numpy as np
import torch
from PIL import Image
from typing import Tuple, List
from VideoInfo import VideoInfo
from utils import get_cuda_free_memory_gb, ResultEntry
from utils import torch_device, torch_dtype
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    logging as transformers_logging,
)


def generate_caption_prompt(
    prompt: str = None,
    triggerword: str = "ohwx",
    class_prompt: str = None,
    person_lora: bool = False,
    add_to_prompt: str = None,
    is_video_dataset: bool = False,
) -> str:
    """
    A person lora should describe everything but the person.<br/>
    For a style lora, the lora should include everything but the style or action.<br/>
     
    :param prompt: the prompt or None, the prompt can include {prompt} {template} or {person_template} as placeholders
    :param triggerword: the triggerword (instance prompt) for the lora
    :param class_prompt: the class_prompt for the lora, an already kown token which the model can associate with your triggerword
    :param person_lora: define if it should be a lora of a person or not
    :param add_to_prompt: add a string at the end of the prompt
    :return: the prompt
    """
    if prompt:
        prompt = re.sub(r"\{prompt\}", DEFAULT_PROMPT, prompt)
        prompt = re.sub(r"\{template\}", DESCRIPTOR_TEMPLATE, prompt)
        prompt = re.sub(r"\{person_template\}", PERSON_DESCRIPTION, prompt)
    elif person_lora:
        # caption everything but the person
        prompt = (DEFAULT_PROMPT +
                  f'Describe, template:\n{DESCRIPTOR_TEMPLATE}\n' +
                  f'Do not describe, template:\n{PERSON_DESCRIPTION}\n')
    elif not person_lora:
        # For a style lora it is better, when the user supplies, a prompt, otherwise everything is captioned (nothing is learned)
        prompt = (DEFAULT_PROMPT +
                  f'Describe, template:\n{DESCRIPTOR_TEMPLATE}{PERSON_DESCRIPTION}')

    if is_video_dataset:
        prompt = prompt + "Describe the motion."
    prompt = prompt + (f'\nThe triggerword "{triggerword}{f" {class_prompt}" if class_prompt else ""}" '
                       f'must appear at least once in the first view words of the caption. '
                       f'Never include ":".'
                       f'Put heavy focus on the describing task, the caption should be short and not overly descriptive, but long enough to mention everything described, as the goal is lora training.')
    
    if add_to_prompt is not None:
        return prompt + add_to_prompt
    return prompt


def generate_caption_qwen3(
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    results: List[ResultEntry],
    max_new_tokens: int = 512,
    num_beams: int = 1,
    prompt: str = "",
    is_video_dataset: bool = False,
) -> List[ResultEntry] | None:
    """Generate caption for an image or frames"""

    try:
        if not is_video_dataset:
            images: List[Image.Image] = [entry["image"] for entry in results]
            batch_messages = []
            for image in images:
                # 1. Structure messages for each image in the batch
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                batch_messages.append(messages)

            # Process text inputs for the entire batch
            text_inputs = [get_text_inputs(processor, msg) for msg in batch_messages]

            # Build batched multimodal inputs
            inputs = processor(
                images=images,
                text=text_inputs,
                return_tensors="pt",
                padding=True,
            )
        elif is_video_dataset:
            videos: List[Tuple[List[np.ndarray], VideoInfo]] = [entry["video"] for entry in results]
            
            batch_videos = []
            batch_messages = []
            batch_metadata = []

            for fs, video_info in videos:
                # Structure messages for each video in the batch
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": fs},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                batch_messages.append(messages)

                # Collect frames
                batch_videos.append(fs)

                # Create metadata dictionary
                metadata = {
                    "fps": video_info.fps,
                    "duration": video_info.duration,
                    "width": video_info.width,
                    "height": video_info.height,
                    "total_num_frames": len(fs),
                    # "total_original_frames": video_info.total_frames,
                    # "codec": video_info.codec,
                    # "original_resolution": f"{video_info.width}x{video_info.height}",
                }
                batch_metadata.append(metadata)

            # Process text inputs for the entire batch
            text_inputs = [get_text_inputs(processor, msg) for msg in batch_messages]

            # Build batched multimodal inputs
            inputs = processor(
                videos=batch_videos,  # List of frame lists/tensors
                text=text_inputs,  # List of text strings/inputs
                return_tensors="pt",
                padding=True,
                video_metadata=batch_metadata,  # List of metadata dicts
            )

        # Move inputs to same device/dtype as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        transformers_logging.set_verbosity_error()  # suppress all non-error messages, when generating

        # Generate caption
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                # pad_token_id=processor.tokenizer.pad_token_id,
                # eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Extract only the newly generated tokens (skipping the prompt tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        if inputs is not None:
            del inputs
        if generated_ids is not None:
            del generated_ids
        # Force garbage collection
        import gc
        gc.collect()
        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Decode the batch into a list of individual string responses
        generated_captions: List[str] = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Map them back to the original items
        for index, text in enumerate(generated_captions):
            result_entry = results[index]
            #print(f"Setting caption of length {len(text.strip())} for media {result_entry['file_path_in_target_dir'].stem}")
            result_entry["caption"] = text.strip()

        return results
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback

        traceback.print_exc()
        return None


def get_text_inputs(
    processor: Qwen3VLProcessor,
    messages,
) -> str:
    # Produce text with image tokens
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def batch_generate_captions(
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    results: List[ResultEntry],
    prompt: str,
    is_video_dataset: bool,
) -> List[ResultEntry] | None:
    return generate_caption_qwen3(
        model,
        processor,
        results=results,
        prompt=prompt,
        is_video_dataset=is_video_dataset,
    )


def process_caption_text(caption: str, triggerword: str) -> str:
    """Process and clean the caption text with trigger word replacement."""
    if not caption:
        return caption

    # Gender term replacements
    replacements = {
        "woman": triggerword,
        "man": triggerword,
        "female": triggerword,
        "male": triggerword,
        "lady": triggerword,
        "gentleman": triggerword,
        "girl": triggerword,
        "boy": triggerword,
    }

    pronoun_replacements = {
        f" {triggerword}{triggerword} ": f" {triggerword} ",  # Cleanup for double trigger words
        f"{triggerword} A {triggerword} ": f"{triggerword} ",
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


def load_caption_model_qwen3() -> Tuple[
    Qwen3VLForConditionalGeneration, Qwen3VLProcessor
]:
    """Load Qwen3-VL model and processor"""
    try:
        repoid = "Qwen/Qwen3-VL-8B-Instruct"
        print(f"Loading Qwen3-VL model from {repoid}...")

        # Load model with proper configuration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            repoid, dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )

        # Load processor
        processor = Qwen3VLProcessor.from_pretrained(repoid, trust_remote_code=True)

        # Set model to eval mode
        model.eval()

        # Optional: compile for performance (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("Model compiled with torch.compile()")

        print(f"Model loaded successfully on device: {model.device}")
        print(f"Using dtype: {torch_dtype}")

        return model, processor
    except Exception as e:
        print(f"Error loading Qwen3-VL model: {e}")
        import traceback

        traceback.print_exc()
        raise
    

DEFAULT_PROMPT = """
You are a professional image annotator. Complete the following captioning task based on the input.
Answer only with the generated caption for the input. Nothing additional.
Focus on the describing task.
Maintain authenticity and accuracy and full grammatical sentences, avoid generalizations.
"""

DESCRIPTOR_TEMPLATE = """
[Describe the poses/positions of the actors]
[Describe the location, furniture, background elements]
[Describe their actions, where they're looking, what they're doing, and their facial expression e.g. neutral, smiling, laughing, serious]
[Background style e.g. bright, dim, cluttered, minimal, cozy, clinical]
[Camera movement e.g. static, panning, handheld, zooming]
[Camera angle and framing, e.g. eye-level, low-angle, high-angle; close-up, medium shot, full-body shot]
[Clothing and accessories, describe any accessories present, if none are visible, state "no accessories"]
"""

PERSON_DESCRIPTION = """
[Body shape, size, skin color, skin details]
[Hair color, hair style, eye color, eyebrow shape, lip color, jaw shape]
"""

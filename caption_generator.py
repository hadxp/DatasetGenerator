import re
import sys
import numpy as np
import torch
import transformers
from PIL import Image
from typing import Tuple, List
from VideoInfo import VideoInfo
from packaging.version import Version
from utils import torch_device, torch_dtype
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)


def get_task_prompt(task: str) -> str:
    """Get the appropriate prompt for the given task."""
    prompts = {
        "region_caption": "<OD>",
        "dense_region_caption": "<DENSE_REGION_CAPTION>",
        "region_proposal": "<REGION_PROPOSAL>",
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        "caption_to_phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
        "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "ocr": "<OCR>",
        "ocr_with_region": "<OCR_WITH_REGION>",
        "docvqa": "<DocVQA>",
        "prompt_gen_tags": "<GENERATE_TAGS>",
        "prompt_gen_mixed_caption": "<MIXED_CAPTION>",
        "prompt_gen_analyze": "<ANALYZE>",
        "prompt_gen_mixed_caption_plus": "<MIXED_CAPTION_PLUS>",
    }

    if task not in prompts:
        raise ValueError(
            f"Unknown task: {task}. Available tasks: {list(prompts.keys())}"
        )

    return prompts[task]


def validate_task_input(task: str, text_input: str) -> None:
    """Validate if text input is allowed for the given task."""
    if text_input and task not in [
        "referring_expression_segmentation",
        "caption_to_phrase_grounding",
        "docvqa",
    ]:
        raise ValueError(
            "Text input is only supported for 'referring_expression_segmentation', "
            "'caption_to_phrase_grounding', and 'docvqa' tasks"
        )


def generate_caption_florence2(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    img: Image.Image,
    task: str = "more_detailed_caption",
    text_input: str = "",
    max_new_tokens: int = 256,
    num_beams: int = 3,
) -> str | None:
    """
    Generate caption for an image
    """

    # Validate task and input
    validate_task_input(task, text_input)
    task_prompt = get_task_prompt(task)

    # Construct final prompt
    if text_input:
        prompt = f"{task_prompt} {text_input}"
    else:
        prompt = task_prompt

    try:
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Load and process image
        image = img

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
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode the generated text
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Clean up the generated text - remove the task prompt if present
        if generated_text.startswith(task_prompt):
            generated_text = generated_text[len(task_prompt) :].strip()

        # Remove any remaining special tokens
        generated_text = generated_text.replace("<|endoftext|>", "").strip()

        return generated_text

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback

        traceback.print_exc()
        return None


DEFAULT_PROMPT = """
You are a professional image annotator. Complete the following task based on the input image or video.
Answer only with the generated caption for the input. Nothing additional.
Skip phrases like "There is no visible text" from the output text.
Focus on the describing task.

Maintain authenticity and accuracy, avoid generalizations.
Do not describe watermarks like "clideo.com"
"""

DESCRIPTOR_TEMPLATE = """
[Describe the actors and their poses/positions]
[Clothing and accessories]
[Describe the location, furniture, background elements]
[Describe their actions, where they're looking, what they're doing]
[Style, Camera movement, Camera angle]
"""

PERSON_DESCRIPTION = """
[Body shape/size, skin color, tattoos and skin details]
[Hair color and style, eye color, eyebrow shape, lip color, etc]
"""


def generate_caption_qwen3(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    frames: Tuple[List[np.ndarray], VideoInfo] | Image.Image,
    max_new_tokens: int = 512,
    num_beams: int = 1,
    prompt: str = "",
) -> str | None:
    """Generate caption for an image or frames"""

    if prompt:
        # (p
        # .replace("{prompt}", DEFAULT_PROMPT)
        # .replace("{template}", DESCRIPTOR_TEMPLATE)
        # .replace("{person_template}", PERSON_DESCRIPTION)
        # )
        prompt = re.sub(r"\{prompt\}", DEFAULT_PROMPT, prompt)
        prompt = re.sub(r"\{template\}", DESCRIPTOR_TEMPLATE, prompt)
        prompt = re.sub(r"\{person_template\}", PERSON_DESCRIPTION, prompt)
    else:
        prompt = DEFAULT_PROMPT

    # print(prompt)
    # sys.exit(1)

    try:
        if isinstance(frames, Image.Image):
            image = frames
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_inputs = get_text_inputs(processor, messages)
            # Build multimodal inputs (image + text)
            inputs = processor(
                images=frames,
                text=text_inputs,
                return_tensors="pt",
                padding=True,
            )
        else:
            fs, video_info = frames
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": fs},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
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
            text_inputs = get_text_inputs(processor, messages)
            # Build multimodal inputs (text + video)
            inputs = processor(
                videos=[fs],
                text=text_inputs,
                return_tensors="pt",
                padding=True,
                video_metadata=metadata,
            )

        # Move inputs to same device/dtype as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
                use_cache=False,
            )

        # This releases unused cached memory back to the GPU driver.
        # It does not free tensors you still hold references to.
        torch.cuda.empty_cache()

        # Remove the input tokens from the output, leaving only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        # Decode the generated text
        generated_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Return as string instead of list
        return generated_text[0] if generated_text else ""
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback

        traceback.print_exc()
        return None


def get_text_inputs(
    processor: AutoProcessor,
    messages,
) -> str:
    # Produce text with image tokens
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )


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
        " the ": f" the {trigger_word} ",  # Added spaces for safety
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
        f" {trigger_word}{trigger_word} ": f" {trigger_word} ",  # Cleanup for double trigger words
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


def generate_caption(
    model: AutoModelForCausalLM | Qwen3VLForConditionalGeneration,
    processor: AutoProcessor | Qwen3VLProcessor,
    source_object,
    task: str = "more_detailed_caption",
    text_input: str = "",
    prompt: str = "",
) -> str | None:
    if source_object is None:
        return None
    processor_class_name = processor.__class__.__name__.lower()
    if "florence" in processor_class_name:
        return generate_caption_florence2(
            model, processor, source_object, task, text_input
        )
    elif "qwen" in processor_class_name:
        return generate_caption_qwen3(model, processor, source_object, prompt=prompt)
    else:
        print("No caption generator found")
    return None


def load_cation_model_florence2() -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Florence2 model and processor with proper data type handling."""
    repoid = "microsoft/Florence-2-large"
    print(f"Loading caption({repoid}) model...")

    version = Version(transformers.__version__)

    if version != Version("4.53.1"):
        print("Please use transformers version 4.53.1")
        sys.exit(1)

    try:
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            repoid, trust_remote_code=True, torch_dtype=torch_dtype
        )

        model = model.to(torch_device)  # move the model to cuda
        model.eval()  # Set to evaluation mode

        processor: AutoProcessor = AutoProcessor.from_pretrained(
            repoid, trust_remote_code=True
        )

        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")
        return model, processor
    except Exception as e:
        print(f"Error loading Florence2 model: {e}")
        sys.exit(1)


def load_caption_model_qwen3() -> Tuple[
    Qwen3VLForConditionalGeneration, Qwen3VLProcessor
]:
    """Load Qwen3-VL model and processor"""
    try:
        repoid = "Qwen/Qwen3-VL-8B-Instruct"
        print(f"Loading Qwen3-VL model from {repoid}...")

        version = Version(transformers.__version__)

        if version != Version("4.57.6"):
            print("Please use transformers version 4.57.6")
            sys.exit(1)

            # Determine optimal dtype based on available memory
        if torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).total_memory >= 16e9:  # 16GB
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

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

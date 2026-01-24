import sys
import torch
import transformers
import numpy as np
from PIL import Image
from packaging.version import Version
from typing import Tuple, List
from utils import torch_device, torch_dtype
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText

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

def generate_caption_florence2(
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        img: Image.Image,
        task: str = "more_detailed_caption",
        text_input: str = "",
        max_new_tokens: int = 256,
        num_beams: int = 3
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
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
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
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_caption_qwen3(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    video_frames: List[np.ndarray],
    max_new_tokens: int = 512,
    num_beams: int = 1
) -> str | None:
    """Generate caption for an image"""
    prompt = ("Describe this video in detail use girl instead of names. Skip in-depth background description. Answer only with the generated caption for the video. Nothing additional."
              "Focus on the describing task")

    try:   
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Produce text with image tokens
        text_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )

        if isinstance(video_frames, Image.Image):
            # Build multimodal inputs (image + video)
            inputs = processor(
                images=video_frames,
                text=text_inputs,
                return_tensors="pt",
            )
        else:
            # Build multimodal inputs (text + video)
            inputs = processor(
                videos=video_frames,
                text=text_inputs,
                return_tensors="pt",
            )

        # Move inputs to same device/dtype as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate caption 
        with torch.no_grad(): 
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

        # Decode the generated text 
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        # Remove everything upto (including) the word "assistant"
        marker = "assistant"

        # Use find() to get the index of the first occurrence (of the marker)
        start_index = generated_text.find(marker) + len(marker)

        # Slice the string from that index to the end
        generated_text = generated_text[start_index:]

        return generated_text
    except Exception as e:
        print(f"Error processing video: {e}")
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

def generate_caption(model, processor: AutoProcessor, source_object, task: str = "more_detailed_caption", text_input: str = "") -> str | None:
    is_florence = "florence" in model.__class__.__name__.lower()
    is_qwen = "qwen" in model.__class__.__name__.lower()
    if is_florence: # florence2
        return generate_caption_florence2(model, processor, source_object, task, text_input)
    elif is_qwen: # qwen
        return generate_caption_qwen3(model, processor, source_object)
    else:
        print("No caption generator found")
    return None

def load_cation_model_florence2() -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Florence2 model and processor with proper data type handling."""
    repoid="microsoft/Florence-2-large"
    print(f"Loading caption({repoid}) model...")
    
    version = Version(transformers.__version__)

    if version == Version("4.53.1"):
        print("Please use transformers version 4.53.1")
        sys.exit(1)

    try:
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(repoid, trust_remote_code=True, torch_dtype=torch_dtype)

        model = model.to(torch_device) # move the model to cuda
        model.eval()  # Set to evaluation mode

        processor: AutoProcessor = AutoProcessor.from_pretrained(repoid, trust_remote_code=True)

        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")
        return model, processor
    except Exception as e:
        print(f"Error loading Florence2 model: {e}")
        sys.exit(1)
        
def load_caption_model_qwen3() -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load Caption model and processor with proper data type handling."""
    try:
        repoid = "Qwen/Qwen3-VL-8B-Instruct"
        print(f"Loading caption({repoid}) model...")

        version = Version(transformers.__version__)

        if version == Version("5.0.0"):
            print("Please use transformers version 5.0.0")
            sys.exit(1)

        model = (AutoModelForImageTextToText.from_pretrained(repoid, device_map="auto", dtype=torch_dtype)
                 .to(torch_device))
        processor = AutoProcessor.from_pretrained(repoid, dtype=torch_dtype)

        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")

        return model, processor
    except Exception as e:
        print(f"Error loading Caption model: {e}")
        sys.exit(1)
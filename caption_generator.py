import re
import sys

import numpy as np
import torch
import transformers
from PIL import Image
from typing import Tuple, List
from VideoInfo import VideoInfo
from packaging.version import Version
from transformers import DynamicCache
from utils import get_cuda_free_memory_gb
from utils import torch_device, torch_dtype
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    logging as transformers_logging,
)


def generate_caption_prompt(
    prompt: str = None,
    triggerword: str = "ohwx",
    class_prompt: str = None,
    person_lora: bool = False,
) -> str:
    """
    A person lora should describe everything but the person.<br/>
    For a style lora, the lora should include everything but the style or action.<br/>
    
    :param prompt: the prompt or None, the prompt can include {prompt} {template} or {person_template} as placeholders
    :param triggerword: the triggerword (instance prompt) for the lora
    :param class_prompt: the class_prompt for the lora, an already kown token which the model can associate with your triggerword
    :param person_lora: define if it should be a lora of a person or not
    :return: the prompt
    """
    if prompt:
        prompt = re.sub(r"\{prompt\}", DEFAULT_PROMPT, prompt)
        prompt = re.sub(r"\{template\}", DESCRIPTOR_TEMPLATE, prompt)
        prompt = re.sub(r"\{person_template\}", PERSON_DESCRIPTION, prompt)
        prompt = re.sub(r"\{handbook_template\}", PROMPT_HANDBOOK_TEMPLATE, prompt)
    elif person_lora:
        prompt = (DEFAULT_PROMPT +
                  f'\nExtensively describe, template:\n{DESCRIPTOR_TEMPLATE}\n' +
                  f'Do not describe, template:\n{PERSON_DESCRIPTION}\n')
    elif not person_lora:
        # For a style lora it is better, when the user supplies his own prompt
        prompt = (DEFAULT_PROMPT +
                  f'\nExtensively describe, template:\n{DESCRIPTOR_TEMPLATE}{PERSON_DESCRIPTION}')
        
    prompt = prompt + (f'\nThe triggerword "{triggerword}" {f" and the class prompt {class_prompt}" if class_prompt else " "} '
                       f'should appear at least once in the first view words of the caption. '
                       f'While generating the caption, adhere to this, handbook:\n{PROMPT_HANDBOOK_TEMPLATE}\n '
                       f'Put heavy focus on the describing task, according to the description templates i provided, and make sure you described everything according to my description templates. '
                       f'Never include ":" '
                       f'If the input is a video or multiple frames descibe the motion, throught the caption. '
                       f'The generated caption should be thorough and include everything described in my templates, described according to the handbook ')
    
    return prompt

def encode_system_prompt(model, processor, system_prompt: str) -> DynamicCache:
    """Pre-encode a static system prompt once, reuse for every caption call."""
    messages = [{"role": "system", "content": system_prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        # Forward pass only — no generation, just encode and grab the cache
        outputs = model(**inputs, use_cache=True, return_dict=True)

    return outputs.past_key_values  # reuse this on every call


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

def generate_caption_qwen3(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    frames: Tuple[List[np.ndarray], VideoInfo] | Image.Image,
    max_new_tokens: int = 512,
    num_beams: int = 1,
    prompt: str = "",
    text_encoder_cache = None,
) -> str | None:
    """Generate caption for an image or frames"""

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

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                # pad_token_id=processor.tokenizer.pad_token_id,
                # eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,                                     # enable KV cache during generation
                past_key_values=text_encoder_cache,          # reuse cached text-encoder output
                return_dict_in_generate=True,                       # needed to get cache back out
            )

        # Extract the updated cache for reuse on next call
        new_text_encoder_cache = generated_ids.past_key_values  # DynamicCache object

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids.sequences)
        ]

        del inputs
        torch.cuda.empty_cache()

        generated_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return (generated_text[0] if generated_text else ""), new_text_encoder_cache
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


def generate_caption(
    model: AutoModelForCausalLM | Qwen3VLForConditionalGeneration,
    processor: AutoProcessor | Qwen3VLProcessor,
    source_object,
    task: str = "more_detailed_caption",
    text_input: str = "",
    prompt: str = "",
    text_encoder_cache : DynamicCache = None,
) -> Tuple[str, DynamicCache] | None:
    if source_object is None:
        return None
    processor_class_name = processor.__class__.__name__.lower()
    if "florence" in processor_class_name:
        return generate_caption_florence2(
            model, processor, source_object, task, text_input
        )
    elif "qwen" in processor_class_name:
        return generate_caption_qwen3(
            model,
            processor,
            source_object,
            prompt=prompt,
            text_encoder_cache=text_encoder_cache,
        )
    else:
        print("No caption generator found")
    return None


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
            repoid, trust_remote_code=True, dtype=torch_dtype
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

PROMPT_HANDBOOK_TEMPLATE = """
# HunyuanVideo Framepack Prompt Handbook

## I. Basic Features
### Image-to-Video
**Feature Overview**: "an image + a text prompt" to generate a corresponding video. The first frame of the video comes from the uploaded image, while the content of subsequent frames will be generated according to the text prompt.
**Core Formula**: Prompt = Subject Motion Dynamics + Scene Motion Dynamics + [Camera Movement]

## II. Advanced Controls
### 1. Style Control

* Photorealistic/Cinematic Style
A tired middle-aged Asian man, wearing a pilling grey sweater, with fine wrinkles around his eyes, looks out the window with a worried expression, cinematic lighting, photorealism style.``` </details> |<video src="https://github.com/user-attachments/assets/45203f76-2ae8-43e4-9a85-73496434938a" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A city night view, flying cars shuttle between skyscrapers, giant holographic billboards flash, strong blue-purple tones, cyberpunk style, neon lights flickering.``` </details>|<video src="https://github.com/user-attachments/assets/95401a66-3713-4052-bcab-1010e934a2c9" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```An astronaut slowly floats inside a derelict International Space Station, outside is the deep cosmos and a blue Earth, cool-toned lighting, slow motion, hard sci-fi movie style.``` </details>|<video src="https://github.com/user-attachments/assets/29b8dbca-5a26-416c-9f49-eb05b2c78015" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A bearded middle-aged detective, wearing a trench coat and fedora, stands on a city street in pouring rain, late at night, with a bar in the background featuring a red and blue neon sign. The camera uses a medium shot, very slowly pushing in on him. Lighting primarily comes from streetlights and neon, creating high-contrast light and shadow on his face and drenched trench coat. The overall style is cinematic Film Noir, evoking a melancholic, mysterious, and suspenseful atmosphere.

* Animation/Illustration Style
Low-Poly style 3D animation, a giant geometrically shaped whale slowly swims in an underwater world composed of angular coral and seaweed. Crystal-like bubbles rise around it, and soft sunbeams pierce through the water's surface, forming ever-shifting light patches on the seabed, illuminating the entire scene. A low-angle shot is used, showcasing the ocean's depth and grandeur, creating a tranquil and geometrically aesthetic atmosphere.``` </details> |<video src="https://github.com/user-attachments/assets/b799c5ec-db5e-417e-a35c-6289a100e3fd" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```Several steep distant mountains, faintly visible amidst the clouds and mist, a small boat slowly glides across the wide river, leaving subtle ripples. A grand wide shot is used, with the camera slowly panning left. The scene is composed of varying shades of ink and ample blank space, embodying a dynamic Chinese ink wash painting (Xieyi style), creating an atmosphere of tranquility, solitary grandeur, and profound artistic depth.``` </details>|<video src="https://github.com/user-attachments/assets/d535c359-d53d-4aca-9be2-01964f54e0f0" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```充满活力的2D动画风格，一位戴着护目镜的少年发明家，驾驶着他自己建造的扑翼飞行器，在一座天空之城的上空轻快地滑翔。他穿梭于巨大的风车和漂浮的岛屿之间，下方是繁忙的空中街道。镜头平稳地跟随他，阳光穿过巨大的风车叶片，投下动态的光影，营造出乐观而富有想象力的氛围。``` </details>|<video src="https://github.com/user-attachments/assets/ad6aafa1-3dd7-40c2-9cb9-e473f42878fd" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A delicate watercolor illustration depicts three young women at a dining table celebrating by toasting with red wine glasses. In a bright, airy setting captured in a beautiful watercolor style, three young women are seated around a wooden dining table. The woman in the center has wavy blonde hair and wears a light blue dress. To her left, a woman with a chic brown bob wears a cream-colored blouse. To her right, a woman with long black hair is in a soft pink top. All three have joyful expressions and are smiling warmly. On the table in front of them are three elegant glasses filled with translucent red wine, their forms defined by soft, flowing colors. Initially, the women's hands, rendered with light, expressive strokes, are near their glasses. Then, they simultaneously reach out, grasp the stems of their wine glasses, and begin to lift them from the table. Next, they smoothly raise the glasses towards the center of the frame in a celebratory gesture, their movements fluid and graceful. Finally, the three glasses gently meet in the middle for a toast, the liquid inside swirling slightly with the motion. The background is an abstract wash of warm yellows and soft browns, suggesting a cozy indoor environment without specific details, characteristic of a wet-on-wet watercolor technique. The light source is diffuse and from the front, casting gentle, soft-edged shadows. The paper texture is subtly visible, with faint water stains at the edges of the color washes, enhancing the authentic watercolor aesthetic. The shot is at an eye-level angle with the main subjects. The camera remains static. The entire scene is defined by its distinct watercolor style, featuring translucent color layers, delicate ink outlines, and visible pigment bleeding. The mood is joyful and celebratory. The overall video presents a classic illustration watercolor style.

### 2. Lighting Control
* **Core Principle**: Light is the soul of atmosphere. Learning to describe light allows you to control the mood of the video.
* **Common Lighting Description Techniques**: 
  * Lighting Style: (e.g., Soft, Hard, Neon lighting)
  * Lighting Direction: (e.g., Top-down, Side lighting)
  * Light Quality: (e.g., Soft, Harsh, Spotlight)
  * Shadow Details: (e.g., Deep shadows, Soft gradients, High contrast)
  * Color Temperature: (e.g., Warm golden hour, Cool daylight, Golden moment)
  * Reflections: (e.g., Reflected light on water, glass, or metal surfaces)
  * Silhouettes and Contours: (e.g., Subject backlit, Creating dramatic contours, Backlight/Silhouette)

### 3.Camera Movement Control
**Camera Movement Library**
"The camera moves upward/downward" = "Vertical crane/pedestal shot"
"The camera moves to the left/right" = "Horizontal truck/tracking shot"
"The camera moves forward" = "Dolly in"
"The camera moves back" = "Dolly out"
"The camera tilts up/down" = "High angle or low angle adjustment"
"The camera pans to the left/right" = "Horizontal rotation around the axis"
"The camera circles around" = "Shooting around the subject"
"The camera rotates 360 degrees" = "Full 360-degree surround"
"The camera follows" = "Lock on and move with subject"
"The camera remains static" = "Fixed camera position"

### 4. In-Video Bilingual Text Rendering
* Usage: Enclose the text you wish to generate within quotation marks in your prompt.
* Chinese Prompt: Please use Chinese double quotes “”.
* English Prompt: Please use English double quotes "".


### 5. Additional Advanced Controls and Instructions
a. **Supported Languages**: Currently supports prompt input in both Chinese and English.
b. **Video Dimensions**: Supports multiple aspect ratios including 16:9 (Landscape), 4:3, 1:1 (Square), 3:4, and 9:16 (Portrait). Please configure this before generation.
c. **Keep it Concise**: Try to use simple, direct vocabulary and grammatical structures.
d. **Prompt Components Breakdown**:

| Component | Description | Examples |
|-----------|------------|----------|
| Subject | The core object of the video. Describe appearance, attire, hairstyle, species, etc. | An Asian woman with long black hair wearing a red dress, a cute ragdoll cat |
| Motion | The action the subject is performing or their state. Should be clear and direct. | Running, typing intently, walking slowly, took a sip of coffee |
| Scene | The environment or background where the subject is located. | On a city street at night, in the kitchen, on the grass, on the surface of the moon |
| Shot Type | The type of camera shot, used to highlight or emphasize specific visual content. | Aerial shot, Close-up shot, Medium shot, Long shot |
| Camera Movement | The way the camera moves. | Refer to the Camera Movement Library above |
| Lighting | Describes the lighting conditions of the video. | Refer to the Lighting Description Techniques above |
| Style | The visual style type of the video. | Photorealistic style, Cyberpunk style, Sci-fi style, Pixel art style, Ink wash painting, etc. |
| Atmosphere | The overall mood and tone of the video. | Warm, Tense, Mysterious, Cinematic |

**e. To make the video generation more accurate and dynamic, it is recommended to follow these requirements to make prompts more specific and responsive**:
* Dynamics and Sequentiality
    * Rule: Describe the visual as a process with a time sequence, using conjunctions to clarify steps.
    * Recommended Sentence Structure: First... then... next... meanwhile... finally...
    * Demo: The girl first arranges her hair, then turns to walk toward the door, and finally stops in front of the door to look back at the camera.
* Objective Description of Details
  * Rule: Reduce the use of abstract emotional words and convert them into "action details."
  * Formula: Subject + Action + Small Detail
  * Demo: The boy smiles, eyes crinkling slightly.
* Precision of Space and Orientation
  * Rule: Use simple directional words to clarify "who is where" and "moving where."
  * Directional Vocabulary: Left/Right side of the frame, Top/Bottom, Center, Foreground/Background.
  * Demo: A hand reaches out from the right side of the frame, touches the tag on the black clothes, and then leaves the frame from the bottom.
* Clear Reference/Attribution
  * Rule: When there are multiple characters in the frame (including input images for Image-to-Video), it is recommended to distinguish individuals by attributes or position to avoid confusion.
  * Demo: The black cat hands the bomb to the gray cat; the gray cat takes the bomb and turns to run toward the right side of the frame.

## III. More Creative Usage and Cases
### 1. Strong Instruction Response
demo:
A young Chinese woman with dark, slightly disheveled long curly hair, wearing a shimmering pearl necklace and round gold earrings, is seen from a top-down angle. Her messy hair is blown by the wind, and she slightly raises her head, looking up at the sky with a very sorrowful expression, tears welling in her eyes, one tear sliding down her cheek. Her lips are painted with red lipstick. The background features an ornate red floral pattern. The scene presents a retro cinematic style, with low saturation tones and a slight soft focus to enhance the emotional atmosphere. The texture resembles classic film from the 1990s, creating a nostalgic yet dramatic feeling.``` </details> |<video src="https://github.com/user-attachments/assets/dd29b6b8-62dd-4aa6-b3ee-5a109d1a7cc8" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```In an empty modern loft, an architectural blueprint is spread out on the center of the floor. Suddenly, the lines on the blueprint begin to glow faintly, as if awakened by some unseen force. Immediately following, the glowing lines start to extend upward, breaking free from the flat surface and outlining three-dimensional contours—like a silent 3D print happening in the air. Subsequently, the miracle accelerates: a minimalist oak desk, an elegant Eames-style leather chair, a tall industrial metal bookshelf, and several Edison bulbs rapidly "grow" with light patterns as their skeleton. In the blink of an eye, the lines are filled with real textures—the warmth of wood, the feel of leather, the coolness of metal—all completely rendered. Finally, all the furniture firmly settles onto the floor, and the blueprint's light quietly fades. A complete office space is thus born from the two-dimensional drawing.``` </details>|<video src="https://github.com/user-attachments/assets/5d23d803-4acc-41ac-ae3e-84fd31b5794a" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```Cinematic 4K macro videography, presented in a hyper-realistic style. A tight, static shot reveals a scene bathed in the focused glow of a single, warm-toned overhead lamp, which casts deep, soft shadows. Upon a soft grey felt jeweler's mat rests an exquisite mechanical clockwork beetle, its carapace fashioned from polished brass and its tiny legs from gleaming silver. Through translucent panels, a complex array of minuscule, ruby-red gears and cogs are visible. A pair of impossibly fine, steel tweezers, held with surgical steadiness, descends into the frame. With breathtaking precision and in extreme slow motion, the tweezers begin a delicate disassembly, first gripping and unscrewing a microscopic brass screw. The screw turns with deliberate slowness, its threads catching the light. The tweezer then lifts the carapace away, revealing the full, intricate clockwork mechanism whirring silently within. One by one, the tweezers pluck individual ruby gears from their mountings; each component lifts away smoothly, its polished teeth glinting, before being placed gently on the felt beside the beetle's inert body. This hyper-realistic footage captures every subtle specular highlight on the metal and the soft, light-absorbing texture of the felt, creating a powerful study in precision engineering and meticulous deconstruction.``` </details>|<video src="https://github.com/user-attachments/assets/fa18b171-cd80-4fa9-80da-5cf5e762b6cb" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A sweet anime girl in a 'HunyuanVideo 1.5' sweater makes a heart gesture with her hands. The main subject is a young anime girl with fair skin, large, sparkling blue eyes, and long, flowing pastel pink hair that frames her face. She is wearing a slightly oversized, cozy, cream-colored knitted sweater. Across the chest area of her sweater, the text "HunyuanVideo 1.5" is clearly and neatly printed in a clean, black sans-serif font. Her expression is cheerful and endearing. Initially, the girl stands facing the camera with a gentle smile, her hands positioned in front of her chest, slightly apart. Then, she smoothly brings her hands together, touching her thumbs to form the bottom point of a heart and curving her index and middle fingers to create the top arches. As the heart shape is completed, her smile widens, and she gives a playful wink with her right eye. She is situated in the center of a softly lit room. The background is blurred with a shallow depth of field, suggesting a clean, minimalist interior with a warm, gentle, and comfortable feel. Sunlight filters in from a window off-screen, casting soft highlights on her hair and shoulders, all rendered in a soft, painterly anime style. Medium close-up shot. The camera is at an eye-level angle with the main subject. The camera zooms in slowly, emphasizing her facial expression and hand gesture. The lighting is soft and diffused, creating a warm and inviting mood. The visual style is a high-quality Japanese cel-shading animation, characterized by clean lines, vibrant yet soft colors, and detailed character design. This is a beautiful anime-style animation. The overall video presents a high-quality Japanese animation style.``` </details>|<video src="https://github.com/user-attachments/assets/700b9008-0a19-4c10-97a2-8671e4b376c3" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```The hiker begins walking forward along the trail, causing the water bottle to swing rhythmically with each step. The camera gradually pulls back and rises to reveal a vast desert landscape stretching out ahead, while the sun position shifts from afternoon to dusk, casting increasingly longer shadows across the terrain as the figure becomes smaller in the frame.

### 2. Fluid Motion Generation
demo:
slowly advancing medium shot, shot from a level angle, focuses on the center of an empty football field, where a DJ is immersed in his musical world. He wears a pair of professional, matte-black headphones, one earcup slightly removed, revealing a focused expression and a brow beaded with sweat from his intense concentration. He wears a black bomber jacket, zipped open to reveal a T-shirt underneath. His upper body sways back and forth rhythmically to the throbbing electronic beats, his head moving with precise movement. The mixing console in front of him serves as the primary source of light. In the distance, the cool white glow of several stadium floodlights casts a deep, dark haze across the vast field, casting long shadows across the emerald green grass, creating a stark contrast to the brightly lit area surrounding the DJ booth. His hands danced swiftly and precisely across the equipment, one hand steadily pushing and pulling a long volume fader, while the fingers of the other nimbly jumped between the illuminated knobs and pads, sometimes decisively cutting a bass line, sometimes triggering an echo effect. The entire scene was filled with high-tech dynamics and the solitary creative passion. Against the backdrop of the vast and silent night stadium, it created an atmosphere of high focus, energy, and a slightly surreal feeling.``` </details> |<video src="https://github.com/user-attachments/assets/a7743347-a5ae-4d2e-b172-6fc9530eadc9" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```蛋糕人坐在椅子上，随后，他用手从自己的腿上掰下一块蛋糕，掉落了一些蛋糕碎屑，腿上显示出蛋糕的缺口。接着，他将掰下的蛋糕块举到嘴边，张开嘴咬了一口，咀嚼了几下。背景中的桌子和墙壁保持静止。``` </details><details><summary>📷 Input image</summary> <img src="https://github.com/user-attachments/assets/42b029ba-1cda-49f7-806e-db4df044cd14" width="600"></img></details>|<video src="https://github.com/user-attachments/assets/25ea0214-f949-479a-b9af-20d66286c295" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```The camera remains static. A massive tiger in the center of the frame runs toward the camera, its four limbs powerfully pushing off the ground, its orange and black stripes slightly shimmering in the sunlight with the ripple of its muscles. A small boy wearing a blue short-sleeved shirt sits on the tiger's back, his hands resting lightly on the sides of the tiger's neck, his feet hanging naturally down the tiger's sides, his face smiling, his mouth curved upward, and his eyes slightly squinted against the wind of the run. The background is a dense forest, with tall, moss-covered trees. Sunlight slants down through the canopy gaps, forming beams of light, and as the tiger runs, the light spots slowly shift across the ground and leaves. Foreground fern leaves gently sway due to the airflow, and dew drops slide from the leaf tips. As the tiger runs, its tail sways side to side, its gait is steady, and every step kicks up faint dust. Within five seconds, the tiger and the boy continuously approach the camera, eventually dominating most of the frame, with the camera remaining fixed, highlighting the fusion of the subject's dynamism and the environmental atmosphere.

### 3. Adherence to Physical Laws
demo:
The video captures a basketball going through the hoop. The subject is the orange ball. Initially, it arcs through the air. Then, it passes through the net without touching the rim (swish). Next, the white net whips up violently. The background is the blurred crowd. The camera shoots from a low angle under the basket. The lighting is focused arena lighting. The overall video presents a satisfying moment style.``` </details> |<video src="https://github.com/user-attachments/assets/54d2e2f4-3567-4d61-b61d-a9ca485378fe" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```In a sleek museum gallery, a woman receives a glass of wine poured directly from an animated oil painting.``` </details>|<video src="https://github.com/user-attachments/assets/80a1b823-efbd-4abc-b0f2-d5e357fba6d8" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```A sophisticated woman with dark hair tied back elegantly stands in the mid-ground. She is wearing a simple, black silk sleeveless dress and holds a clear, crystal wine glass in her right hand. She is positioned before a large, baroque-style oil painting in an ornate, gilded frame. Inside the painting, an aristocratic man with a mustache, dressed in a dark velvet doublet with a white lace collar, is depicted. His form is defined by visible, impasto oil brushstrokes. Initially, the woman watches the painting with calm poise. Then, the painted man's arm slowly animates, his painted texture retained as he lifts a dark bottle. Next, a photorealistic stream of red wine emerges directly from the flat canvas surface, arcing through the air and splashing gently into the real crystal glass she holds. She remains perfectly still, accepting the impossible pour with a subtle, knowing smile.and slowly. The setting is a modern art gallery with high white walls and polished dark concrete floors that reflect the ambient light. Focused track lighting from the high ceiling casts a warm, dramatic spotlight on the woman and the painting, creating soft shadows. In the background, two other gallery patrons, a man and a woman in stylish, modern attire, stroll slowly from right to left, their figures slightly blurred by a shallow depth of field, moving naturally through the hall. The shot is at an eye-level angle with the woman. The camera remains static, capturing the surreal event in a steady medium shot. The lighting is high-contrast and dramatic, reminiscent of a cinematic photography realistic style, using soft side lighting to accentuate the woman's features and the texture of the painting. The mood is surreal, elegant, and mysterious. The overall video presents a cinematic photography realistic style. crushes a red and white soda can on a dark surface, captured in a cinematic, realistic style.

### 4. Cross-Dimensional Generation
demo:
The camera remains static. The scene features ultra-high-definition picture quality, with clear and sharp details. Inside a computer screen, the scene shows the Krusty Krab kitchen, where the cartoon character SpongeBob SquarePants stands in front of the grill, holding a spatula in his right hand and supporting a freshly made Krabby Patty with his left, his mouth wide in a grin, revealing neat teeth, his eyes wide, and his gaze bright. Subsequently, SpongeBob's right hand slowly extends forward from the screen image, breaking through the screen plane and entering the foreground space. He holds a complete Krabby Patty in his hand; the lettuce, patty, cheese, and sauce of the burger are clearly layered, sesame seeds are sprinkled on the top bun, and steam gently rises from the burger's seams. His hand moves smoothly, his wrist slightly turns, gently placing the Krabby Patty into the palm of a real human hand in the foreground. The palm is naturally open, its skin texture clearly visible, and after the Krabby Patty touches the palm, the hand slightly depresses due to the weight. Subsequently, SpongeBob's hand slowly retracts, moving upward from the foreground, finally pulling back completely into the screen and returning to the kitchen scene.``` </details> <details><summary>📷 Input image</summary> <img src="https://github.com/user-attachments/assets/77c53ac6-69af-469a-8246-f4428a717e98" width="600"></img></details>|<video src="https://github.com/user-attachments/assets/2789eb3d-7697-4abe-b6e7-47daf50723d7" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```镜头静止不动，随后缓缓向前推进，聚焦于前景中一只真实的人手与背景电脑屏幕上的卡通人物水冰月之间的互动场景。屏幕中的水冰月保持施法准备姿势，双眼注视前方，嘴角微微上扬。接着，水冰月缓缓伸出右手，从屏幕平面中探出，手臂跨越虚拟与现实的边界，进入前景空间。她的手在真实人手的掌心上方悬停，手部在人手上投射出清晰的倒影。随后，水冰月的手指轻触人手掌心，接触瞬间，人手皮肤表面浮现出一道银色星光印记，印记随即开始闪烁，频率逐渐减缓，最终完全消失。虚拟人物的手与真实人手在空间位置与视觉层次上无缝衔接，呈现浑然一体的视觉效果。``` </details><details><summary>📷 Input image</summary> <img src="https://github.com/user-attachments/assets/cbd404df-18c8-4661-81d4-778f3159e742" width="600"></img></details>|<video src="https://github.com/user-attachments/assets/4e4eeb83-d350-4b04-aa29-50fd0454ac0f" width="600"> </video> <details><summary>📋 Show input prompt</summary> ```The camera remains static. In the foreground, a hand with clear skin texture firmly holds an empty glass cup, which is positioned above the laptop keyboard. In the background, the laptop screen lights up, and a cartoon version of Harry Potter appears on the screen, looking focused with a slight upturn of his mouth. The cartoon Harry Potter raises his wand and then waves it forward, the tip of the wand seemingly extending out of the screen, pointing toward the real glass cup in the foreground. A beam of golden light shoots out from the tip of the wand and shines into the glass cup. Simultaneously, steaming Butterbeer begins to generate spontaneously in the cup; the liquid level gradually rises, and a thick layer of white foam forms at the top of the cup, then the foam slowly spills over the cup rim. The golden light casts a clear reflection on the surface of the glass cup.

### 5. Action Logic and Decomposition
Core Formula: Prompt = Scene Setting + Sequential Action Decomposition + Key Details
"""

DEFAULT_PROMPT = """
You are a professional image annotator. Complete the following captioning task based on the input.
Answer only with the generated caption for the input. Nothing additional.
Skip phrases like "There is no visible text" from the output text.
Focus on the describing task.

Maintain authenticity and accuracy, avoid generalizations.
Do not describe watermarks like "clideo.com".
"""

DESCRIPTOR_TEMPLATE = """
[Describe the actors and their poses/positions]
[Describe the location, furniture, background elements]
[Describe their actions, where they're looking, what they're doing]
[Background Style, Camera movement, Camera angle]
[Clothing and accessories]
"""

PERSON_DESCRIPTION = """
[Body shape/size, skin color, skin details]
[Hair color and style, eye color, eyebrow shape, lip color, etc]
"""

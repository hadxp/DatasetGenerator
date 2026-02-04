import cv2
import torch
import numpy as np
from PIL import Image
import math
from typing import Tuple, Union
from utils import torch_device, torch_dtype
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

IMAGE_FACTOR = 28  # The image size must be divisible by this factor
DEFAULT_MAX_SIZE = 1280


def load_upscaler_model() -> (
    Tuple[Swin2SRImageProcessor, Swin2SRForImageSuperResolution] | None
):
    """Load the DiffusionPipeline for upscaling."""

    repo_id = "caidas/swin2SR-classical-sr-x2-64"
    print(f"Loading upscale({repo_id}) model...")

    try:
        upscale_processor: Swin2SRImageProcessor = (
            Swin2SRImageProcessor.from_pretrained(repo_id, trust_remote_code=True)
        )
        upscale_model: Swin2SRForImageSuperResolution = (
            Swin2SRForImageSuperResolution.from_pretrained(
                repo_id, trust_remote_code=True, dtype=torch_dtype
            )
        )
        upscale_model = upscale_model.to(torch_device)
        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")
        return upscale_processor, upscale_model
    except Exception as e:
        print(f"Warning: Could not load upscaler model. Skipping upscaling. Error: {e}")
        # Return a mock object or None if upscaler is critical
        return None


def resize_image_to_bucket(
    image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]
) -> np.ndarray:
    """
    Resize the image to the bucket resolution.

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    # resize the image to the bucket resolution to match the short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(
            image, (image_width, image_height), interpolation=cv2.INTER_AREA
        )

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[
        crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width
    ]
    return image


def sharpen_image(
    img: Image.Image,
    upscale_processor: Swin2SRImageProcessor,
    upscale_model: Swin2SRForImageSuperResolution,
) -> Image.Image:
    """Sharpens image"""

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    if upscale_processor or upscale_model is None:
        return img

    input_image = img

    inputs = upscale_processor(images=input_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

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


def preprocess_image(
    img: Image.Image,
    upscale_processor: Swin2SRImageProcessor,
    upscale_model: Swin2SRForImageSuperResolution,
) -> Image.Image:
    return preprocess_image_v2(
        img, upscale_processor, upscale_model,
    )


def preprocess_image_v2(
    image: Image.Image,
    upscale_processor: Swin2SRImageProcessor,
    upscale_model: Swin2SRForImageSuperResolution,
    max_size: int = DEFAULT_MAX_SIZE,
) -> Image.Image:
    # Open and process image (sharpening happens inside sharpen_image)
    image = sharpen_image(image, upscale_processor, upscale_model)

    """Resize image to a suitable resolution"""
    min_area = 256 * 256
    max_area = max_size * max_size
    width, height = image.size
    width_rounded = int((width / IMAGE_FACTOR) + 0.5) * IMAGE_FACTOR
    height_rounded = int((height / IMAGE_FACTOR) + 0.5) * IMAGE_FACTOR

    bucket_resos = []
    if width_rounded * height_rounded < min_area:
        # Scale up to min area
        scale_factor = math.sqrt(min_area / (width_rounded * height_rounded))
        new_width = math.ceil(width * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR
        new_height = math.ceil(height * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR

        # Add to bucket resolutions: default and slight variations for keeping aspect ratio
        bucket_resos.append((new_width, new_height))
        bucket_resos.append((new_width + IMAGE_FACTOR, new_height))
        bucket_resos.append((new_width, new_height + IMAGE_FACTOR))
    elif width_rounded * height_rounded > max_area:
        # Scale down to max area
        scale_factor = math.sqrt(max_area / (width_rounded * height_rounded))
        new_width = math.floor(width * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR
        new_height = math.floor(height * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR

        # Add to bucket resolutions: default and slight variations for keeping aspect ratio
        bucket_resos.append((new_width, new_height))
        bucket_resos.append((new_width - IMAGE_FACTOR, new_height))
        bucket_resos.append((new_width, new_height - IMAGE_FACTOR))
    else:
        # Keep original resolution, but add slight variations for keeping aspect ratio
        bucket_resos.append((width_rounded, height_rounded))
        bucket_resos.append((width_rounded - IMAGE_FACTOR, height_rounded))
        bucket_resos.append((width_rounded, height_rounded - IMAGE_FACTOR))
        bucket_resos.append((width_rounded + IMAGE_FACTOR, height_rounded))
        bucket_resos.append((width_rounded, height_rounded + IMAGE_FACTOR))

    # Min/max area filtering
    bucket_resos = [
        (w, h) for w, h in bucket_resos if w * h >= min_area and w * h <= max_area
    ]

    # Select bucket which has the nearest aspect ratio
    aspect_ratio = width / height
    bucket_resos.sort(key=lambda x: abs((x[0] / x[1]) - aspect_ratio))
    bucket_reso = bucket_resos[0]

    # Resize to bucket
    image_np = resize_image_to_bucket(image, bucket_reso)

    # Convert back to PIL
    image = Image.fromarray(image_np)
    return image


def preprocess_image_v1(
    img: Image.Image,
    upscale_processor: Swin2SRImageProcessor,
    upscale_model: Swin2SRForImageSuperResolution,
    imgwidth: int,
    imgheight: int,
) -> Image.Image:
    # Open and process image (sharpening happens inside sharpen_image)
    img = sharpen_image(img, upscale_processor, upscale_model)

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
    padded_img = Image.new(
        "RGB", (imgwidth, imgheight), (255, 255, 255)
    )  # White background

    # Calculate position to center the image
    x_offset = (imgwidth - new_width) // 2
    y_offset = (imgheight - new_height) // 2

    # Paste resized image onto centered position
    padded_img.paste(resized_img, (x_offset, y_offset))

    return padded_img

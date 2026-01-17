import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple
from utils import torch_device, torch_dtype
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor


def load_upscaler_model() -> Tuple[Swin2SRImageProcessor, Swin2SRForImageSuperResolution] | None:
    """Load the DiffusionPipeline for upscaling."""
    
    repo_id="caidas/swin2SR-classical-sr-x2-64"
    print(f"Loading upscale({repo_id}) model...")
    
    try:
        upscale_processor: Swin2SRImageProcessor = Swin2SRImageProcessor.from_pretrained(
            repo_id, trust_remote_code=True)
        upscale_model: Swin2SRForImageSuperResolution = Swin2SRForImageSuperResolution.from_pretrained(
            repo_id, trust_remote_code=True, torch_dtype=torch_dtype
        )
        upscale_model = upscale_model.to(torch_device)
        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")
        return upscale_processor, upscale_model
    except Exception as e:
        print(f"Warning: Could not load upscaler model. Skipping upscaling. Error: {e}")
        # Return a mock object or None if upscaler is critical
        return None


def sharpen_image(img: Image.Image, upscale_processor: Swin2SRImageProcessor,
                  upscale_model: Swin2SRForImageSuperResolution) -> Image.Image:
    """Sharpens image using Upscaler-Ultra"""

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    if upscale_processor or upscale_model is None:
        return img

    input_image = img

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


def preprocess_image(
    img: Image.Image,
    upscale_processor: Swin2SRImageProcessor,
    upscale_model: Swin2SRForImageSuperResolution,
    imgwidth: int,
    imgheight: int) -> Image.Image:
    """Copy image to target directory with format conversion to PNG and padding to fit dimensions."""

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
    padded_img = Image.new('RGB', (imgwidth, imgheight), (255, 255, 255))  # White background

    # Calculate position to center the image
    x_offset = (imgwidth - new_width) // 2
    y_offset = (imgheight - new_height) // 2

    # Paste resized image onto centered position
    padded_img.paste(resized_img, (x_offset, y_offset))

    return padded_img
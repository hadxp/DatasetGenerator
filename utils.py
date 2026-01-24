import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, TypedDict

torch_dtype = torch.float32
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
video_extensions = {'.mp4', '.webm', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.m4v'}

class ResultEntry(TypedDict):
    video: List[np.ndarray]
    image: Image.Image
    control: Image.Image
    caption: str
    
class SaveImageEntry(TypedDict):
    """
    image_path: str
    control_image: str → target_image (how image changes, general starting image or the image itself)
    caption: str"""
    image_path: str
    control_path: str
    caption: str

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from directory."""
    
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    return sorted(image_files)

def get_video_files(directory: Path) -> List[Path]:
    """Get all video files from directory."""
    video_files = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)

    return sorted(video_files)

def save_images(results: List[ResultEntry], target_path: Path) -> List[SaveImageEntry]:
    """Save images to directory."""
    save_image_entries = []
    # Create target directory if it does not exist
    target_path.mkdir(parents=True, exist_ok=True)
    for i, result in enumerate(results, 1):
        img = result["image"]
        caption = result["caption"]
        save_path = target_path / f"{i}.png"
        # Check if the target image already exists
        if save_path.exists():
            # print(f"  Warning: {target_filename} already exists, skipping copy")
            pass
        else:
            # Save as PNG
            img.save(save_path, 'PNG')
            # print(f"  ✓ Copied image {target_path.name}")
        save_image_entry = SaveImageEntry(image_path=str(save_path), control_path=str(save_path), caption=caption)
        save_image_entries.append(save_image_entry)
    return save_image_entries
import av
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
    image_path: str
    control_path: str
    caption: str


class InMemoryResultEntry(TypedDict):
    video: List[np.ndarray]
    image: Image.Image
    control_image: Image.Image
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

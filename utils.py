import torch
from PIL import Image
from pathlib import Path
from typing import List, TypedDict

torch_dtype = torch.float32
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

class ResultEntry(TypedDict):
    image_path: str
    control_path: str
    caption: str


class InMemoryResultEntry(TypedDict):
    image: Image.Image
    control_image: Image.Image
    caption: str

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    return sorted(image_files)
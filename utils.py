import os
import av
import torch
import logging
import warnings
import numpy as np
from PIL import Image
from pathlib import Path

from VideoInfo import VideoInfo
from typing import List, TypedDict, Tuple

working_dir: Path = Path.cwd()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch_dtype = torch.float32
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# attention configuration

# FlashAttention (must be installed)
torch.backends.cuda.enable_flash_sdp(True)
# xformers or native implementation (xformers must be installed)
torch.backends.cuda.enable_mem_efficient_sdp(True)
# standard PyTorch implementation (slow and uses a lot of memory)
torch.backends.cuda.enable_math_sdp(True)

# Suppress library logging
logging.getLogger("basicsr").setLevel(logging.WARNING)
logging.getLogger("realesrgan").setLevel(logging.WARNING)
logging.getLogger("gfpgan").setLevel(logging.WARNING)
# Suppress all torchvision warnings
warnings.filterwarnings("ignore", module="torchvision")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
video_extensions = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".m4v"}
caption_extensions = {".txt"}

class ResultEntry(TypedDict):
    file_path_in_target_dir: Path
    video: Tuple[List[Image.Image], VideoInfo]
    image: Image.Image
    control: Image.Image
    caption: str


class SaveImageEntry(TypedDict):
    """
    image_path: str
    control_image: str → target_image (starting image)
    caption: str
    """

    image_path: str
    control_path: str
    caption: str


class SaveVideoEntry(TypedDict):
    """
    video_path: str
    caption: str
    """

    video_path: str
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


def save_video(
    path: str, video: Tuple[List[np.ndarray], VideoInfo], max_fps=30
) -> None:
    # Open container for writing
    container = av.open(path, mode="w")

    frames, video_info = video

    int_fps = int(round(video_info.fps))
    if int_fps > max_fps:
        int_fps = max_fps

    # Create video stream
    stream = container.add_stream(video_info.codec, rate=int_fps)
    stream.width = video_info.width
    stream.height = video_info.height
    stream.pix_fmt = "yuv420p"  # required by most codecs

    for frame in frames:
        # Convert numpy → AVFrame
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

        # Encode the frame
        for packet in stream.encode(av_frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()


def save(
    results: List[ResultEntry], target_path: Path
) -> List[SaveImageEntry | SaveVideoEntry]:
    """Save images or videos to target directory."""
    save_entries = []
    # Create target directory if it does not exist
    target_path.mkdir(parents=True, exist_ok=True)
    for i, result in enumerate(results, 1):
        img = result["image"]
        video = result["video"]
        caption = result["caption"]
        file_path_in_target_dir = result["file_path_in_target_dir"]
        if img is not None:
            if not file_path_in_target_dir.exists():
                img.save(file_path_in_target_dir, "PNG")
            save_entry = SaveImageEntry(
                image_path=str(file_path_in_target_dir), control_path=str(file_path_in_target_dir), caption=caption
            )
        else:
            if not file_path_in_target_dir.exists():
                save_video(str(file_path_in_target_dir), video)
            save_entry = SaveVideoEntry(
                video_path=str(file_path_in_target_dir), caption=caption # no control is needed for videos
            )

        save_entries.append(save_entry)
    return save_entries


def get_cuda_free_memory_gb(device: torch.device = None) -> float:
    if device is None:
        device = torch.cuda.current_device()

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats["active_bytes.all.current"]
    bytes_reserved = memory_stats["reserved_bytes.all.current"]
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024**3)


def get_detailed_memory_usage(device: torch.device = None):
    """Detailed memory breakdown for debugging"""
    if device is None:
        device = torch.cuda.current_device()

    # Get PyTorch memory stats
    memory_stats = torch.cuda.memory_stats(device)

    active = memory_stats.get("active_bytes.all.current", 0) / (1024 ** 3)
    reserved = memory_stats.get("reserved_bytes.all.current", 0) / (1024 ** 3)

    # Get CUDA stats
    free_cuda, total_cuda = torch.cuda.mem_get_info(device)
    free_cuda_gb = free_cuda / (1024 ** 3)
    total_cuda_gb = total_cuda / (1024 ** 3)

    # Calculate usable
    inactive_reserved = reserved - active
    usable = free_cuda_gb + inactive_reserved

    print(f"GPU {device}:")
    print(f"  Total capacity: {total_cuda_gb:.2f} GB")
    print(f"  CUDA reported free: {free_cuda_gb:.2f} GB")
    print(f"  PyTorch reserved: {reserved:.2f} GB")
    print(f"  PyTorch active: {active:.2f} GB")
    print(f"  PyTorch inactive (cache): {inactive_reserved:.2f} GB")
    print(f"  → Usable memory: {usable:.2f} GB")

    return {
        'total': total_cuda_gb,
        'cuda_free': free_cuda_gb,
        'pytorch_reserved': reserved,
        'pytorch_active': active,
        'pytorch_cache': inactive_reserved,
        'usable': usable
    }
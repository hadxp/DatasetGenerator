import av
import torch
from PIL import Image
from pathlib import Path
from typing import List, TypedDict, Tuple
from numpy_pil_conv import pil_to_numpy
from VideoInfo import VideoInfo

torch_dtype = torch.float32
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
video_extensions = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".m4v"}
caption_extensions = {".txt"}


class ResultEntry(TypedDict):
    video: Tuple[List[Image.Image], VideoInfo]
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


class SaveVideoEntry(TypedDict):
    """
    video_path: str
    control_image: str → target_image (how image changes, general starting image or the image itself)
    caption: str"""

    video_path: str
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

def save_video(path: str, video: Tuple[List[Image.Image], VideoInfo], max_fps=30) -> None:
    # Open container for writing
    container = av.open(path, mode="w")
    
    frames, video_info = video

    int_fps = int(round(video_info.fps))
    if int_fps > max_fps:
        int_fps = max_fps

    # Create video stream
    stream = container.add_stream(video_info.codec, rate=int_fps)
    stream.width = frames[0].width
    stream.height = frames[0].height
    stream.pix_fmt = "yuv420p"  # required by most codecs

    for img in frames:
        frame = pil_to_numpy(img)

        # Convert numpy → AVFrame
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

        # Encode the frame
        for packet in stream.encode(av_frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    

def save_images(results: List[ResultEntry], target_path: Path) -> List[SaveImageEntry | SaveVideoEntry]:
    """Save images to directory."""
    save_entries = []
    # Create target directory if it does not exist
    target_path.mkdir(parents=True, exist_ok=True)
    for i, result in enumerate(results, 1):
        img = result["image"]
        video = result["video"]
        caption = result["caption"]
        if img is not None:
            save_path = target_path / f"{i}.png"
            if not save_path.exists():
                img.save(save_path, "PNG")
            save_entry = SaveImageEntry(
                image_path=str(save_path), control_path=str(save_path), caption=caption
            )
        else:
            save_path = target_path / f"{i}.mp4"
            if not save_path.exists():
                save_video(str(save_path), video)
            save_entry = SaveVideoEntry(
                video_path=str(save_path), control_path=str(save_path), caption=caption
            )
            
        save_entries.append(save_entry)
    return save_entries

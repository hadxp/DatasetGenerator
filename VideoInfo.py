import av
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video metadata container"""

    file_path: str
    duration: float  # in seconds
    fps: float
    width: int
    height: int
    total_frames: int
    codec: str
    format: str
    total_audio_streams: int
    audio_streams: List[av.audio.stream.AudioStream]
    audio_data: List[np.ndarray]
    audio_sample_rates: List[int]
    audio_channels: List[int]
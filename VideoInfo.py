from dataclasses import dataclass

@dataclass
class VideoInfo:
    """Video metadata container"""
    duration: float  # in seconds
    fps: float
    width: int
    height: int
    total_frames: int
    codec: str
    format: str
    audio_streams: int
import av
import sys

import numpy as np
from pathlib import Path
from VideoInfo import VideoInfo
from typing import List, Tuple, Union
from transformers import Swin2SRImageProcessor, Swin2SRForImageSuperResolution
from scripts.framerate_converter import interpolate_and_scale


class VideoFrameExtractor:
    """Extract frames from video files using PyAV"""

    def __init__(
        self,
        file_path: Union[str, Path],
        model: Swin2SRImageProcessor,
        processor: Swin2SRForImageSuperResolution,
    ):
        self.file_path = file_path
        self.upscale_model = model
        self.upscale_processor = processor

    def get_video(self) -> Tuple[List[np.ndarray], VideoInfo]:
        file_path = str(self.file_path)

        try:
            container = av.open(file_path)
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )

            if not video_stream:
                raise ValueError(f"No video stream found in {file_path}")

            # Get video properties
            fps = float(video_stream.average_rate) if video_stream.average_rate else 0
            duration = (
                float(video_stream.duration * video_stream.time_base)
                if video_stream.duration
                else 0
            )

            # Try to get total frames
            try:
                total_frames = video_stream.frames
            except:
                total_frames = (
                    int(duration * fps) if duration and fps else 0
                )  # Estimate total frames

            info = VideoInfo(
                duration=duration,
                fps=fps,
                width=video_stream.width,
                height=video_stream.height,
                total_frames=total_frames,
                codec=video_stream.codec_context.name,
                format=container.format.name,
                audio_streams=sum(1 for s in container.streams if s.type == "audio"),
            )

            frames: List[np.ndarray] = []

            # upscale each video frame
            for frame in container.decode(video=0):
                frame_img = frame.to_ndarray(format="rgb24")
                # pil_img = numpy_to_pil(frame_img)
                # img = preprocess_image(
                # pil_img,
                # self.upscale_processor,
                # self.upscale_model,
                # )
                frames.append(frame_img)

            container.close()

            return frames, info
        except Exception:
            raise


if __name__ == "__main__":
    args = sys.argv
    file_path = args[1]
    output_path = args[2]
    # upscale_processor, upscale_model = load_upscaler_model()
    # vfe = VideoFrameExtractor(file_path, upscale_model, upscale_processor)
    # video = vfe.get_video()
    interpolate_and_scale(file_path, output_path, 30, 0.5, 0.5)
    # save_video(output_path, video)

import av
import numpy as np
from pathlib import Path
from numpy_pil_conv import numpy_to_pil, pil_to_numpy
from image_preprocessor import preprocess_image
from VideoInfo import VideoInfo
from typing import List, Tuple, Union, Optional

video_extensions = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".m4v"}

class VideoFrameExtractor:
    """Extract frames from video files using PyAV"""

    def __init__(
        self,
        file_path: Union[str, Path],
        upsample: bool = False,
        upscale: bool = False,
    ):
        self.file_path = file_path
        self.upsample = upsample
        self.upscale = upscale

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
            except Exception:
                total_frames = (
                    int(duration * fps) if duration and fps else 0
                )  # Estimate total frames

            audio_streams: List[av.audio.stream.AudioStream] = [s for s in container.streams if s.type == "audio"]
            audio_data: List[np.ndarray] = []
            audio_sample_rates: List[int] = []
            audio_channels: List[int] = []

            # Extract audio data from each audio stream
            for audio_stream in audio_streams:
                # Collect audio frames for this stream
                stream_audio_data = []
                for frame in container.decode(audio_stream):
                    # Convert audio frame to numpy array
                    # PyAV audio frames can be converted to ndarray
                    if frame.format.is_planar:
                        # Planar audio - each channel as separate array
                        audio_array = frame.to_ndarray()
                        # Shape: (channels, samples)
                    else:
                        # Packed audio - interleaved channels
                        audio_array = frame.to_ndarray()
                        # Shape: (samples, channels) or (samples,)

                    stream_audio_data.append(audio_array)

                if stream_audio_data:
                    # Concatenate all audio frames for this stream
                    if frame.format.is_planar:
                        # For planar format, concatenate along samples axis (axis=1)
                        combined_audio = np.concatenate(stream_audio_data, axis=1)
                    else:
                        # For packed format, concatenate along samples axis (axis=0)
                        combined_audio = np.concatenate(stream_audio_data, axis=0)

                    audio_data.append(combined_audio)
                    audio_sample_rates.append(audio_stream.sample_rate)
                    audio_channels.append(audio_stream.channels)

            # Reset container to beginning for video extraction
            container.seek(0)

            info = VideoInfo(
                file_path=str(file_path),
                duration=duration,
                fps=fps,
                width=video_stream.width,
                height=video_stream.height,
                total_frames=total_frames,
                codec=video_stream.codec_context.name,
                format=container.format.name,
                total_audio_streams=sum(1 for s in container.streams if s.type == "audio"),
                audio_streams=audio_streams,
                audio_data=audio_data if audio_data else None,
                audio_sample_rates=audio_sample_rates if audio_sample_rates else None,
                audio_channels=audio_channels if audio_channels else None,
            )

            frames: List[np.ndarray] = []

            # upscale each video frame
            for frame in container.decode(video=0):
                frame_np = frame.to_ndarray(format="rgb24")
                pil_img = numpy_to_pil(frame_np)
                img = preprocess_image(
                    pil_img,
                    upsample=self.upsample,
                    upscale=self.upscale,
                    resize=False, # do not resize the frame
                )
                np_img = pil_to_numpy(img)
                frames.append(np_img)

            container.close()

            return frames, info
        except Exception:
            raise

    def save_video(
            self,
            frames: List[np.ndarray],
            info: VideoInfo,
            output_path: Union[str, Path],
            codec: str = "libx264",
            audio_codec: str = "aac",
            video_bitrate: Optional[str] = None,
            preset: str = "medium"
    ) -> None:
        """Save frames and audio to a video file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with av.open(str(output_path), 'w') as output_container:
            # Add video stream with proper configuration
            video_stream = output_container.add_stream(codec, rate=info.fps)
            video_stream.width = info.width
            video_stream.height = info.height
            video_stream.pix_fmt = 'yuv420p'
            video_stream.options = {'preset': preset}

            for frame_idx, frame_data in enumerate(frames):
                # Create frame
                frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')

                # Encode and mux
                for packet in video_stream.encode(frame):
                    output_container.mux(packet)

            # Flush video encoder
            for packet in video_stream.encode():
                output_container.mux(packet)

            # Handle audio if available
            if hasattr(info, 'audio_data') and info.audio_data:
                for audio_data, sample_rate, channels in zip(
                        info.audio_data,
                        info.audio_sample_rates,
                        info.audio_channels
                ):
                    # Add audio stream
                    audio_stream = output_container.add_stream(audio_codec, rate=sample_rate)
                    audio_stream.channels = channels

                    # Prepare audio data
                    if len(audio_data.shape) == 1:
                        audio_data = audio_data.reshape(1, -1)
                    elif audio_data.shape[0] > audio_data.shape[1]:
                        audio_data = audio_data.T

                    # Convert to int16 if needed
                    if audio_data.dtype != np.int16:
                        if audio_data.dtype in [np.float32, np.float64]:
                            audio_data = (audio_data * 32767).astype(np.int16)
                        else:
                            audio_data = audio_data.astype(np.int16)

                    # Write audio frames
                    num_channels, num_samples = audio_data.shape
                    samples_per_frame = 1024

                    for frame_start in range(0, num_samples, samples_per_frame):
                        frame_end = min(frame_start + samples_per_frame, num_samples)
                        frame_samples = audio_data[:, frame_start:frame_end]

                        # Pad if needed
                        if frame_samples.shape[1] < samples_per_frame:
                            pad_width = samples_per_frame - frame_samples.shape[1]
                            frame_samples = np.pad(frame_samples, ((0, 0), (0, pad_width)))

                        # Create audio frame
                        audio_frame = av.AudioFrame.from_ndarray(
                            frame_samples,
                            format='s16',
                            layout='stereo' if num_channels == 2 else 'mono'
                        )

                        # Set properties
                        audio_frame.sample_rate = sample_rate
                        audio_frame.time_base = audio_stream.time_base
                        audio_frame.pts = frame_start  # PTS in samples

                        # Encode and mux
                        for packet in audio_stream.encode(audio_frame):
                            output_container.mux(packet)

                    # Flush audio encoder
                    for packet in audio_stream.encode():
                        output_container.mux(packet)
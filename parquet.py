import av
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from io import BytesIO
from huggingface_hub import HfApi
from utils import ResultEntry
from datasets import Dataset, Features, Value

def create_parquet(dataset_dir_path: Path, data: List[ResultEntry]) -> Path:
    """
    Creates a dataset with columns "image_bytes" and "captions,
    saves a parquet file in the root directory of the dataset (parent folder of dataset folder)
    and uploads it to huggingface
    """
    dataset_name = dataset_dir_path.name.replace(" - ","--").replace(" ", "-")
    parquet_path = dataset_dir_path / f"{dataset_name}.parquet"
    print(f"Parquet name is '{parquet_path}'")
    
    if parquet_path.exists():
        return parquet_path
    else:
        bytes_column_name = "bytes"
        captions_column_name = "captions"
    
        byte_list: List[bytes] = []
        captions: List[str] = []
    
        for result_entry in data:
            b = None
            
            if result_entry["video"] is not None:
                video_frames: List[np.ndarray] = result_entry["video"]
                b = frames_to_bytes(video_frames)
                
            if result_entry["image"] is not None:
                image: Image.Image = result_entry["image"]
                b = image_to_bytes(image, format="PNG")

            if b is not None:
                caption = result_entry["caption"]
                #control = result_entry["control"]
                byte_list.append(b)
                captions.append(caption)
    
        # list of raw image bytes and captions
        dataset_dict = {
            bytes_column_name: byte_list,
            captions_column_name: captions
        }
    
        features = Features({
            bytes_column_name: Value("binary"),
            captions_column_name: Value("string")
        })
    
        ds = Dataset.from_dict(dataset_dict, features=features)
    
        # Save to Parquet
        ds.to_parquet(parquet_path)
        return parquet_path
        
def upload_to_hf(parquet_path: Path, huggingface_repoid: str, huggingface_token: str):
    if parquet_path.exists():
        api = HfApi()

        api.upload_file(
            path_or_fileobj=parquet_path,  # local file path
            path_in_repo=parquet_path.name,  # filename inside repo
            repo_id=huggingface_repoid,  # your dataset repo
            repo_type="dataset",
            token=huggingface_token
        )

def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buffer = BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()

def frames_to_bytes(frames: List[np.ndarray], fps=30, codec="libx264"):
    buffer = BytesIO()

    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream(codec, rate=fps)

    for frame in frames:
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        packet = stream.encode(video_frame)
        if packet:
            container.mux(packet)

    # flush encoder
    packet = stream.encode(None)
    if packet:
        container.mux(packet)

    container.close()
    return buffer.getvalue()

import os
from pathlib import Path
from typing import List
from PIL import Image
from io import BytesIO
from utils import InMemoryResultEntry

from datasets import Dataset, Features, Value
from huggingface_hub import HfApi

def create_parquet(dataset_dir_path: Path, data: List[InMemoryResultEntry]) -> Path:
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
        image_bytes_column_name = "image_bytes"
        captions_column_name = "captions"
    
        image_bytes: List[bytes] = []
        captions: List[str] = []
    
        for result_entry in data:
            image = result_entry["image"]
            control = result_entry["control_image"] # unused
            caption = result_entry["caption"]
    
            # read image bytes
            ib = image_to_bytes(image, format="PNG")
            image_bytes.append(ib)
            captions.append(caption)
    
        # list of raw image bytes and captions
        dataset_dict = {
            image_bytes_column_name: image_bytes,
            captions_column_name: captions
        }
    
        features = Features({
            image_bytes_column_name: Value("binary"),
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

import os

import numpy as np
import torch
from PIL import Image
from gfpgan import GFPGANer
from utils import working_dir
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.download_util import load_file_from_url

models = {
    "realesrgan": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        "dir": f"{working_dir / 'CodeFormer' / 'weights' / 'realesrgan'}",
        "filename": "RealESRGAN_x2plus.pth",
    },
    "gfpgan_RestoreFormer": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        "dir": f"{working_dir / 'weights'}",
        "filename": "RestoreFormer.pth",
    },
}

def check_ckpts():
    for model_entry_name, model_info in models.items():
        model_url = model_info["url"]
        model_dir = model_info["dir"]
        model_name = model_info["filename"]  # with extension
        full_model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(full_model_path):
            load_file_from_url(
                url=model_url, model_dir=model_dir, progress=True, file_name=model_name
            )


def set_realesrgan() -> RealESRGANer:
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    realesrgan_model_path = os.path.join(
        models["realesrgan"]["dir"], models["realesrgan"]["filename"]
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path=realesrgan_model_path,  # ../CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    restoreformer_path = os.path.join(
        models["gfpgan_RestoreFormer"]["dir"], models["gfpgan_RestoreFormer"]["filename"]
    )
    arch = (models["gfpgan_RestoreFormer"]["filename"]).replace(".pth", "")
    restorer = GFPGANer(
        model_path=restoreformer_path,
        upscale=1,
        arch=arch,
        channel_multiplier=2,
        bg_upsampler=upsampler)

    return restorer


def realesgan_upsample(img: Image.Image, upsampler_model: GFPGANer) -> Image.Image:
    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = upsampler_model.enhance(
        np.array(img),
        has_aligned=False,
        paste_back=True,
        weight=0.5 # Balanced between quality and identity
    )
    return restored_img
import os
import io
import torch
from PIL import Image
from gfpgan import GFPGANer
from utils import working_dir
from realesrgan import RealESRGANer
from contextlib import redirect_stdout
from numpy_pil_conv import numpy_to_pil, pil_to_numpy
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

models = {
    "realesrgan": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        "dir": f"{working_dir / 'weights'}",
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

_upsampler_model: GFPGANer = None

def load_upsampler() -> GFPGANer:
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2, # 1=better for removing artifacts or denoising, 2 = Better for preserving fine details
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
    
    global _upsampler_model
    _upsampler_model = restorer
    
    return restorer


def upsample(img: Image.Image) -> Image.Image:
    # Suppress RealESRGAN tile printing
    with redirect_stdout(io.StringIO()):
        if _upsampler_model is None:
            check_ckpts()
            load_upsampler()
        cropped_faces, restored_faces, restored_img = _upsampler_model.enhance(
            pil_to_numpy(img),
            has_aligned=False,
            paste_back=True,
            weight=0.5 # Balance between quality and identity
        )
        return numpy_to_pil(restored_img)
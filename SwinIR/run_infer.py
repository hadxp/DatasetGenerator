import os
import torch
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from ..utils import torch_device
from models.network_swinir import SwinIR


def improve_image_quality(img: Image.Image, model: SwinIR, scale: int = 4, window_size: int = 8) -> Image.Image:
    # read image
    img_lq = get_image(img)  # image to HWC-BGR, float32
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(torch_device)  # CHW-RGB to NCHW-RGB

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old * scale, :w_old * scale]

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8 / needed for pil RGB mode

    # Convert to PIL Image in RGB mode
    pil_image = Image.fromarray(output, mode='RGB')
    
    return pil_image

def load_restoration_model(scale: int = 4) -> SwinIR:
    # scale factor: 1, 2, 3, 4, 8 / default 4

    model_name: str = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    model_path: Path = Path.cwd() / "SwinIR" / "model_zoo" / model_name

    # set up model
    if model_path.exists():
        print(f'loading model from {model_path}')
    else:
        os.makedirs(model_path.parent, exist_ok=True)
        model_url: str = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
        r = requests.get(model_url, allow_redirects=True)
        print(f'downloading model {model_path}')
        open(model_path, 'wb').write(r.content)
    
    # larger model size; use '3conv' to save parameters and memory
    model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'
    pretrained_model = torch.load(str(model_path))
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)
    model.eval()
    model = model.to(torch_device, dtype=torch.bfloat16)
    return model, scale


def get_image(img: Image.Image):
    # Open image and convert to RGB
    pil_img = img
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # Convert to numpy array and normalize
    img_lq = np.array(pil_img, dtype=np.float32) / 255.

    return img_lq